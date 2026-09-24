#include <serving/router/router_server.hpp>

#include <chrono>
#include <stdexcept>

#include <httplib.h>
#include <nlohmann/json.hpp>

namespace serving::router
{

namespace
{
using Json = nlohmann::json;
}

RouterServer::RouterServer(RouterServerConfig config, const std::vector<ReplicaSpec> &replicas)
  : _config(std::move(config)),
    _sessions(_config.sessionTtlSeconds),
    _registry(RoutingConfig{_config.sessionTtlSeconds, _config.affinitySlack}, _sessions, replicas),
    _proxy(_registry, _registryMutex, _config.connectTimeoutSeconds, _config.requestTimeoutSeconds),
    _server(std::make_unique<httplib::Server>())
{
  registerRoutes();
}

RouterServer::~RouterServer() { stop(); }

void RouterServer::start()
{
  if (_config.port == 0)
  {
    const int port = _server->bind_to_any_port(_config.host.c_str());
    if (port <= 0) { throw std::runtime_error("router could not bind an ephemeral port"); }
    _boundPort = port;
  }
  else
  {
    if (!_server->bind_to_port(_config.host.c_str(), _config.port)) { throw std::runtime_error("router could not bind " + _config.host + ":" + std::to_string(_config.port)); }
    _boundPort = _config.port;
  }

  _running = true;
  if (_config.startHealthPoller)
  {
    _healthThread = std::thread([this] { healthLoop(); });
  }
  _serverThread = std::thread([this] { _server->listen_after_bind(); });
  _server->wait_until_ready();
}

void RouterServer::stop()
{
  if (!_running.exchange(false)) { return; }
  _wakeCv.notify_all();
  if (_healthThread.joinable()) { _healthThread.join(); }
  if (_server) { _server->stop(); }
  if (_serverThread.joinable()) { _serverThread.join(); }
}

void RouterServer::healthLoop()
{
  while (_running.load())
  {
    try
    {
      probeOnce();
      const std::lock_guard<std::mutex> lock(_registryMutex);
      _sessions.sweep();
    }
    catch (const std::exception &)
    {
      // The poller must outlive one bad cycle.
    }
    std::unique_lock<std::mutex> lock(_wakeMutex);
    _wakeCv.wait_for(lock, std::chrono::duration<double>(_config.healthIntervalSeconds), [this] { return !_running.load(); });
  }
}

void RouterServer::probeOnce()
{
  // Snapshot the specs so the probes themselves run without the lock.
  std::vector<ReplicaSpec> specs;
  {
    const std::lock_guard<std::mutex> lock(_registryMutex);
    for (const ReplicaState &state : _registry.states()) { specs.push_back(state.spec); }
  }

  for (const ReplicaSpec &spec : specs)
  {
    bool healthy = false;
    try
    {
      httplib::Client client(spec.host, spec.port);
      client.set_connection_timeout(static_cast<time_t>(_config.connectTimeoutSeconds), 0);
      client.set_read_timeout(static_cast<time_t>(_config.connectTimeoutSeconds), 0);
      // A replica still loading refuses the connection outright, while a 503
      // means the process is up but its engine is gone. Both are unroutable.
      const auto response = client.Get("/health");
      healthy             = response && response->status == 200;
    }
    catch (const std::exception &)
    {
      healthy = false;
    }

    const std::lock_guard<std::mutex> lock(_registryMutex);
    ReplicaState                     *state = _registry.state(spec.name);
    if (state == nullptr) { continue; }
    if (healthy)
    {
      state->failures = 0;
      _registry.setReady(spec.name, true);
      continue;
    }
    state->failures += 1;
    // One blip should not depin every session; recovery is immediate on the
    // first success.
    if (state->failures >= UNHEALTHY_THRESHOLD) { _registry.setReady(spec.name, false); }
  }
}

void RouterServer::registerRoutes()
{
  _server->Get("/health", [this](const httplib::Request &, httplib::Response &response) {
    Json replicas = Json::array();
    int  routable = 0;
    {
      const std::lock_guard<std::mutex> lock(_registryMutex);
      for (const ReplicaState &state : _registry.states())
      {
        replicas.push_back(Json{{"name", state.name()},
                                {"base_url", state.spec.baseUrl()},
                                {"ready", state.ready},
                                {"draining", state.draining},
                                {"outstanding", state.outstanding},
                                {"routed", state.routed}});
        if (state.routable()) { routable += 1; }
      }
      response.set_content(Json{{"status", routable > 0 ? "ok" : "no_replicas"},
                                {"routable", routable},
                                {"affinity_hits", _registry.affinityHits()},
                                {"rejected", _registry.rejected()},
                                {"replicas", replicas}}
                             .dump(),
                           "application/json");
    }
    response.status = routable > 0 ? 200 : 503;
  });

  _server->Get("/v1/models", [this](const httplib::Request &, httplib::Response &response) {
    // Answered by a routable replica, so the list reflects what is actually
    // servable rather than what is configured.
    ReplicaSpec target;
    bool        found = false;
    {
      const std::lock_guard<std::mutex> lock(_registryMutex);
      for (const ReplicaState &state : _registry.states())
      {
        if (state.routable())
        {
          target = state.spec;
          found  = true;
          break;
        }
      }
    }
    if (!found)
    {
      response.status = 503;
      response.set_content(Json{{"object", "error"}, {"message", "no serving replica is currently routable"}}.dump(), "application/json");
      return;
    }
    httplib::Client client(target.host, target.port);
    client.set_connection_timeout(static_cast<time_t>(_config.connectTimeoutSeconds), 0);
    const auto upstream = client.Get("/v1/models");
    if (!upstream)
    {
      response.status = 502;
      response.set_content(Json{{"object", "error"}, {"message", "replica " + target.name + " is unreachable"}}.dump(), "application/json");
      return;
    }
    response.status = upstream->status;
    response.set_content(upstream->body, "application/json");
  });

  const auto proxyTo = [this](const std::string &path) {
    return [this, path](const httplib::Request &request, httplib::Response &response) { _proxy.forward(request, response, path); };
  };
  _server->Post("/v1/completions", proxyTo("/v1/completions"));
  _server->Post("/v1/chat/completions", proxyTo("/v1/chat/completions"));
}

} // namespace serving::router
