#include <serving/infra/deployment_server.hpp>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <serving/infra/document.hpp>

namespace serving::infra
{

namespace
{

using json = nlohmann::ordered_json;

void sendJson(httplib::Response &response, const json &payload, int status = 200)
{
  response.status = status;
  response.set_content(payload.dump(), "application/json");
}

void sendError(httplib::Response &response, const std::string &message, int status) { sendJson(response, json{{"error", message}}, status); }

json describe(const Replica &replica, const ReplicaLoad *load)
{
  json entry{{"name", replica.name},
             {"partition", replica.partition},
             {"node", replica.node},
             {"devices", replica.devices},
             {"port", replica.port},
             {"phase", phaseName(replica.phase)},
             {"desired_phase", phaseName(replica.desiredPhase)}};
  // Null rather than zero when a replica did not answer: nobody should read a
  // silent replica as an idle one and drain it.
  if (load == nullptr) { entry["load"] = nullptr; }
  else { entry["load"] = json{{"pending_tokens", load->pendingTokens}, {"preemptions", load->preemptions}, {"outstanding", load->outstanding}}; }
  return entry;
}

} // namespace

DeploymentServer::DeploymentServer(DeploymentServerConfig config, StateMap &map, std::mutex &mapMutex, CapacityProvider &provider, ScalingConfig scaling)
  : _config(std::move(config)),
    _map(map),
    _mutex(mapMutex),
    _provider(provider),
    _scaling(scaling),
    _server(std::make_unique<httplib::Server>())
{
  _scaling.validate();
  registerRoutes();
}

DeploymentServer::~DeploymentServer() { stop(); }

LoadSnapshot DeploymentServer::collectLoad() const
{
  std::vector<Replica> targets;
  {
    const std::lock_guard<std::mutex> lock(_mutex);
    for (const Replica &replica : _map.replicas())
    {
      if (replica.phase == Phase::Active || replica.phase == Phase::Reserved) { targets.push_back(replica); }
    }
  }

  LoadSnapshot snapshot;
  for (const Replica &replica : targets)
  {
    std::string host = replica.node;
    {
      const std::lock_guard<std::mutex> lock(_mutex);
      if (const Node *node = _map.node(replica.node); node != nullptr && !node->host.empty()) { host = node->host; }
    }
    if (replica.port == 0) { continue; }

    httplib::Client client(host, replica.port);
    client.set_connection_timeout(static_cast<time_t>(_config.probeTimeoutSeconds), 0);
    client.set_read_timeout(static_cast<time_t>(_config.probeTimeoutSeconds), 0);

    const auto response = client.Get("/control/load");
    // A replica that does not answer contributes nothing rather than a zero:
    // see describe(), where the difference is kept visible.
    if (response == nullptr || response->status != 200) { continue; }

    try
    {
      const auto body = nlohmann::json::parse(response->body);
      snapshot.replicas.push_back(
        ReplicaLoad{replica.name, body.value("outstanding", 0), body.value("pending_tokens", 0), body.value("preemptions", 0)});
    }
    catch (const std::exception &)
    {
      continue;
    }
  }
  return snapshot;
}

void DeploymentServer::registerRoutes()
{
  _server->Get("/deployment", [this](const httplib::Request &, httplib::Response &response) {
    const auto load = collectLoad();

    const std::lock_guard<std::mutex> lock(_mutex);
    json                              nodes = json::array();
    for (const Node &node : _map.nodes())
    {
      nodes.push_back(json{{"name", node.name}, {"host", node.host}, {"devices", node.devices}, {"in_use", _map.devicesInUse(node.name)}, {"free", _map.devicesFree(node.name)}});
    }
    json replicas = json::array();
    for (const Replica &replica : _map.replicas()) { replicas.push_back(describe(replica, load.find(replica.name))); }

    sendJson(response, json{{"version", _map.version()}, {"nodes", std::move(nodes)}, {"replicas", std::move(replicas)}, {"problems", validate(_map)}});
  });

  _server->Get("/deployment/document", [this](const httplib::Request &, httplib::Response &response) {
    const std::lock_guard<std::mutex> lock(_mutex);
    response.status = 200;
    response.set_content(writeDocument(_map), "application/json");
  });

  // What the policy would do. Reported, never acted on: the decision is the
  // operator's, and an opinion they did not ask to be executed is advice.
  _server->Get("/deployment/advice", [this](const httplib::Request &, httplib::Response &response) {
    const auto load = collectLoad();

    const std::lock_guard<std::mutex> lock(_mutex);
    json                              advice = json::array();
    for (const ScalingDecision &decision : decideScaling(_map, load, _scaling))
    {
      advice.push_back(json{{"partition", decision.partition}, {"action", scalingActionName(decision.action)}, {"replica", decision.replica}, {"reason", decision.reason}});
    }
    sendJson(response, json{{"advice", std::move(advice)}});
  });

  _server->Post("/deployment/scale", [this](const httplib::Request &request, httplib::Response &response) {
    json body;
    try
    {
      body = json::parse(request.body);
    }
    catch (const std::exception &)
    {
      sendError(response, "request body is not valid JSON", 400);
      return;
    }

    const auto direction = body.value("direction", std::string{});
    if (direction != "up" && direction != "down")
    {
      sendError(response, "'direction' must be 'up' or 'down'", 400);
      return;
    }
    if (!body.contains("partition") || !body.at("partition").is_number_integer())
    {
      sendError(response, "'partition' is required", 400);
      return;
    }
    const auto partition = body.at("partition").get<PartitionId>();

    // Probed before the lock is taken: collectLoad dials every replica, and the
    // map's lock is the one the deployer's ticker needs. A drain also has to
    // know what each replica is carrying before it can pick the lightest.
    const LoadSnapshot load = direction == "down" ? collectLoad() : LoadSnapshot{};

    const std::lock_guard<std::mutex> lock(_mutex);
    if (_map.replicasIn(partition).empty())
    {
      sendError(response, "no partition " + std::to_string(partition) + " in this deployment", 404);
      return;
    }

    ScalingDecision decision;
    decision.partition = partition;
    decision.reason    = "asked for by an operator";

    if (direction == "up") { decision.action = ScalingAction::ScaleUp; }
    else
    {
      // Drain the least loaded of the ones that are actually serving: it has
      // the least in flight to finish, so the capacity comes back soonest. The
      // same helper the automatic policy uses, so an operator's scale-down and
      // the advice they were shown pick the same replica.
      decision.action       = ScalingAction::ScaleDown;
      const Replica *victim = leastLoadedServing(_map, partition, load);
      if (victim == nullptr)
      {
        sendError(response, "partition " + std::to_string(partition) + " has no serving replica to drain", 409);
        return;
      }
      decision.replica = victim->name;
    }

    const auto  outcomes = applyScaling(_map, _provider, {decision}, _scaling);
    const auto &outcome  = outcomes.front();
    if (outcome.applied) { _scaled.fetch_add(1); }

    sendJson(response,
             json{{"applied", outcome.applied},
                  {"action", scalingActionName(decision.action)},
                  {"partition", partition},
                  {"replica", outcome.replica},
                  {"declined", outcome.declined},
                  {"version", _map.version()}},
             outcome.applied ? 200 : 409);
  });
}

void DeploymentServer::start()
{
  if (_config.port == 0)
  {
    // bind_to_port(host, 0) reports success without saying which port the
    // kernel chose; bind_to_any_port returns it.
    const int port = _server->bind_to_any_port(_config.host.c_str());
    if (port <= 0) { throw std::runtime_error("deployment server could not bind " + _config.host + " to an ephemeral port"); }
    _boundPort.store(port);
  }
  else
  {
    if (!_server->bind_to_port(_config.host.c_str(), _config.port))
    {
      throw std::runtime_error("deployment server could not bind " + _config.host + ":" + std::to_string(_config.port));
    }
    _boundPort.store(_config.port);
  }

  _thread = std::thread([this] { _server->listen_after_bind(); });
  _server->wait_until_ready();
}

void DeploymentServer::stop()
{
  if (_server) { _server->stop(); }
  if (_thread.joinable()) { _thread.join(); }
}

} // namespace serving::infra
