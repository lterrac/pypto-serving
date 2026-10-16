#include <serving/deployer/deployment.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <thread>
#include <utility>

#include <nlohmann/json.hpp>

#include <serving/infra/document.hpp>
#include <serving/util/signals.hpp>

namespace serving::deployer
{

namespace
{

using json = nlohmann::json;

/// Where in the file a message is about. Mirrors the coordinator's config
/// reader: a configuration that is silently half-read is how a deployment ends
/// up launching something nobody intended.
class Context
{
  public:

  Context(std::string source, std::string where)
    : _source(std::move(source)),
      _where(std::move(where))
  {}

  [[nodiscard]] Context at(const std::string &where) const { return Context(_source, _where.empty() ? where : _where + "." + where); }

  [[nodiscard]] ConfigError error(const std::string &message) const { return ConfigError(_source + (_where.empty() ? "" : ": " + _where) + ": " + message); }

  private:

  std::string _source;
  std::string _where;
};

void rejectUnknownKeys(const json &object, const std::set<std::string> &allowed, const Context &ctx)
{
  for (const auto &entry : object.items())
  {
    if (allowed.count(entry.key()) == 0) { throw ctx.error("unknown key '" + entry.key() + "'"); }
  }
}

std::string stringAt(const json &object, const char *key, const Context &ctx, const std::string &fallback)
{
  const auto it = object.find(key);
  if (it == object.end()) { return fallback; }
  if (!it->is_string()) { throw ctx.error(std::string("'") + key + "' must be a string"); }
  return it->get<std::string>();
}

double numberAt(const json &object, const char *key, const Context &ctx, double fallback, double low)
{
  const auto it = object.find(key);
  if (it == object.end()) { return fallback; }
  if (!it->is_number()) { throw ctx.error(std::string("'") + key + "' must be a number"); }
  const double value = it->get<double>();
  if (value < low) { throw ctx.error(std::string("'") + key + "' must be at least " + std::to_string(low)); }
  return value;
}

int integerAt(const json &object, const char *key, const Context &ctx, int fallback, int low)
{
  const auto it = object.find(key);
  if (it == object.end()) { return fallback; }
  if (!it->is_number_integer()) { throw ctx.error(std::string("'") + key + "' must be an integer"); }
  const int value = it->get<int>();
  if (value < low) { throw ctx.error(std::string("'") + key + "' must be at least " + std::to_string(low)); }
  return value;
}

bool booleanAt(const json &object, const char *key, const Context &ctx, bool fallback)
{
  const auto it = object.find(key);
  if (it == object.end()) { return fallback; }
  if (!it->is_boolean()) { throw ctx.error(std::string("'") + key + "' must be true or false"); }
  return it->get<bool>();
}

/// `{ "host": ..., "port": ... }`, used by both servers.
void parseEndpoint(const json &object, const Context &ctx, std::string &host, int &port)
{
  if (!object.is_object()) { throw ctx.error("must be a JSON object"); }
  rejectUnknownKeys(object, {"host", "port"}, ctx);
  host = stringAt(object, "host", ctx, host);
  port = integerAt(object, "port", ctx, port, 0);
}

} // namespace

DeployerConfig parseDeployerConfig(const std::string &text, const std::string &source)
{
  const Context ctx(source, "");

  json document;
  try
  {
    document = json::parse(text);
  }
  catch (const json::exception &e)
  {
    throw ctx.error(std::string("not valid JSON: ") + e.what());
  }
  if (!document.is_object()) { throw ctx.error("must be a JSON object"); }

  rejectUnknownKeys(document,
                    {"document",
                     "document_out",
                     "persist_document",
                     "launch",
                     "node",
                     "remote",
                     "routing_mode",
                     "routing",
                     "partition",
                     "ingress",
                     "operator_api",
                     "ready_timeout_seconds",
                     "service_interval_seconds",
                     "backoff",
                     "monitor",
                     "scaling",
                     "connect_timeout_seconds",
                     "request_timeout_seconds",
                     "max_attempts"},
                    ctx);

  DeployerConfig config;

  const std::string documentPath = stringAt(document, "document", ctx, "");
  if (documentPath.empty()) { throw ctx.error("'document' is required: a deployment without one is a deployment nobody wrote down"); }
  config.document    = documentPath;
  config.documentOut = stringAt(document, "document_out", ctx, documentPath);
  config.persist     = booleanAt(document, "persist_document", ctx, true);

  config.launch = stringAt(document, "launch", ctx, "");
  config.node   = stringAt(document, "node", ctx, "");

  if (const auto it = document.find("remote"); it != document.end())
  {
    const auto remoteCtx = ctx.at("remote");
    if (!it->is_object()) { throw remoteCtx.error("must be a JSON object"); }
    rejectUnknownKeys(*it, {"reach", "state_directory", "grace_seconds"}, remoteCtx);
    config.remote.reach          = stringAt(*it, "reach", remoteCtx, config.remote.reach);
    config.remote.stateDirectory = stringAt(*it, "state_directory", remoteCtx, config.remote.stateDirectory);
    config.remote.graceSeconds   = integerAt(*it, "grace_seconds", remoteCtx, config.remote.graceSeconds, 0);
  }

  const std::string modeText = stringAt(document, "routing_mode", ctx, "ingress");
  try
  {
    config.mode = router::parseRoutingMode(modeText);
  }
  catch (const std::invalid_argument &e)
  {
    throw ctx.error(std::string("'routing_mode': ") + e.what());
  }

  if (const auto it = document.find("routing"); it != document.end())
  {
    const auto routingCtx = ctx.at("routing");
    if (!it->is_object()) { throw routingCtx.error("must be a JSON object"); }
    rejectUnknownKeys(*it, {"session_ttl_seconds", "affinity_slack"}, routingCtx);
    config.routing.sessionTtlSeconds = numberAt(*it, "session_ttl_seconds", routingCtx, config.routing.sessionTtlSeconds, 0.0);
    config.routing.affinitySlack     = integerAt(*it, "affinity_slack", routingCtx, config.routing.affinitySlack, 0);
  }

  config.partition = static_cast<router::PartitionId>(integerAt(document, "partition", ctx, 0, 0));

  if (const auto it = document.find("ingress"); it != document.end()) { parseEndpoint(*it, ctx.at("ingress"), config.ingressHost, config.ingressPort); }
  if (const auto it = document.find("operator_api"); it != document.end()) { parseEndpoint(*it, ctx.at("operator_api"), config.operatorHost, config.operatorPort); }

  config.readyTimeoutSeconds    = numberAt(document, "ready_timeout_seconds", ctx, config.readyTimeoutSeconds, 1.0);
  config.serviceIntervalSeconds = numberAt(document, "service_interval_seconds", ctx, config.serviceIntervalSeconds, 0.1);
  config.connectTimeoutSeconds  = numberAt(document, "connect_timeout_seconds", ctx, config.connectTimeoutSeconds, 0.1);
  config.requestTimeoutSeconds  = numberAt(document, "request_timeout_seconds", ctx, config.requestTimeoutSeconds, 1.0);
  config.maxAttempts            = integerAt(document, "max_attempts", ctx, config.maxAttempts, 1);

  if (const auto it = document.find("backoff"); it != document.end())
  {
    const auto backoffCtx = ctx.at("backoff");
    if (!it->is_object()) { throw backoffCtx.error("must be a JSON object"); }
    rejectUnknownKeys(*it, {"base_seconds", "max_seconds", "give_up_after"}, backoffCtx);
    config.backoff =
      infra::Backoff(numberAt(*it, "base_seconds", backoffCtx, 5.0, 0.0), numberAt(*it, "max_seconds", backoffCtx, 60.0, 0.0), integerAt(*it, "give_up_after", backoffCtx, 3, 1));
  }

  if (const auto it = document.find("monitor"); it != document.end())
  {
    const auto monitorCtx = ctx.at("monitor");
    if (!it->is_object()) { throw monitorCtx.error("must be a JSON object"); }
    rejectUnknownKeys(*it, {"failures_before_dead"}, monitorCtx);
    config.monitor.failuresBeforeDead = integerAt(*it, "failures_before_dead", monitorCtx, config.monitor.failuresBeforeDead, 1);
  }

  if (const auto it = document.find("scaling"); it != document.end())
  {
    const auto scalingCtx = ctx.at("scaling");
    if (!it->is_object()) { throw scalingCtx.error("must be a JSON object"); }
    rejectUnknownKeys(*it, {"scale_up_above", "scale_down_below", "min_replicas", "max_replicas", "preemptions_above", "devices_per_replica"}, scalingCtx);
    config.scaling.scaleUpAbove      = numberAt(*it, "scale_up_above", scalingCtx, config.scaling.scaleUpAbove, 0.0);
    config.scaling.scaleDownBelow    = numberAt(*it, "scale_down_below", scalingCtx, config.scaling.scaleDownBelow, 0.0);
    config.scaling.minReplicas       = integerAt(*it, "min_replicas", scalingCtx, config.scaling.minReplicas, 0);
    config.scaling.maxReplicas       = integerAt(*it, "max_replicas", scalingCtx, config.scaling.maxReplicas, 1);
    config.scaling.preemptionsAbove  = integerAt(*it, "preemptions_above", scalingCtx, config.scaling.preemptionsAbove, 0);
    config.scaling.devicesPerReplica = integerAt(*it, "devices_per_replica", scalingCtx, config.scaling.devicesPerReplica, 1);
  }

  try
  {
    config.scaling.validate();
    config.monitor.validate();
  }
  catch (const std::invalid_argument &e)
  {
    throw ctx.error(e.what());
  }

  return config;
}

DeployerConfig loadDeployerConfig(const std::filesystem::path &path)
{
  std::ifstream file(path);
  if (!file) { throw ConfigError(path.string() + ": cannot be read"); }
  std::ostringstream text;
  text << file.rdbuf();

  auto config = parseDeployerConfig(text.str(), path.string());
  // A relative document path is relative to the configuration that names it,
  // not to whatever directory the deployer happens to be started from.
  if (config.document.is_relative()) { config.document = path.parent_path() / config.document; }
  if (config.documentOut.is_relative()) { config.documentOut = path.parent_path() / config.documentOut; }
  return config;
}

Deployment::Deployment(DeployerConfig config)
  : _config(std::move(config))
{}

Deployment::~Deployment() { stop(); }

void Deployment::start()
{
  infra::StateMap document = infra::loadDocument(_config.document);

  const auto problems = infra::validate(document);
  if (!problems.empty())
  {
    // Slide 6's "does this partition fit that node", answered before anything
    // is started rather than after.
    std::string message = _config.document.string() + ": the deployment is not coherent:";
    for (const auto &problem : problems) { message += "\n  " + problem; }
    throw ConfigError(message);
  }
  if (document.replicas().empty()) { throw ConfigError(_config.document.string() + ": no replicas"); }

  coordinator::CoordinatorConfig coordinatorConfig;
  coordinatorConfig.partition = _config.partition;
  coordinatorConfig.mode      = _config.mode;
  coordinatorConfig.routing   = _config.routing;

  std::vector<router::ReplicaSpec> specs;
  std::map<std::string, uint64_t>  instances;
  for (const infra::Replica &replica : document.replicas())
  {
    const infra::Node *node = document.node(replica.node);
    if (node == nullptr) { throw ConfigError(_config.document.string() + ": replica '" + replica.name + "' is on unknown node '" + replica.node + "'"); }

    coordinator::ReplicaBinding binding;
    binding.spec.name      = replica.name;
    binding.spec.partition = replica.partition;
    binding.spec.host      = node->host;
    binding.spec.port      = replica.port;
    binding.spec.scheme    = replica.scheme;
    binding.instanceId     = replica.instance;
    coordinatorConfig.replicas.push_back(binding);
    specs.push_back(binding.spec);
    instances.emplace(replica.name, replica.instance);
  }

  _transport   = std::make_unique<router::HttpRuleTransport>(specs, instances, router::HttpRuleTransportConfig{_config.connectTimeoutSeconds, _config.requestTimeoutSeconds});
  _coordinator = std::make_unique<coordinator::Coordinator>(std::move(coordinatorConfig), _transport.get());
  _coordinator->initialize();

  // `initialize()` seeds the coordinator's map from its configuration and marks
  // every replica active, which is a statement a coordinator-only deployment
  // can make -- its replicas already run. A deployer has to start them, so the
  // document's phases replace that seed before anything reads it. Safe without
  // the routing lock: nothing else is running yet.
  _coordinator->stateMap() = document;
  _coordinator->syncFromMap();

  if (!_config.launch.empty())
  {
    // The probe needs the map to turn a replica's node name into an address.
    // That works across machines already: a replica's health is an HTTP
    // request to wherever its node lives.
    _httpProbe = std::make_unique<infra::HttpHealthProbe>(&map(), _config.connectTimeoutSeconds);

    infra::ReplicaStarter *starter = nullptr;
    if (_config.node.empty())
    {
      _starter = std::make_unique<infra::ProcessStarter>(_config.launch);
      starter  = _starter.get();
      // A replica whose process has gone is failed however its endpoint looks,
      // which is the case a bare health check cannot tell from still loading.
      _probe = std::make_unique<infra::LiveAndReadyProbe>(*_starter, *_httpProbe);
    }
    else
    {
      _remoteStarter = std::make_unique<infra::RemoteStarter>(map(), _config.node, _config.launch, _config.remote);
      starter        = _remoteStarter.get();
      // No liveness pairing across the hop: for a remote replica the local pid
      // belongs to the reach command, and a reach that has exited says the
      // replica has too -- but a reach that is alive says only that ssh is.
      // The endpoint is the honest signal, and it is the same one the monitor
      // uses.
      _probe = std::make_unique<infra::HttpHealthProbe>(&map(), _config.connectTimeoutSeconds);
    }

    _supervisor = std::make_unique<infra::Supervisor>(map(), *starter, *_probe, _config.backoff);
    _supervisor->setReadyTimeout(_config.readyTimeoutSeconds);
  }
  else
  {
    // Nothing to launch: the replicas are someone else's, and this process only
    // coordinates and watches them.
    _probe = std::make_unique<infra::HttpHealthProbe>(&map(), _config.connectTimeoutSeconds);
  }

  _monitor  = std::make_unique<infra::HealthMonitor>(map(), *_probe, _config.monitor);
  _capacity = std::make_unique<infra::LocalCapacity>(map());

  infra::DeploymentServerConfig operatorConfig;
  operatorConfig.host = _config.operatorHost;
  operatorConfig.port = _config.operatorPort;
  _operator           = std::make_unique<infra::DeploymentServer>(operatorConfig, map(), *_capacity, _config.scaling);

  // The ingress registry holds the partition requests enter at: a pinned hop
  // that cannot be reached falls back to a sibling, and a sibling must be one
  // that serves the same stage of the pipeline.
  std::vector<router::ReplicaSpec> entry;
  for (const auto &spec : specs)
  {
    if (spec.partition == _config.partition) { entry.push_back(spec); }
  }
  if (entry.empty()) { throw ConfigError(_config.document.string() + ": no replica in partition " + std::to_string(_config.partition) + ", which is where requests enter"); }

  router::RouterServerConfig ingressConfig;
  ingressConfig.host                  = _config.ingressHost;
  ingressConfig.port                  = _config.ingressPort;
  ingressConfig.connectTimeoutSeconds = _config.connectTimeoutSeconds;
  ingressConfig.requestTimeoutSeconds = _config.requestTimeoutSeconds;
  ingressConfig.maxAttempts           = _config.maxAttempts;
  ingressConfig.sessionTtlSeconds     = _config.routing.sessionTtlSeconds;
  ingressConfig.affinitySlack         = _config.routing.affinitySlack;
  // One thing watching each replica. The deployer's monitor already probes
  // them and writes the result to the map, and `refreshRouting` takes the
  // ingress to that; a second poller racing it can only disagree.
  ingressConfig.startHealthPoller = false;
  _ingress                        = std::make_unique<router::RouterServer>(ingressConfig, entry, _coordinator.get());

  _operator->start();
  _ingress->start();
  _persistedVersion = map().version();
  _running          = true;
}

void Deployment::refreshRouting()
{
  if (_ingress == nullptr) { return; }

  struct Live
  {
    router::ReplicaSpec spec;
    infra::Phase        phase;
  };

  std::vector<Live> live;
  {
    const infra::StateMap &current = map();
    for (const infra::Replica &replica : current.replicas())
    {
      if (replica.partition != _config.partition) { continue; }
      const infra::Node *node = current.node(replica.node);
      if (node == nullptr) { continue; }
      live.push_back({router::ReplicaSpec{replica.name, replica.partition, node->host, replica.port, replica.scheme}, replica.phase});
    }
  }

  const std::lock_guard<std::mutex> lock(_ingress->registryMutex());
  auto                             &registry = _ingress->registry();
  for (const auto &entry : live)
  {
    // Added not ready: a replica this deployment just launched provably cannot
    // serve for the minutes its model takes to load.
    if (registry.state(entry.spec.name) == nullptr) { registry.add(entry.spec, false); }

    // Routability comes from the map, not from a second opinion. The ingress
    // has its own poller for the deployments where nothing else is watching,
    // but here the supervisor and the monitor are already probing these
    // replicas and writing what they find to the map -- and a replica the map
    // calls active while the ingress still calls it loading answers 503. The
    // phases map the same way the coordinator's own registries map them:
    // reserved is deliberately not routable, because routing to a spare makes
    // it an ordinary replica.
    registry.setReady(entry.spec.name, entry.phase == infra::Phase::Active);
    registry.setDraining(entry.spec.name, entry.phase == infra::Phase::Draining);
  }

  std::set<std::string> present;
  for (const auto &entry : live) { present.insert(entry.spec.name); }
  std::vector<std::string> gone;
  for (const auto &state : registry.states())
  {
    if (present.count(state.name()) == 0) { gone.push_back(state.name()); }
  }
  for (const auto &name : gone) { registry.remove(name); }
}

TickReport Deployment::service()
{
  TickReport report;
  if (!_running) { return report; }

  if (_supervisor != nullptr)
  {
    const auto supervision = _supervisor->service();
    report.started         = supervision.started;
    report.promoted        = supervision.promoted;
    report.retrying        = supervision.retrying;
    // Given up on is lost: the supervisor has already marked it dead, and the
    // coordinator has to hear that from somewhere.
    report.lost = supervision.gaveUp;
  }

  const auto health = _monitor->service();
  report.lost.insert(report.lost.end(), health.lost.begin(), health.lost.end());
  for (const auto &name : report.lost)
  {
    // Both halves: routing stops sending there, and the transport stops
    // dialling it -- a rule posted to a dead replica is a routing thread
    // waiting out a connect timeout for nothing.
    _coordinator->reportReplicaLost(name);
    _transport->onReplicaUnreachable(name);
  }
  for (const auto &name : health.recovered) { _transport->markReachable(name); }

  // Drains the losses reported above and brings routing to the map.
  _coordinator->service();
  const auto sync = _coordinator->syncFromMap();
  report.rephased = sync.rephased + sync.added + sync.removed;

  refreshRouting();

  const uint64_t version = map().version();
  if (_config.persist && version != _persistedVersion)
  {
    infra::saveDocument(map(), _config.documentOut);
    _persistedVersion = version;
    report.persisted  = true;
  }

  return report;
}

bool Deployment::settled() const
{
  for (const infra::Replica &replica : _coordinator->stateMap().replicas())
  {
    if (replica.phase != replica.desiredPhase) { return false; }
  }
  return true;
}

void Deployment::run(std::function<void()> waitForShutdown)
{
  const auto interval = std::chrono::duration<double>(_config.serviceIntervalSeconds);

  _ticking = true;
  std::thread ticker([this, interval] {
    while (_ticking.load())
    {
      const auto report = service();
      if (!report.empty())
      {
        for (const auto &name : report.started) { std::printf("[deployer] starting %s\n", name.c_str()); }
        for (const auto &name : report.promoted) { std::printf("[deployer] %s is serving\n", name.c_str()); }
        for (const auto &name : report.retrying) { std::printf("[deployer] %s failed to start; will retry\n", name.c_str()); }
        for (const auto &name : report.lost) { std::printf("[deployer] %s is gone; routing around it\n", name.c_str()); }
        std::fflush(stdout);
      }
      std::this_thread::sleep_for(std::chrono::duration_cast<std::chrono::milliseconds>(interval));
    }
  });

  if (waitForShutdown) { waitForShutdown(); }
  else { util::awaitShutdownSignal(); }
  std::printf("[deployer] shutting down\n");
  std::fflush(stdout);

  _ticking = false;
  ticker.join();
  stop();
}

void Deployment::stop()
{
  if (!_running) { return; }
  _running = false;

  if (_ingress != nullptr) { _ingress->stop(); }
  if (_operator != nullptr) { _operator->stop(); }

  // The replicas this process started are its to stop; ones it merely
  // coordinates are not.
  if (_starter != nullptr || _remoteStarter != nullptr)
  {
    for (const infra::Replica &replica : _coordinator->stateMap().replicas())
    {
      if (_starter != nullptr) { _starter->stop(replica); }
      else { _remoteStarter->stop(replica); }
    }
  }

  if (_coordinator != nullptr) { _coordinator->finalize(); }
}

int Deployment::ingressPort() const { return _ingress == nullptr ? 0 : _ingress->boundPort(); }
int Deployment::operatorPort() const { return _operator == nullptr ? 0 : _operator->boundPort(); }

} // namespace serving::deployer
