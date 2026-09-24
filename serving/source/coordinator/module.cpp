#include <serving/coordinator/module.hpp>

#include <stdexcept>

namespace serving::coordinator
{

Coordinator::Coordinator(CoordinatorConfig config, router::RuleTransport *transport)
  : _config(std::move(config)),
    _transport(transport)
{}

void Coordinator::initialize()
{
  if (_config.replicas.empty()) { throw std::invalid_argument("a coordinator needs at least one replica"); }

  std::vector<router::ReplicaSpec> specs;
  specs.reserve(_config.replicas.size());
  for (const ReplicaBinding &binding : _config.replicas)
  {
    specs.push_back(binding.spec);
    _instanceByName[binding.spec.name] = binding.instanceId;
  }

  // Built in initialize(), not the constructor: modules are constructed before
  // the deployment is known.
  _sessions = std::make_unique<router::SessionDirectory>(_config.routing.sessionTtlSeconds);
  _planner  = std::make_unique<router::RoutingPlanner>(_config.routing, *_sessions, specs);
  _strategy = router::makeRoutingStrategy(_config.mode, *_planner, _transport);
}

router::RoutingPath Coordinator::routeFor(const std::string &sessionId)
{
  if (_strategy == nullptr) { throw std::runtime_error("coordinator is not initialized"); }
  return _strategy->onRequest(sessionId);
}

void Coordinator::reportReplicaLost(const std::string &replicaName)
{
  const std::lock_guard<std::mutex> lock(_mutex);
  _pendingLosses.push_back(replicaName);
}

void Coordinator::service()
{
  if (_strategy == nullptr) { return; }

  std::deque<std::string> losses;
  {
    const std::lock_guard<std::mutex> lock(_mutex);
    losses.swap(_pendingLosses);
  }

  // Applied on the service tick, so a heartbeat signal never runs routing policy
  // on the reporting thread.
  for (const std::string &replicaName : losses)
  {
    _strategy->onReplicaLost(replicaName);
    _lossesApplied += 1;
  }
}

void Coordinator::finalize()
{
  _strategy.reset();
  _planner.reset();
  _sessions.reset();
  _instanceByName.clear();
}

std::optional<uint64_t> Coordinator::instanceFor(const std::string &replicaName) const
{
  const auto it = _instanceByName.find(replicaName);
  if (it == _instanceByName.end()) { return std::nullopt; }
  return it->second;
}

router::RoutingStats Coordinator::stats() const { return _strategy == nullptr ? router::RoutingStats{} : _strategy->stats(); }

} // namespace serving::coordinator
