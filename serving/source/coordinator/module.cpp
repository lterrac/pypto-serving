#include <serving/coordinator/module.hpp>

#include <serving/coordinator/map_routing.hpp>

#include <set>

#include <stdexcept>

namespace serving::coordinator
{

Coordinator::Coordinator(CoordinatorConfig config, router::RuleTransport *transport)
  : _config(std::move(config)),
    _transport(transport)
{}

void Coordinator::initialize()
{
  const std::lock_guard<std::mutex> lock(_routingMutex);
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

  // The map starts describing what the configuration already said, so routing
  // and the map agree before anything edits either. A configured replica is
  // active: the file is a statement that it should be serving.
  for (const ReplicaBinding &binding : _config.replicas)
  {
    infra::Replica replica;
    replica.name      = binding.spec.name;
    replica.partition = binding.spec.partition;
    replica.node      = binding.spec.host;
    replica.port      = binding.spec.port;
    replica.scheme    = binding.spec.scheme;
    replica.instance  = binding.instanceId;
    replica.phase     = infra::Phase::Active;
    _stateMap.addReplica(replica);
    // One node per distinct address, so `bindingsFrom` can resolve an endpoint.
    // A deployment that knows its real nodes replaces these.
    if (_stateMap.node(binding.spec.host) == nullptr) { _stateMap.addNode(infra::Node{binding.spec.host, binding.spec.host, {}}); }
  }
  _syncedVersion = _stateMap.version();
}

uint64_t Coordinator::syncedVersion() const
{
  const std::lock_guard<std::mutex> lock(_routingMutex);
  return _syncedVersion;
}

SyncReport Coordinator::syncFromMap()
{
  SyncReport report;
  {
    const std::lock_guard<std::mutex> lock(_routingMutex);
    report = syncLocked();
  }
  // A replica dropped from the map revokes the sessions that ran through it,
  // which is transport work and must not happen under the routing lock.
  flushRouting();
  return report;
}

// Caller holds _routingMutex.
SyncReport Coordinator::syncLocked()
{
  SyncReport report;
  if (_planner == nullptr || _strategy == nullptr) { return report; }

  _syncedVersion = _stateMap.version();

  std::set<std::string> inMap;
  for (const infra::Replica &replica : _stateMap.replicas())
  {
    inMap.insert(replica.name);

    router::ReplicaRegistry *registry = _planner->registry(replica.partition);
    if (registry == nullptr)
    {
      // Routing was built for a fixed set of partitions; a replica in a new one
      // needs the pipeline rebuilt, which is repartitioning and not this.
      report.unplaceable.push_back(replica.name);
      continue;
    }

    if (replica.phase == infra::Phase::Dead)
    {
      // The loss path, not a flag: paths through it are revoked and its
      // sessions released, the same as a health signal reporting it gone.
      if (registry->state(replica.name) != nullptr)
      {
        _strategy->onReplicaLost(replica.name);
        report.lost.push_back(replica.name);
      }
      continue;
    }

    const infra::Node *node = _stateMap.node(replica.node);
    if (node == nullptr) { continue; }

    if (registry->state(replica.name) == nullptr)
    {
      router::ReplicaSpec spec;
      spec.name      = replica.name;
      spec.partition = replica.partition;
      spec.host      = node->host;
      spec.port      = replica.port;
      spec.scheme    = replica.scheme;
      registry->add(spec, replica.phase == infra::Phase::Active);
      _instanceByName[replica.name] = replica.instance;
      report.added += 1;
    }

    const router::ReplicaState *before      = registry->state(replica.name);
    const bool                  wasRoutable = before != nullptr && before->routable();

    // Reserved is deliberately not routable: it is loaded and waiting, and
    // routing to it would make it an ordinary replica rather than a spare.
    registry->setReady(replica.name, replica.phase != infra::Phase::Pending && replica.phase != infra::Phase::Reserved);
    registry->setDraining(replica.name, replica.phase == infra::Phase::Draining);

    const router::ReplicaState *after = registry->state(replica.name);
    if (after != nullptr && after->routable() != wasRoutable) { report.rephased += 1; }
  }

  // Gone from the map is gone from routing.
  for (const router::PartitionId partition : _planner->partitions())
  {
    router::ReplicaRegistry *registry = _planner->registry(partition);
    if (registry == nullptr) { continue; }
    std::vector<std::string> stale;
    for (const router::ReplicaState &state : registry->states())
    {
      if (inMap.count(state.name()) == 0) { stale.push_back(state.name()); }
    }
    for (const std::string &name : stale)
    {
      _strategy->onReplicaLost(name);
      // Actually taken out, not just marked unready. onReplicaLost only clears
      // the ready flag, so the entry stayed in the registry for ever: every
      // later tick found the same ghost and counted it removed again, and a
      // replacement replica registered under the same name was skipped by the
      // `state(name) == nullptr` test above -- keeping the dead host and port
      // and losing its instance id, so channel rules for it silently stopped
      // being published.
      registry->remove(name);
      _instanceByName.erase(name);
      report.removed += 1;
    }
  }
  return report;
}

std::optional<router::RoutingPath> Coordinator::peekRoute(const std::string &sessionId)
{
  const std::lock_guard<std::mutex> lock(_routingMutex);
  if (_strategy == nullptr) { return std::nullopt; }
  return _strategy->peek(sessionId);
}

void Coordinator::flushRouting()
{
  // The flush lock is taken first and held across the strategy lookup, so a
  // concurrent finalize() cannot destroy the strategy between reading the
  // pointer and sending through it.
  const std::lock_guard<std::mutex> flushLock(_flushMutex);
  router::RoutingStrategy          *strategy = nullptr;
  {
    const std::lock_guard<std::mutex> lock(_routingMutex);
    strategy = _strategy.get();
  }
  if (strategy != nullptr) { strategy->flush(); }
}

router::RoutingPath Coordinator::routeFor(const std::string &sessionId)
{
  router::RoutingPath path;
  {
    const std::lock_guard<std::mutex> lock(_routingMutex);
    if (_strategy == nullptr) { throw std::runtime_error("coordinator is not initialized"); }
    path = _strategy->onRequest(sessionId);
  }
  // Publishing is a blocking round trip per hop. Held under the routing lock it
  // meant the router could plan exactly one session at a time, and one hop that
  // accepted a connection without answering stalled every other request's
  // routing for its whole timeout.
  flushRouting();
  return path;
}

void Coordinator::reportReplicaLost(const std::string &replicaName)
{
  const std::lock_guard<std::mutex> lock(_mutex);
  _pendingLosses.push_back(replicaName);
}

void Coordinator::service()
{
  std::deque<std::string> losses;
  {
    const std::lock_guard<std::mutex> lock(_mutex);
    losses.swap(_pendingLosses);
  }

  if (losses.empty())
  {
    // Nothing reported, but the map may have moved under us: a writer marks a
    // replica draining or dead and the tick is what acts on it.
    {
      const std::lock_guard<std::mutex> lock(_routingMutex);
      if (_stateMap.version() != _syncedVersion) { syncLocked(); }
    }
    // syncLocked's loss path queues revokes too.
    flushRouting();
    return;
  }

  // Applied on the service tick, so a heartbeat signal never runs routing policy
  // on the reporting thread.
  {
    const std::lock_guard<std::mutex> lock(_routingMutex);
    if (_strategy == nullptr) { return; }
    for (const std::string &replicaName : losses)
    {
      _strategy->onReplicaLost(replicaName);
      _lossesApplied += 1;
    }
  }
  // A loss revokes once per affected session, each a round trip to every
  // replica. This is the moment a replica has just died -- exactly when routing
  // must not freeze -- so the sending happens with the lock released.
  flushRouting();
}

void Coordinator::finalize()
{
  // Same order as flushRouting: flush lock first. Holding it here is what makes
  // destroying the strategy safe against a send still running through it.
  const std::lock_guard<std::mutex> flushLock(_flushMutex);
  const std::lock_guard<std::mutex> lock(_routingMutex);
  _strategy.reset();
  _planner.reset();
  _sessions.reset();
  _instanceByName.clear();
}

std::optional<uint64_t> Coordinator::instanceFor(const std::string &replicaName) const
{
  const std::lock_guard<std::mutex> lock(_routingMutex);
  const auto                        it = _instanceByName.find(replicaName);
  if (it == _instanceByName.end()) { return std::nullopt; }
  return it->second;
}

router::RoutingStats Coordinator::stats() const
{
  const std::lock_guard<std::mutex> lock(_routingMutex);
  return _strategy == nullptr ? router::RoutingStats{} : _strategy->stats();
}

bool Coordinator::initialized() const
{
  const std::lock_guard<std::mutex> lock(_routingMutex);
  return _strategy != nullptr;
}

int Coordinator::lossesApplied() const
{
  const std::lock_guard<std::mutex> lock(_routingMutex);
  return _lossesApplied;
}

} // namespace serving::coordinator
