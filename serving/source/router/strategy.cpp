#include <serving/router/strategy.hpp>

#include <algorithm>
#include <stdexcept>

namespace serving::router
{

// ---------------------------------------------------------------------------
// RoutingPlanner
// ---------------------------------------------------------------------------

RoutingPlanner::RoutingPlanner(RoutingConfig config, SessionDirectory &sessions, const std::vector<ReplicaSpec> &replicas)
  : _sessions(sessions)
{
  std::map<PartitionId, std::vector<ReplicaSpec>> grouped;
  for (const ReplicaSpec &replica : replicas) { grouped[replica.partition].push_back(replica); }
  for (auto &[partition, members] : grouped) { _byPartition.emplace(partition, std::make_unique<ReplicaRegistry>(config, sessions, members)); }
}

std::vector<PartitionId> RoutingPlanner::partitions() const
{
  std::vector<PartitionId> ids;
  ids.reserve(_byPartition.size());
  for (const auto &[partition, registry] : _byPartition) { ids.push_back(partition); }
  return ids;
}

ReplicaRegistry *RoutingPlanner::registry(PartitionId partition)
{
  const auto it = _byPartition.find(partition);
  return it == _byPartition.end() ? nullptr : it->second.get();
}

RoutingPath RoutingPlanner::plan(const std::string &sessionId)
{
  RoutingPath path;
  path.sessionId = sessionId;

  // _byPartition is ordered, so the hops come out in pipeline order.
  for (auto &[partition, registry] : _byPartition)
  {
    // The session directory is shared, so a plan across several partitions would
    // have each hop overwrite the previous one's pin. Key the pin per partition
    // instead: affinity is per partition, since that is where the KV lives.
    const std::string scoped = sessionId + "#p" + std::to_string(partition);
    path.hops.push_back(registry->select(scoped).replica);
  }
  return path;
}

void RoutingPlanner::setReady(const std::string &replicaName, bool ready)
{
  for (auto &[partition, registry] : _byPartition)
  {
    if (registry->state(replicaName) != nullptr) { registry->setReady(replicaName, ready); }
  }
}

void RoutingPlanner::acquire(const ReplicaSpec &replica)
{
  ReplicaRegistry *target = registry(replica.partition);
  if (target != nullptr) { target->acquire(replica.name); }
}

void RoutingPlanner::release(const ReplicaSpec &replica)
{
  ReplicaRegistry *target = registry(replica.partition);
  if (target != nullptr) { target->release(replica.name); }
}

// ---------------------------------------------------------------------------
// Mode
// ---------------------------------------------------------------------------

const char *routingModeName(RoutingMode mode)
{
  switch (mode)
  {
  case RoutingMode::CoordinatorIngress: return "ingress";
  case RoutingMode::DistributedRules: return "rules";
  }
  return "unknown";
}

RoutingMode parseRoutingMode(const std::string &text)
{
  if (text == "ingress") { return RoutingMode::CoordinatorIngress; }
  if (text == "rules") { return RoutingMode::DistributedRules; }
  throw std::invalid_argument("unknown routing mode '" + text + "' (expected 'ingress' or 'rules')");
}

// ---------------------------------------------------------------------------
// Option A
// ---------------------------------------------------------------------------

CoordinatorIngressStrategy::CoordinatorIngressStrategy(RoutingPlanner &planner)
  : _planner(planner)
{}

RoutingPath CoordinatorIngressStrategy::onRequest(const std::string &sessionId)
{
  // The coordinator is on the path anyway, so it may as well decide afresh:
  // this is the "fully informed load balancing" the option buys.
  _stats.plansComputed += 1;
  return _planner.plan(sessionId);
}

void CoordinatorIngressStrategy::onReplicaLost(const std::string &replicaName)
{
  _planner.setReady(replicaName, false);
  // Nothing to invalidate: the next request re-plans by construction.
}

// ---------------------------------------------------------------------------
// Option B
// ---------------------------------------------------------------------------

DistributedRuleStrategy::DistributedRuleStrategy(RoutingPlanner &planner, RuleTransport &transport, size_t maxPaths)
  : _planner(planner),
    _transport(transport),
    _maxPaths(maxPaths == 0 ? 1 : maxPaths)
{}

RoutingPath DistributedRuleStrategy::onRequest(const std::string &sessionId)
{
  const auto existing = _paths.find(sessionId);
  if (existing != _paths.end())
  {
    // The standing path is reused without consulting the planner. Routing
    // happens once per session -- that is what keeps the decision off the
    // critical path.
    _order.splice(_order.end(), _order, existing->second.position);
    return existing->second.path;
  }

  _stats.plansComputed += 1;
  RoutingPath path = _planner.plan(sessionId);
  path.generation  = ++_generation;
  publish(path);

  while (_paths.size() >= _maxPaths && !_order.empty())
  {
    // Evicting without revoking would leave replicas forwarding along a path
    // this coordinator no longer knows about.
    const std::string oldest = _order.front();
    const auto        it     = _paths.find(oldest);
    if (it != _paths.end()) { _transport.revoke(oldest, it->second.path.generation); }
    forget(oldest);
  }

  const auto position = _order.insert(_order.end(), sessionId);
  _paths.emplace(sessionId, Standing{path, position});
  return path;
}

void DistributedRuleStrategy::forget(const std::string &sessionId)
{
  const auto it = _paths.find(sessionId);
  if (it == _paths.end()) { return; }
  _order.erase(it->second.position);
  _paths.erase(it);
}

void DistributedRuleStrategy::publish(const RoutingPath &path)
{
  // Each hop is told where to forward next; the last hop has nowhere to go.
  for (size_t i = 0; i + 1 < path.hops.size(); ++i)
  {
    _transport.publish(path.hops[i], path.sessionId, path.hops[i + 1], path.generation);
    _stats.rulesPublished += 1;
  }
}

void DistributedRuleStrategy::onReplicaLost(const std::string &replicaName)
{
  _planner.setReady(replicaName, false);
  // Before any revoke: the transport must stop writing to this replica's own
  // channel, which nothing is draining any more.
  _transport.onReplicaUnreachable(replicaName);

  // Every standing path through the lost replica is now wrong. Revoke it rather
  // than leaving replicas forwarding into a dead node, and let the next request
  // replan -- the generation on the new rule is what lets a replica holding the
  // old one recognise it as stale.
  std::vector<std::string> affected;
  for (const auto &[sessionId, standing] : _paths)
  {
    const bool touches = std::any_of(standing.path.hops.begin(), standing.path.hops.end(), [&](const ReplicaSpec &hop) { return hop.name == replicaName; });
    if (touches) { affected.push_back(sessionId); }
  }

  for (const std::string &sessionId : affected)
  {
    _transport.revoke(sessionId, _paths.at(sessionId).path.generation);
    forget(sessionId);
    _stats.pathsInvalidated += 1;
  }
}

// ---------------------------------------------------------------------------

std::unique_ptr<RoutingStrategy> makeRoutingStrategy(RoutingMode mode, RoutingPlanner &planner, RuleTransport *transport)
{
  if (mode == RoutingMode::CoordinatorIngress) { return std::make_unique<CoordinatorIngressStrategy>(planner); }
  if (transport == nullptr) { throw std::invalid_argument("the 'rules' routing mode needs a RuleTransport to publish through"); }
  return std::make_unique<DistributedRuleStrategy>(planner, *transport);
}

} // namespace serving::router
