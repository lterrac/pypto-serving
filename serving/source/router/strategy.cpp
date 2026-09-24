#include <serving/router/strategy.hpp>

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
  }
  return "unknown";
}

RoutingMode parseRoutingMode(const std::string &text)
{
  if (text == "ingress") { return RoutingMode::CoordinatorIngress; }
  throw std::invalid_argument("unknown routing mode '" + text + "' (expected 'ingress')");
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

std::unique_ptr<RoutingStrategy> makeRoutingStrategy(RoutingMode mode, RoutingPlanner &planner)
{
  switch (mode)
  {
  case RoutingMode::CoordinatorIngress: return std::make_unique<CoordinatorIngressStrategy>(planner);
  }
  throw std::invalid_argument("unknown routing mode");
}

} // namespace serving::router
