#pragma once

/**
 * Routing for a partitioned deployment: a request follows a path of one
 * replica per partition, in order.
 *
 * RoutingPlanner plans the path over a per-partition ReplicaRegistry;
 * RoutingStrategy is how a coordinator applies it. CoordinatorIngressStrategy
 * (routing_mode `ingress`) routes every request and forwards to the next
 * partition.
 */

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <serving/router/routing.hpp>

namespace serving::router
{

/// One replica per partition, in pipeline order: the path a session follows.
struct RoutingPath
{
  std::string              sessionId;
  std::vector<ReplicaSpec> hops;

  [[nodiscard]] bool empty() const { return hops.empty(); }
};

/**
 * Picks a path for a session, honouring affinity within each partition.
 *
 * One `ReplicaRegistry` per partition, because a partition is the unit that
 * scales: replicas are added or drained inside one without touching the others.
 */
class RoutingPlanner
{
  public:

  RoutingPlanner(RoutingConfig config, SessionDirectory &sessions, const std::vector<ReplicaSpec> &replicas);

  /// Choose one replica per partition. Throws NoReplicaAvailable if any
  /// partition has nothing routable -- a pipeline with a broken stage cannot
  /// serve at all.
  [[nodiscard]] RoutingPath plan(const std::string &sessionId);

  [[nodiscard]] std::vector<PartitionId> partitions() const;
  [[nodiscard]] ReplicaRegistry         *registry(PartitionId partition);

  /// Mark a replica unroutable across whichever partition owns it.
  void setReady(const std::string &replicaName, bool ready);

  void acquire(const ReplicaSpec &replica);
  void release(const ReplicaSpec &replica);

  private:

  SessionDirectory                                       &_sessions;
  std::map<PartitionId, std::unique_ptr<ReplicaRegistry>> _byPartition;
};

/// Which option a deployment runs. Chosen at launch.
enum class RoutingMode
{
  CoordinatorIngress, ///< Option A
};

[[nodiscard]] const char *routingModeName(RoutingMode mode);
/// Parse "ingress"; throws std::invalid_argument on anything else.
[[nodiscard]] RoutingMode parseRoutingMode(const std::string &text);

/// What a strategy did, so a caller (and a test) can see where the decision fell.
struct RoutingStats
{
  /// Paths computed. Option A: one per request.
  int plansComputed = 0;
};

class RoutingStrategy
{
  public:

  virtual ~RoutingStrategy() = default;

  /// The path this request should follow.
  [[nodiscard]] virtual RoutingPath onRequest(const std::string &sessionId) = 0;

  /// A replica died: take it out and drop whatever depended on it.
  virtual void onReplicaLost(const std::string &replicaName) = 0;

  [[nodiscard]] virtual RoutingMode  mode() const  = 0;
  [[nodiscard]] virtual RoutingStats stats() const = 0;
};

/// Option A: the coordinator is on the path, so it decides per request.
class CoordinatorIngressStrategy : public RoutingStrategy
{
  public:

  explicit CoordinatorIngressStrategy(RoutingPlanner &planner);

  [[nodiscard]] RoutingPath onRequest(const std::string &sessionId) override;
  void                      onReplicaLost(const std::string &replicaName) override;

  [[nodiscard]] RoutingMode  mode() const override { return RoutingMode::CoordinatorIngress; }
  [[nodiscard]] RoutingStats stats() const override { return _stats; }

  private:

  RoutingPlanner &_planner;
  RoutingStats    _stats;
};

/// Build the strategy a deployment was launched with.
[[nodiscard]] std::unique_ptr<RoutingStrategy> makeRoutingStrategy(RoutingMode mode, RoutingPlanner &planner);

} // namespace serving::router
