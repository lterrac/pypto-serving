#pragma once

/**
 * Routing for a partitioned deployment: a request follows a path of one
 * replica per partition, in order.
 *
 * Two strategies, chosen by the configuration file's routing_mode:
 *
 *  ingress  the coordinator routes every request and forwards to the next
 *           partition.
 *  rules    the coordinator plans once per session and publishes a forwarding
 *           rule to each replica on the path; replicas forward directly. A
 *           conversation follows the same path every turn, so once per session
 *           is enough.
 *
 * Both use RoutingPlanner over a per-partition ReplicaRegistry.
 */

#include <cstdint>
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
  /// Bumped whenever the path is replanned, so a replica holding an older rule
  /// can tell that it is stale. Failure recovery depends on this: a lost replica
  /// invalidates the paths through it, and in-flight work must not follow them.
  uint64_t generation = 0;

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

/// Which option a deployment runs. Chosen in the configuration file.
enum class RoutingMode
{
  CoordinatorIngress, ///< Option A
  DistributedRules,   ///< Option B
};

[[nodiscard]] const char *routingModeName(RoutingMode mode);
/// Parse "ingress" / "rules"; throws std::invalid_argument on anything else.
[[nodiscard]] RoutingMode parseRoutingMode(const std::string &text);

/**
 * How a coordinator tells a replica where to forward. `rules` publishes
 * through it; `ingress` never calls it. channel_transport.hpp is the
 * platform-channel implementation.
 */
class RuleTransport
{
  public:

  virtual ~RuleTransport() = default;

  /// Tell `holder` that `sessionId` continues at `nextHop`. A path's last hop
  /// gets no rule: there is nothing after it.
  virtual void publish(const ReplicaSpec &holder, const std::string &sessionId, const ReplicaSpec &nextHop, uint64_t generation) = 0;

  /// Withdraw every rule for a session, because its path is being replanned.
  virtual void revoke(const std::string &sessionId, uint64_t generation) = 0;
};

/// What a strategy did, so a caller (and a test) can tell the options apart.
struct RoutingStats
{
  /// Paths computed. Option A: one per request. Option B: one per session.
  int plansComputed = 0;
  /// Rules pushed to replicas. Option A never publishes.
  int rulesPublished = 0;
  /// Paths thrown away because a replica was lost.
  int pathsInvalidated = 0;
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

/// Option B: the coordinator decides once per session and publishes the rule.
class DistributedRuleStrategy : public RoutingStrategy
{
  public:

  DistributedRuleStrategy(RoutingPlanner &planner, RuleTransport &transport);

  /// The first request for a session plans and publishes; later ones reuse the
  /// standing path without consulting the planner.
  [[nodiscard]] RoutingPath onRequest(const std::string &sessionId) override;

  void onReplicaLost(const std::string &replicaName) override;

  [[nodiscard]] RoutingMode  mode() const override { return RoutingMode::DistributedRules; }
  [[nodiscard]] RoutingStats stats() const override { return _stats; }

  /// Sessions with a standing path.
  [[nodiscard]] size_t activePaths() const { return _paths.size(); }

  private:

  void publish(const RoutingPath &path);

  RoutingPlanner                    &_planner;
  RuleTransport                     &_transport;
  std::map<std::string, RoutingPath> _paths;
  uint64_t                           _generation = 0;
  RoutingStats                       _stats;
};

/// Build the strategy a deployment was launched with. `transport` may be null
/// for Option A, which never publishes.
[[nodiscard]] std::unique_ptr<RoutingStrategy> makeRoutingStrategy(RoutingMode mode, RoutingPlanner &planner, RuleTransport *transport);

} // namespace serving::router
