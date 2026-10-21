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
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
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
  ///
  /// The high half is the coordinator's epoch and the low half a counter, so a
  /// restarted coordinator issues generations above the ones replicas already
  /// hold. A bare counter would restart at 1 and every new rule would look
  /// stale to a replica that survived the restart.
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

  /// Whether a replica named in a path planned earlier can still be sent to.
  /// A path is planned once and reused for the life of a session, so something
  /// has to re-ask: a replica that has since started draining or gone pending
  /// is still in the path, and nothing else would notice.
  [[nodiscard]] bool routable(const ReplicaSpec &replica);

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

  /// A replica is gone. A transport that holds a per-replica channel stops
  /// writing to it: a dead consumer never drains, so pushing to it blocks.
  virtual void onReplicaUnreachable(const std::string & /*replicaName*/) {}
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

  /**
   * The path this session would take, worked out without committing to it.
   *
   * For inspection. `onRequest` is not a query: it bumps the generation,
   * publishes a rule to every hop, and evicts and revokes another session's
   * path to make room. An endpoint that exists to report what was decided must
   * not publish, revoke or evict, so it asks this instead -- which still
   * answers for a session that has no standing path yet, because reporting
   * only on sessions that already exist would make the endpoint useless.
   *
   * Nullopt means this strategy cannot answer without committing, and the
   * caller should fall back to `onRequest`.
   */
  [[nodiscard]] virtual std::optional<RoutingPath> peek(const std::string & /*sessionId*/) { return std::nullopt; }

  /**
   * Perform any transport work the last call deferred.
   *
   * Publishing a rule is a blocking HTTP round trip per hop, and a revoke is
   * one per replica. Done inline they ran with the coordinator's routing lock
   * held, so the router planned one session at a time and a replica that
   * accepted a connection and never answered froze all routing for its whole
   * timeout -- at precisely the moment a replica dies, which is when the loss
   * path issues the most revokes.
   *
   * Callers take the decision under the lock and then call this without it.
   * Safe to call when nothing is pending.
   */
  virtual void flush() {}

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

  /// `epoch` is the high half of every generation this instance issues. It must
  /// increase across restarts of the same coordinator; the default is the wall
  /// clock in seconds, which a deployment can override with something it
  /// persists.
  DistributedRuleStrategy(RoutingPlanner &planner, RuleTransport &transport, size_t maxPaths = DEFAULT_MAX_SESSIONS, uint64_t epoch = 0);

  /// Generations are (epoch << 32 | counter).
  [[nodiscard]] static uint64_t makeGeneration(uint64_t epoch, uint32_t counter) { return (epoch << 32) | counter; }

  /// The first request for a session plans and publishes; later ones reuse the
  /// standing path without consulting the planner.
  [[nodiscard]] RoutingPath onRequest(const std::string &sessionId) override;

  void onReplicaLost(const std::string &replicaName) override;

  [[nodiscard]] RoutingMode  mode() const override { return RoutingMode::DistributedRules; }
  [[nodiscard]] RoutingStats stats() const override { return _stats; }

  [[nodiscard]] std::optional<RoutingPath> peek(const std::string &sessionId) override;

  /// Sessions with a standing path.
  [[nodiscard]] size_t activePaths() const { return _paths.size(); }

  void flush() override;

  private:

  void publish(const RoutingPath &path);
  void forget(const std::string &sessionId);

  /// One deferred transport call. A publish carries a holder and a next hop; a
  /// revoke carries neither, which is what tells them apart.
  struct Deferred
  {
    bool        isRevoke = false;
    ReplicaSpec holder;
    std::string sessionId;
    ReplicaSpec nextHop;
    uint64_t    generation = 0;
  };

  /// Queue one deferred transport call.
  void defer(Deferred operation);

  /// Guards `_outbox` alone, and is never held across a transport call. The
  /// routing lock the caller holds does not cover flush(), which is the whole
  /// point of deferring.
  std::mutex            _outboxMutex;
  std::vector<Deferred> _outbox;

  struct Standing
  {
    RoutingPath                      path;
    std::list<std::string>::iterator position;
  };

  RoutingPlanner &_planner;
  RuleTransport  &_transport;

  /// A standing path is keyed by a client-supplied session id and reused for as
  /// long as it exists, so without a bound the map grows with the number of
  /// distinct ids a caller sends. Mirrors SessionDirectory's cap.
  size_t                          _maxPaths;
  std::map<std::string, Standing> _paths;
  /// Least recently used at the front.
  std::list<std::string> _order;
  uint64_t               _epoch;
  uint32_t               _counter = 0;
  RoutingStats           _stats;
};

/// Build the strategy a deployment was launched with. `transport` may be null
/// for Option A, which never publishes.
[[nodiscard]] std::unique_ptr<RoutingStrategy> makeRoutingStrategy(RoutingMode mode, RoutingPlanner &planner, RuleTransport *transport);

} // namespace serving::router
