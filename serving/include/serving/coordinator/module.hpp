#pragma once

/**
 * The partition coordinator, a serving::modules::Module.
 *
 * The platform's engine owns its lifecycle and drives service() on a timer.
 * It owns the routing strategy, the map from routing name to HiCR instance,
 * and replica-loss handling: losses are queued and applied on the service
 * tick, never on the reporting thread. Routing rules are control-plane
 * messages; tensors between kernels do not pass through here.
 */

#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <serving/infra/state.hpp>
#include <serving/router/strategy.hpp>

namespace serving::coordinator
{

/// What one synchronisation changed, so a caller (and a test) can see the tick
/// did work without reading the registries.
struct SyncReport
{
  int added   = 0;
  int removed = 0;
  /// Replicas whose routable state changed.
  int rephased = 0;
  /// Replicas that went to `dead`, which is the loss path: paths through them
  /// are revoked and their sessions released.
  std::vector<std::string> lost;
  /// Replicas the map has in a partition routing does not know about.
  /// Repartitioning is M1's problem, not a routing update's.
  std::vector<std::string> unplaceable;

  [[nodiscard]] bool empty() const { return added == 0 && removed == 0 && rephased == 0 && lost.empty(); }
};

/// A replica as the coordinator sees it: a routing identity plus the instance
/// that serves it. The routing layer works in names; the platform works in
/// instance ids, and this is where the two meet.
struct ReplicaBinding
{
  router::ReplicaSpec spec;
  /// HiCR::Instance::instanceId_t, kept as a plain integer so this header does
  /// not drag HiCR into every translation unit that merely routes.
  uint64_t instanceId = 0;
};

struct CoordinatorConfig
{
  /// Which partition this coordinator owns. Its own replicas live here; the
  /// other partitions' entries are what let it plan a whole path.
  router::PartitionId partition = 0;

  /// Option A or Option B, chosen at launch.
  router::RoutingMode mode = router::RoutingMode::CoordinatorIngress;

  router::RoutingConfig routing;

  /// Every replica in the pipeline, across all partitions.
  std::vector<ReplicaBinding> replicas;
};

/**
 * Coordinates one partition. HiCR-free; platform_module.hpp binds it to Module.
 */
class Coordinator
{
  public:

  Coordinator(CoordinatorConfig config, router::RuleTransport *transport);

  /// Build the planner and the strategy. Mirrors Module::initialize.
  void initialize();

  /// Where this session's request should go, hop by hop.
  [[nodiscard]] router::RoutingPath routeFor(const std::string &sessionId);

  /**
   * Report a replica as gone.
   *
   * Queued rather than applied: this arrives from a health or heartbeat signal,
   * and running routing policy on that thread would put the control plane's
   * timing in the request path. `service()` drains it.
   */
  void reportReplicaLost(const std::string &replicaName);

  /// One tick of the platform's service timer: apply whatever has been reported,
  /// then take routing to whatever the state map now says.
  void service();

  /**
   * The infrastructure state map this coordinator routes from.
   *
   * Writers edit it; `service()` notices the version moved and brings routing
   * into line. Deliberately not applied on the writing thread: a health signal
   * marking a replica dead must not run routing policy where it lands, the same
   * reason `reportReplicaLost` queues.
   */
  [[nodiscard]] infra::StateMap       &stateMap() { return _stateMap; }
  [[nodiscard]] const infra::StateMap &stateMap() const { return _stateMap; }

  /**
   * The lock that guards the map, for the other threads that edit it.
   *
   * The coordinator owns the map, so it owns the lock. A deployer's ticker and
   * an operator's HTTP handler both reach `stateMap()` directly; without one
   * shared lock between them `addReplica` reallocates the replica vector under
   * a `const Replica *` another thread is still holding. The coordinator's own
   * entry points take this internally, so a caller that holds it must not call
   * back into them.
   */
  [[nodiscard]] std::mutex &stateMapMutex() const { return _routingMutex; }

  /// The path a session already has, without planning or publishing one. For
  /// inspection endpoints; see RoutingStrategy::peek.
  [[nodiscard]] std::optional<router::RoutingPath> peekRoute(const std::string &sessionId) const;

  /// Bring routing to the map, now, and say what changed. `service()` calls
  /// this; a test calls it directly.
  SyncReport syncFromMap();

  /// The map version routing currently reflects.
  [[nodiscard]] uint64_t syncedVersion() const;

  /// Release the strategy. Mirrors Module::finalize.
  void finalize();

  [[nodiscard]] router::PartitionId partition() const { return _config.partition; }
  [[nodiscard]] router::RoutingMode mode() const { return _config.mode; }
  [[nodiscard]] bool                initialized() const;

  /// The instance serving a routing name, or nullopt if it is not one of ours.
  [[nodiscard]] std::optional<uint64_t> instanceFor(const std::string &replicaName) const;

  [[nodiscard]] router::RoutingStats stats() const;

  /// Replica-loss reports drained so far, so a test can see the tick did work.
  [[nodiscard]] int lossesApplied() const;

  private:

  /// Caller holds `_routingMutex`.
  SyncReport syncLocked();

  CoordinatorConfig      _config;
  router::RuleTransport *_transport = nullptr;

  std::unique_ptr<router::SessionDirectory> _sessions;
  std::unique_ptr<router::RoutingPlanner>   _planner;
  std::unique_ptr<router::RoutingStrategy>  _strategy;

  std::map<std::string, uint64_t> _instanceByName;

  /// Guards the loss queue only, so a heartbeat never waits behind a route.
  mutable std::mutex      _mutex;
  std::deque<std::string> _pendingLosses;

  /// Guards the routing state. Requests arrive on their own threads while the
  /// platform's timer drives service(), and both reach the same strategy,
  /// planner and session directory.
  mutable std::mutex _routingMutex;
  int                _lossesApplied = 0;

  /// What runs where. Guarded by `_routingMutex` like the routing state it
  /// feeds, so a write and a sync cannot interleave.
  infra::StateMap _stateMap;
  /// The version `syncFromMap` last brought routing to. Starts at the version
  /// `initialize()` seeded, so an untouched map costs nothing per tick.
  uint64_t _syncedVersion = 0;
};

} // namespace serving::coordinator
