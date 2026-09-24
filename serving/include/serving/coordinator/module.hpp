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

#include <serving/router/strategy.hpp>

namespace serving::coordinator
{

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

  /// One tick of the platform's service timer: apply whatever has been reported.
  void service();

  /// Release the strategy. Mirrors Module::finalize.
  void finalize();

  [[nodiscard]] router::PartitionId partition() const { return _config.partition; }
  [[nodiscard]] router::RoutingMode mode() const { return _config.mode; }
  [[nodiscard]] bool                initialized() const { return _strategy != nullptr; }

  /// The instance serving a routing name, or nullopt if it is not one of ours.
  [[nodiscard]] std::optional<uint64_t> instanceFor(const std::string &replicaName) const;

  [[nodiscard]] router::RoutingStats stats() const;

  /// Replica-loss reports drained so far, so a test can see the tick did work.
  [[nodiscard]] int lossesApplied() const { return _lossesApplied; }

  private:

  CoordinatorConfig      _config;
  router::RuleTransport *_transport = nullptr;

  std::unique_ptr<router::SessionDirectory> _sessions;
  std::unique_ptr<router::RoutingPlanner>   _planner;
  std::unique_ptr<router::RoutingStrategy>  _strategy;

  std::map<std::string, uint64_t> _instanceByName;

  mutable std::mutex      _mutex;
  std::deque<std::string> _pendingLosses;
  int                     _lossesApplied = 0;
};

} // namespace serving::coordinator
