#pragma once

/**
 * When to scale (`scaling-pypto-serving.pptx`, M2).
 *
 * The deck puts this decision in serving, not the control plane: the control
 * plane executes actions and decides nothing, so what it means to be
 * overloaded is serving's to say. `decideScaling` reads and changes nothing;
 * `applyScaling` is what acts, and it is a separate call so a deployment can
 * watch the policy's opinion before letting it run.
 *
 * Per partition, because a partition is the unit that scales: replicas are
 * added or drained inside one without touching the others.
 *
 * Load is passed in rather than read off the map. The map is a durable
 * document with a version readers use as a change signal; folding a figure
 * that moves every step into it would churn that version and make it
 * worthless. What runs where is durable, how busy it is is not.
 */

#include <optional>
#include <string>
#include <vector>

#include <serving/infra/capacity.hpp>
#include <serving/infra/state.hpp>

namespace serving::infra
{

/// What one replica is carrying, sampled now.
struct ReplicaLoad
{
  std::string replica;
  /// Requests admitted and not yet finished.
  int outstanding = 0;
  /// Tokens admitted and not yet generated: the engine's own queue depth.
  int pendingTokens = 0;
  /// Preemptions since start. Rising means the KV pool cannot hold what has
  /// been admitted, and the replica is already failing to serve it.
  int preemptions = 0;
};

struct LoadSnapshot
{
  std::vector<ReplicaLoad> replicas;

  [[nodiscard]] const ReplicaLoad *find(const std::string &replica) const;
  /// Mean outstanding across the named replicas. Zero when none are named,
  /// which reads as idle and is what an empty partition should look like.
  [[nodiscard]] double meanOutstanding(const std::vector<const Replica *> &members) const;
  [[nodiscard]] int    totalPreemptions(const std::vector<const Replica *> &members) const;
};

/**
 * The serving replica of a partition carrying the least, or nullptr when the
 * partition has none serving.
 *
 * Draining the least loaded is what makes a scale-down cheap: it has the least
 * in flight to finish, so the capacity comes back soonest. Shared so the
 * automatic policy and an operator's explicit scale-down cannot disagree about
 * which replica that is.
 */
[[nodiscard]] const Replica *leastLoadedServing(const StateMap &map, PartitionId partition, const LoadSnapshot &load);

struct ScalingConfig
{
  /// Add capacity when the mean outstanding per routable replica is above this.
  double scaleUpAbove = 8.0;
  /// Drain one when it is below this. Deliberately far below `scaleUpAbove`:
  /// the gap is what stops a partition flapping between the two.
  double scaleDownBelow = 2.0;

  /// A partition never drops below this many routable replicas, whatever the
  /// load says: scaling to zero makes the pipeline unservable.
  int minReplicas = 1;
  int maxReplicas = 8;

  /// Preemptions across a partition that force a scale-up regardless of queue
  /// depth. Thrashing is a harder signal than a queue: a preempted request has
  /// already thrown away work. Zero disables it.
  int preemptionsAbove = 0;

  /// Devices a new replica of this deployment needs.
  int devicesPerReplica = 1;

  /// Throws std::invalid_argument on a configuration that would flap or
  /// scale to nothing.
  void validate() const;
};

enum class ScalingAction
{
  ScaleUp,
  ScaleDown,
};

[[nodiscard]] const char *scalingActionName(ScalingAction action);

struct ScalingDecision
{
  PartitionId   partition = 0;
  ScalingAction action    = ScalingAction::ScaleUp;
  /// Which replica to drain. Empty for a scale-up, where none exists yet.
  std::string replica;
  /// Why, in a form fit for a log: a decision nobody can explain is one nobody
  /// will trust enough to leave switched on.
  std::string reason;

  bool operator==(const ScalingDecision &) const = default;
};

/// What the policy thinks, per partition, in partition order. Reads only.
[[nodiscard]] std::vector<ScalingDecision> decideScaling(const StateMap &map, const LoadSnapshot &load, const ScalingConfig &config);

/// What acting on a decision did.
struct ScalingOutcome
{
  ScalingDecision decision;
  /// True when the map was edited.
  bool applied = false;
  /// The replica added or drained, once it has a name.
  std::string replica;
  /// Why not, when `applied` is false -- the provider being empty is the
  /// ordinary case and not an error.
  std::string declined;
};

/**
 * Act on decisions: acquire capacity and put the new replica in the map as
 * `pending`, or set an existing one `draining`.
 *
 * A new replica is `pending`, never `active`: it provably cannot serve for the
 * minutes its model takes to load, and routing already refuses to send to a
 * pending replica. Whatever starts it marks it active when it is up.
 */
[[nodiscard]] std::vector<ScalingOutcome> applyScaling(StateMap &map, CapacityProvider &provider, const std::vector<ScalingDecision> &decisions, const ScalingConfig &config);

} // namespace serving::infra
