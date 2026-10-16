#include <serving/infra/scaling.hpp>

#include <algorithm>
#include <stdexcept>

namespace serving::infra
{

namespace
{

/// Replicas of a partition that count towards its capacity: the ones serving
/// and the ones on their way up. A draining or dead one is leaving, so neither
/// holds the partition above its floor nor argues against adding more.
std::vector<const Replica *> contributing(const StateMap &map, PartitionId partition)
{
  std::vector<const Replica *> members;
  for (const Replica *replica : map.replicasIn(partition))
  {
    // A reserved spare counts towards the partition: it is paid for and one
    // message away, so ordering more capacity while one waits is waste.
    if (replica->phase == Phase::Active || replica->phase == Phase::Pending || replica->phase == Phase::Reserved) { members.push_back(replica); }
  }
  return members;
}

/// Only an active replica is carrying anything, so only an active one is worth
/// draining and only active ones belong in a load average.
std::vector<const Replica *> serving(const StateMap &map, PartitionId partition)
{
  std::vector<const Replica *> members;
  for (const Replica *replica : map.replicasIn(partition))
  {
    if (replica->phase == Phase::Active) { members.push_back(replica); }
  }
  return members;
}

std::string number(double value)
{
  std::string text = std::to_string(value);
  // Trailing zeroes in a reason string are noise.
  while (text.size() > 1 && text.back() == '0') { text.pop_back(); }
  if (!text.empty() && text.back() == '.') { text.pop_back(); }
  return text;
}

} // namespace

const ReplicaLoad *LoadSnapshot::find(const std::string &replicaName) const
{
  const auto it = std::find_if(replicas.begin(), replicas.end(), [&](const ReplicaLoad &load) { return load.replica == replicaName; });
  return it == replicas.end() ? nullptr : &*it;
}

double LoadSnapshot::meanOutstanding(const std::vector<const Replica *> &members) const
{
  if (members.empty()) { return 0.0; }
  int total = 0;
  for (const Replica *replica : members)
  {
    if (const ReplicaLoad *load = find(replica->name); load != nullptr) { total += load->outstanding; }
  }
  return static_cast<double>(total) / static_cast<double>(members.size());
}

int LoadSnapshot::totalPreemptions(const std::vector<const Replica *> &members) const
{
  int total = 0;
  for (const Replica *replica : members)
  {
    if (const ReplicaLoad *load = find(replica->name); load != nullptr) { total += load->preemptions; }
  }
  return total;
}

void ScalingConfig::validate() const
{
  if (minReplicas < 1) { throw std::invalid_argument("min_replicas must be at least 1: a partition with no replica makes the pipeline unservable"); }
  if (maxReplicas < minReplicas) { throw std::invalid_argument("max_replicas must not be below min_replicas"); }
  if (devicesPerReplica < 1) { throw std::invalid_argument("devices_per_replica must be at least 1"); }
  if (scaleUpAbove <= scaleDownBelow)
  {
    // Without a gap the partition sits between the two thresholds and scales
    // up and down forever.
    throw std::invalid_argument("scale_up_above must be greater than scale_down_below, or the partition flaps");
  }
  if (preemptionsAbove < 0) { throw std::invalid_argument("preemptions_above must not be negative"); }
}

const char *scalingActionName(ScalingAction action) { return action == ScalingAction::ScaleUp ? "scale-up" : "scale-down"; }

std::vector<ScalingDecision> decideScaling(const StateMap &map, const LoadSnapshot &load, const ScalingConfig &config)
{
  config.validate();

  std::vector<ScalingDecision> decisions;
  for (const PartitionId partition : map.partitions())
  {
    const auto members = contributing(map, partition);
    const auto active  = serving(map, partition);
    const int  count   = static_cast<int>(members.size());

    const double mean        = load.meanOutstanding(active);
    const int    preemptions = load.totalPreemptions(active);

    // Thrashing first: a preempted request has already thrown away work, which
    // a queue depth does not tell you.
    if (config.preemptionsAbove > 0 && preemptions > config.preemptionsAbove && count < config.maxReplicas)
    {
      decisions.push_back({partition, ScalingAction::ScaleUp, "", "preemptions " + std::to_string(preemptions) + " above " + std::to_string(config.preemptionsAbove)});
      continue;
    }

    if (mean > config.scaleUpAbove)
    {
      if (count >= config.maxReplicas) { continue; }
      decisions.push_back({partition, ScalingAction::ScaleUp, "", "mean outstanding " + number(mean) + " above " + number(config.scaleUpAbove)});
      continue;
    }

    if (mean < config.scaleDownBelow && count > config.minReplicas && !active.empty())
    {
      // Drain the least loaded: it has the least in flight to finish, so the
      // capacity comes back soonest.
      const Replica *victim = active.front();
      int            lowest = load.find(victim->name) == nullptr ? 0 : load.find(victim->name)->outstanding;
      for (const Replica *candidate : active)
      {
        const ReplicaLoad *sample  = load.find(candidate->name);
        const int          current = sample == nullptr ? 0 : sample->outstanding;
        if (current < lowest)
        {
          lowest = current;
          victim = candidate;
        }
      }
      decisions.push_back({partition, ScalingAction::ScaleDown, victim->name, "mean outstanding " + number(mean) + " below " + number(config.scaleDownBelow)});
    }
  }
  return decisions;
}

namespace
{

/// One past the highest port the deployment already uses.
///
/// `validate()` requires endpoints to be unique, so a new replica cannot reuse
/// one. Taking the maximum rather than the first gap keeps a drained replica's
/// port out of circulation while it is still finishing its work.
int nextFreePort(const StateMap &map)
{
  int highest = 0;
  for (const Replica &replica : map.replicas()) { highest = std::max(highest, replica.port); }
  // Nothing in the map has one: a deployment of replicas on ephemeral ports,
  // which a launcher cannot target anyway. Leave it at zero and let the
  // supervisor's probe report it rather than inventing a number.
  return highest == 0 ? 0 : highest + 1;
}

} // namespace

std::vector<ScalingOutcome> applyScaling(StateMap &map, CapacityProvider &provider, const std::vector<ScalingDecision> &decisions, const ScalingConfig &config)
{
  std::vector<ScalingOutcome> outcomes;
  outcomes.reserve(decisions.size());

  for (const ScalingDecision &decision : decisions)
  {
    ScalingOutcome outcome;
    outcome.decision = decision;

    if (decision.action == ScalingAction::ScaleDown)
    {
      // Draining, not stopping: it keeps serving what it already has, and
      // routing stops sending it anything new.
      outcome.applied = map.setDesiredPhase(decision.replica, Phase::Draining) && map.setPhase(decision.replica, Phase::Draining);
      outcome.replica = decision.replica;
      if (!outcome.applied) { outcome.declined = "no replica '" + decision.replica + "' to drain"; }
      outcomes.push_back(std::move(outcome));
      continue;
    }

    CapacityRequest request;
    request.partition = decision.partition;
    request.devices   = config.devicesPerReplica;
    // A provider hands out machines and cards; the port a replica listens on
    // is a property of the deployment, and the map is where the deployment is
    // written down. Without one the new replica lands on port 0, which cannot
    // be launched, probed or routed to -- it would sit pending for ever.
    request.port = nextFreePort(map);

    const auto grant = provider.acquire(request);
    if (!grant.has_value())
    {
      // The ordinary answer when there is nothing left, not an error: the
      // policy asked, the provider had nothing, the deployment stays as it is.
      outcome.declined = "no capacity available from the " + std::string(provider.kind()) + " provider";
      outcomes.push_back(std::move(outcome));
      continue;
    }

    // A cloud provider hands back a machine, not a card on one. Register it
    // before the replica, or the replica names a node the map does not have
    // and validate() rightly calls the deployment broken.
    if (grant->newNode.has_value()) { map.addNode(*grant->newNode); }

    Replica replica;
    replica.name      = grant->replica;
    replica.partition = decision.partition;
    replica.node      = grant->node;
    replica.devices   = grant->devices;
    replica.port      = grant->port;
    replica.instance  = grant->instance;
    // Pending, never active: it cannot serve for the minutes its model takes
    // to load, and routing already refuses to send to a pending replica.
    // Whatever starts it marks it active when it is up.
    replica.phase        = Phase::Pending;
    replica.desiredPhase = Phase::Active;
    map.addReplica(replica);

    outcome.applied = true;
    outcome.replica = grant->replica;
    outcomes.push_back(std::move(outcome));
  }
  return outcomes;
}

} // namespace serving::infra
