#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <serving/infra/scaling.hpp>

using serving::infra::applyScaling;
using serving::infra::decideScaling;
using serving::infra::LoadSnapshot;
using serving::infra::LocalCapacity;
using serving::infra::Node;
using serving::infra::Phase;
using serving::infra::Replica;
using serving::infra::ReplicaLoad;
using serving::infra::ScalingAction;
using serving::infra::ScalingConfig;
using serving::infra::StateMap;
using serving::infra::validate;

namespace
{

StateMap atlas()
{
  StateMap map;
  map.addNode(Node{"hg-atlas-01", "192.168.150.11", {0, 1, 2, 3, 4, 5, 6, 7}});
  return map;
}

Replica replica(const std::string &name, int partition, std::vector<int> devices, Phase phase = Phase::Active)
{
  Replica r;
  r.name      = name;
  r.partition = partition;
  r.node      = "hg-atlas-01";
  r.devices   = std::move(devices);
  r.phase     = phase;
  return r;
}

LoadSnapshot busy(const std::vector<std::pair<std::string, int>> &outstanding)
{
  LoadSnapshot snapshot;
  for (const auto &[name, value] : outstanding) { snapshot.replicas.push_back(ReplicaLoad{name, value, 0, 0}); }
  return snapshot;
}

ScalingConfig config()
{
  ScalingConfig c;
  c.scaleUpAbove   = 8.0;
  c.scaleDownBelow = 2.0;
  c.minReplicas    = 1;
  c.maxReplicas    = 4;
  return c;
}

} // namespace

TEST(ScalingConfigTest, RefusesSettingsThatWouldFlapOrScaleToNothing)
{
  ScalingConfig c = config();
  c.minReplicas   = 0;
  EXPECT_THROW(c.validate(), std::invalid_argument);

  c                = config();
  c.scaleDownBelow = c.scaleUpAbove;
  // Without a gap the partition sits between the thresholds and oscillates.
  EXPECT_THROW(c.validate(), std::invalid_argument);

  c             = config();
  c.maxReplicas = 0;
  EXPECT_THROW(c.validate(), std::invalid_argument);
}

TEST(ScalingPolicyTest, SaysNothingWhileLoadSitsBetweenTheThresholds)
{
  StateMap map = atlas();
  map.addReplica(replica("p0a", 0, {0}));
  map.addReplica(replica("p0b", 0, {1}));

  EXPECT_TRUE(decideScaling(map, busy({{"p0a", 5}, {"p0b", 5}}), config()).empty());
}

TEST(ScalingPolicyTest, AddsCapacityWhenTheQueueIsDeep)
{
  StateMap map = atlas();
  map.addReplica(replica("p0a", 0, {0}));

  const auto decisions = decideScaling(map, busy({{"p0a", 20}}), config());
  ASSERT_EQ(decisions.size(), 1U);
  EXPECT_EQ(decisions[0].action, ScalingAction::ScaleUp);
  EXPECT_EQ(decisions[0].partition, 0);
  // A decision nobody can explain is one nobody leaves switched on.
  EXPECT_NE(decisions[0].reason.find("above"), std::string::npos) << decisions[0].reason;
}

TEST(ScalingPolicyTest, ThrashingForcesAScaleUpEvenOnAShallowQueue)
{
  StateMap map = atlas();
  map.addReplica(replica("p0a", 0, {0}));
  ScalingConfig c    = config();
  c.preemptionsAbove = 10;

  LoadSnapshot load;
  load.replicas.push_back(ReplicaLoad{"p0a", 1, 0, 50});

  // The queue is shallow enough to argue for scaling *down*, but a preempted
  // request has already thrown work away, which a queue depth cannot say.
  const auto decisions = decideScaling(map, load, c);
  ASSERT_EQ(decisions.size(), 1U);
  EXPECT_EQ(decisions[0].action, ScalingAction::ScaleUp);
  EXPECT_NE(decisions[0].reason.find("preemptions"), std::string::npos);
}

TEST(ScalingPolicyTest, DrainsTheLeastLoadedWhenQuiet)
{
  StateMap map = atlas();
  map.addReplica(replica("p0a", 0, {0}));
  map.addReplica(replica("p0b", 0, {1}));
  map.addReplica(replica("p0c", 0, {2}));

  const auto decisions = decideScaling(map, busy({{"p0a", 1}, {"p0b", 0}, {"p0c", 1}}), config());
  ASSERT_EQ(decisions.size(), 1U);
  EXPECT_EQ(decisions[0].action, ScalingAction::ScaleDown);
  // Least loaded: least in flight to finish, so the capacity returns soonest.
  EXPECT_EQ(decisions[0].replica, "p0b");
}

TEST(ScalingPolicyTest, NeverDrainsThePartitionBelowItsFloor)
{
  StateMap map = atlas();
  map.addReplica(replica("p0a", 0, {0}));

  // Idle, but scaling to zero makes the pipeline unservable.
  EXPECT_TRUE(decideScaling(map, busy({{"p0a", 0}}), config()).empty());
}

TEST(ScalingPolicyTest, StopsAtTheCeiling)
{
  StateMap      map = atlas();
  ScalingConfig c   = config();
  for (int i = 0; i < c.maxReplicas; ++i) { map.addReplica(replica("p0" + std::to_string(i), 0, {i})); }

  LoadSnapshot load;
  for (int i = 0; i < c.maxReplicas; ++i) { load.replicas.push_back(ReplicaLoad{"p0" + std::to_string(i), 99, 0, 0}); }

  EXPECT_TRUE(decideScaling(map, load, c).empty());
}

TEST(ScalingPolicyTest, APendingReplicaAlreadyCountsSoCapacityIsNotOrderedTwice)
{
  StateMap map = atlas();
  map.addReplica(replica("p0a", 0, {0}));
  map.addReplica(replica("p0b", 0, {1}, Phase::Pending));

  ScalingConfig c = config();
  c.maxReplicas   = 2;

  // p0b is loading, not serving, so it carries nothing -- but it is on its way
  // up and must not be ordered a second time.
  EXPECT_TRUE(decideScaling(map, busy({{"p0a", 99}}), c).empty());
}

TEST(ScalingPolicyTest, ADrainingReplicaNoLongerHoldsThePartitionUp)
{
  StateMap map = atlas();
  map.addReplica(replica("p0a", 0, {0}));
  map.addReplica(replica("p0b", 0, {1}, Phase::Draining));

  ScalingConfig c = config();
  c.minReplicas   = 2;

  // Only p0a counts, so the partition is below its floor and the drained one
  // is not an argument against replacing it.
  const auto decisions = decideScaling(map, busy({{"p0a", 99}}), c);
  ASSERT_EQ(decisions.size(), 1U);
  EXPECT_EQ(decisions[0].action, ScalingAction::ScaleUp);
}

TEST(ScalingPolicyTest, ScalesEachPartitionOnItsOwn)
{
  StateMap map = atlas();
  map.addReplica(replica("p0a", 0, {0}));
  map.addReplica(replica("p1a", 1, {1}));
  map.addReplica(replica("p1b", 1, {2}));

  const auto decisions = decideScaling(map, busy({{"p0a", 99}, {"p1a", 0}, {"p1b", 0}}), config());
  ASSERT_EQ(decisions.size(), 2U);
  EXPECT_EQ(decisions[0].partition, 0);
  EXPECT_EQ(decisions[0].action, ScalingAction::ScaleUp);
  EXPECT_EQ(decisions[1].partition, 1);
  EXPECT_EQ(decisions[1].action, ScalingAction::ScaleDown);
}

// ---------------------------------------------------------------------------
// applying
// ---------------------------------------------------------------------------

TEST(ScalingApplyTest, AddsTheNewReplicaPendingNotActive)
{
  StateMap map = atlas();
  map.addReplica(replica("p0a", 0, {0}));
  LocalCapacity provider(map);

  const auto decisions = decideScaling(map, busy({{"p0a", 99}}), config());
  const auto outcomes  = applyScaling(map, provider, decisions, config());

  ASSERT_EQ(outcomes.size(), 1U);
  EXPECT_TRUE(outcomes[0].applied) << outcomes[0].declined;

  const Replica *added = map.replica(outcomes[0].replica);
  ASSERT_NE(added, nullptr);
  // It cannot serve for the minutes its model takes to load, and routing
  // already refuses to send to a pending replica.
  EXPECT_EQ(added->phase, Phase::Pending);
  EXPECT_EQ(added->desiredPhase, Phase::Active);
  EXPECT_EQ(added->partition, 0);
  EXPECT_FALSE(added->devices.empty());
}

TEST(ScalingApplyTest, TheNewReplicaDoesNotCollideWithTheRunningOne)
{
  StateMap map = atlas();
  map.addReplica(replica("p0a", 0, {4}));
  LocalCapacity provider(map);

  const auto outcomes = applyScaling(map, provider, decideScaling(map, busy({{"p0a", 99}}), config()), config());
  ASSERT_TRUE(outcomes[0].applied);

  EXPECT_NE(map.replica(outcomes[0].replica)->devices.front(), 4);
  EXPECT_TRUE(validate(map).empty()) << validate(map).front();
}

TEST(ScalingApplyTest, DrainingLeavesItServingWhatItHas)
{
  StateMap map = atlas();
  map.addReplica(replica("p0a", 0, {0}));
  map.addReplica(replica("p0b", 0, {1}));
  LocalCapacity provider(map);

  const auto outcomes = applyScaling(map, provider, decideScaling(map, busy({{"p0a", 0}, {"p0b", 0}}), config()), config());
  ASSERT_EQ(outcomes.size(), 1U);
  ASSERT_TRUE(outcomes[0].applied);

  const Replica *drained = map.replica(outcomes[0].replica);
  ASSERT_NE(drained, nullptr);
  EXPECT_EQ(drained->phase, Phase::Draining);
  // Draining still holds its card: it is finishing what it already took.
  EXPECT_EQ(map.devicesInUse("hg-atlas-01").size(), 2U);
}

TEST(ScalingApplyTest, AnExhaustedProviderIsDeclinedNotFailed)
{
  StateMap map = atlas();
  // Every card taken.
  for (int i = 0; i < 8; ++i) { map.addReplica(replica("p0" + std::to_string(i), 0, {i})); }
  LocalCapacity provider(map);

  ScalingConfig c = config();
  c.maxReplicas   = 16;

  LoadSnapshot load;
  for (int i = 0; i < 8; ++i) { load.replicas.push_back(ReplicaLoad{"p0" + std::to_string(i), 99, 0, 0}); }

  const auto outcomes = applyScaling(map, provider, decideScaling(map, load, c), c);
  ASSERT_EQ(outcomes.size(), 1U);
  // The policy asked, the provider had nothing, the deployment stays as it is.
  EXPECT_FALSE(outcomes[0].applied);
  EXPECT_NE(outcomes[0].declined.find("no capacity"), std::string::npos) << outcomes[0].declined;
  EXPECT_EQ(map.replicas().size(), 8U);
}

TEST(ScalingApplyTest, DecidingTwiceInARowDoesNotOrderTwice)
{
  StateMap map = atlas();
  map.addReplica(replica("p0a", 0, {0}));
  LocalCapacity provider(map);
  const auto    c = config();

  (void)applyScaling(map, provider, decideScaling(map, busy({{"p0a", 99}}), c), c);
  // The replica it just added is pending, so it already counts towards the
  // partition: a second pass must not order capacity for the same shortfall.
  const auto second = decideScaling(map, busy({{"p0a", 99}}), c);
  ASSERT_EQ(second.size(), 1U);

  // ...until the ceiling, which is where that argument runs out.
  ScalingConfig capped = c;
  capped.maxReplicas   = 2;
  EXPECT_TRUE(decideScaling(map, busy({{"p0a", 99}}), capped).empty());
}

namespace
{

/// Stands in for a cloud: every grant brings a machine that was not there.
class NodeGrantingProvider final : public serving::infra::CapacityProvider
{
  public:

  std::optional<serving::infra::Grant> acquire(const serving::infra::CapacityRequest &request) override
  {
    if (_granted >= budget) { return std::nullopt; }
    const std::string name = "cloud-" + std::to_string(_granted++);

    serving::infra::Grant grant;
    grant.replica  = request.name.empty() ? name + "-r0" : request.name;
    grant.node     = name;
    grant.devices  = {0};
    grant.instance = 1000 + _granted;
    grant.newNode  = serving::infra::Node{name, "10.9.0." + std::to_string(_granted), {0}};
    return grant;
  }

  void        release(const std::string &) override { released += 1; }
  int         available(const serving::infra::CapacityRequest &) const override { return budget - _granted; }
  const char *kind() const override { return "fake-cloud"; }

  int budget   = 2;
  int released = 0;

  private:

  int _granted = 0;
};

} // namespace

TEST(ScalingApplyTest, ACloudGrantBringsItsMachineIntoTheMap)
{
  StateMap map;
  map.addNode(Node{"hg-atlas-01", "192.168.150.11", {0}});
  map.addReplica(replica("p0a", 0, {0}));
  NodeGrantingProvider provider;

  const auto outcomes = applyScaling(map, provider, decideScaling(map, busy({{"p0a", 99}}), config()), config());
  ASSERT_EQ(outcomes.size(), 1U);
  ASSERT_TRUE(outcomes[0].applied) << outcomes[0].declined;

  // The local node was full; the new replica is on a machine that did not
  // exist a moment ago, which is what "wire new nodes in" has to mean.
  const Replica *added = map.replica(outcomes[0].replica);
  ASSERT_NE(added, nullptr);
  EXPECT_EQ(added->node, "cloud-0");
  EXPECT_EQ(added->instance, 1001U);

  // Registered before the replica, or validate() would call this broken.
  ASSERT_NE(map.node("cloud-0"), nullptr);
  EXPECT_EQ(map.node("cloud-0")->host, "10.9.0.1");
  EXPECT_TRUE(validate(map).empty()) << validate(map).front();
}

TEST(ScalingApplyTest, TheLocalNodeStaysSeparateFromTheGrantedOnes)
{
  StateMap map;
  map.addNode(Node{"hg-atlas-01", "192.168.150.11", {0}});
  map.addReplica(replica("p0a", 0, {0}));
  NodeGrantingProvider provider;

  ScalingConfig c = config();
  c.maxReplicas   = 3;
  for (int i = 0; i < 2; ++i)
  {
    const auto decisions = decideScaling(map, busy({{"p0a", 99}}), c);
    if (decisions.empty()) { break; }
    applyScaling(map, provider, decisions, c);
  }

  // Each grant is its own machine, each with its own card: nothing is
  // double-booked across them.
  EXPECT_EQ(map.nodes().size(), 3U);
  EXPECT_EQ(map.devicesInUse("hg-atlas-01"), std::vector<int>{0});
  EXPECT_TRUE(validate(map).empty());
}

TEST(ScalingPolicyTest, ASpareCountsSoCapacityIsNotOrderedOnTopOfIt)
{
  StateMap map = atlas();
  map.addReplica(replica("p0a", 0, {0}));
  Replica spare = replica("spare", 0, {1});
  spare.phase   = Phase::Reserved;
  map.addReplica(spare);

  ScalingConfig c = config();
  c.maxReplicas   = 2;

  // The spare is paid for and one message away, so ordering more while it
  // waits is waste.
  EXPECT_TRUE(decideScaling(map, busy({{"p0a", 99}}), c).empty());
}

// ---------------------------------------------------------------------------
// a scaled-up replica has to be reachable
// ---------------------------------------------------------------------------

TEST(ApplyScalingTest, GivesANewReplicaAPortSoItCanBeLaunchedAndProbed)
{
  // Port 0 is not a harmless default here: the launcher substitutes it into
  // the command, the probe dials it and the router routes to it, so a replica
  // granted without one sits pending for ever.
  StateMap map;
  map.addNode(Node{"hg-atlas-01", "10.0.0.1", {0, 1, 2}});
  Replica first = replica("r0", 0, {0});
  first.port    = 8000;
  map.addReplica(first);

  LocalCapacity provider(map);
  ScalingConfig config;
  config.minReplicas = 1;
  config.maxReplicas = 4;

  const auto outcomes = applyScaling(map, provider, {{0, ScalingAction::ScaleUp, "", "asked for"}}, config);
  ASSERT_EQ(outcomes.size(), 1U);
  ASSERT_TRUE(outcomes[0].applied) << outcomes[0].declined;

  const Replica *added = map.replica(outcomes[0].replica);
  ASSERT_NE(added, nullptr);
  EXPECT_EQ(added->port, 8001) << "one past the highest the deployment already uses";
  EXPECT_TRUE(validate(map).empty()) << "endpoints must stay unique";

  // And again, so two scale-ups in a row do not collide.
  const auto second = applyScaling(map, provider, {{0, ScalingAction::ScaleUp, "", "asked for"}}, config);
  ASSERT_TRUE(second[0].applied) << second[0].declined;
  EXPECT_EQ(map.replica(second[0].replica)->port, 8002);
  EXPECT_TRUE(validate(map).empty());
}
