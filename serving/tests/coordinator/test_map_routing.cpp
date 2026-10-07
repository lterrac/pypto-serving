#include <algorithm>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <serving/coordinator/map_routing.hpp>
#include <serving/coordinator/module.hpp>

using serving::coordinator::bindingsFrom;
using serving::coordinator::Coordinator;
using serving::coordinator::CoordinatorConfig;
using serving::coordinator::ReplicaBinding;
using serving::infra::Node;
using serving::infra::Phase;
using serving::infra::Replica;
using serving::infra::StateMap;
using serving::router::NoReplicaAvailable;
using serving::router::ReplicaSpec;
using serving::router::RoutingMode;
using serving::router::RuleTransport;

namespace
{

ReplicaBinding binding(const std::string &name, int partition, uint64_t instanceId)
{
  ReplicaBinding b;
  b.spec.name      = name;
  b.spec.partition = partition;
  b.spec.host      = "node-" + std::to_string(partition);
  b.spec.port      = 8000 + static_cast<int>(instanceId);
  b.instanceId     = instanceId;
  return b;
}

/// Two partitions, two replicas each: enough that one can be taken out without
/// the pipeline losing a stage.
CoordinatorConfig config(RoutingMode mode = RoutingMode::CoordinatorIngress)
{
  CoordinatorConfig c;
  c.partition = 0;
  c.mode      = mode;
  c.replicas  = {binding("p0a", 0, 10), binding("p0b", 0, 11), binding("p1a", 1, 20), binding("p1b", 1, 21)};
  return c;
}

class RecordingTransport : public RuleTransport
{
  public:

  void publish(const ReplicaSpec &, const std::string &, const ReplicaSpec &, uint64_t) override {}
  void revoke(const std::string &sessionId, uint64_t) override { revoked.push_back(sessionId); }

  std::vector<std::string> revoked;
};

/// Which replicas a run of requests actually reached, in partition 0.
///
/// Every call uses session ids it has not used before: a session is pinned to
/// the replica that served it, so reusing one measures affinity rather than
/// which replicas are routable.
std::vector<std::string> hopsSeen(Coordinator &coordinator, int requests)
{
  static int               next = 0;
  std::vector<std::string> seen;
  for (int i = 0; i < requests; ++i)
  {
    const auto path = coordinator.routeFor("session-" + std::to_string(next++));
    if (!path.hops.empty() && std::find(seen.begin(), seen.end(), path.hops.front().name) == seen.end()) { seen.push_back(path.hops.front().name); }
  }
  std::sort(seen.begin(), seen.end());
  return seen;
}

} // namespace

TEST(MapRoutingTest, TranslatesTheMapIntoEndpoints)
{
  StateMap map;
  map.addNode(Node{"hg-atlas-01", "192.168.150.11", {0, 1, 2, 3}});
  Replica r;
  r.name      = "qwen-0";
  r.partition = 2;
  r.node      = "hg-atlas-01";
  r.port      = 8000;
  r.instance  = 77;
  map.addReplica(r);

  const auto bindings = bindingsFrom(map);
  ASSERT_EQ(bindings.size(), 1U);
  // The address comes off the node; the replica only carries its port.
  EXPECT_EQ(bindings[0].spec.host, "192.168.150.11");
  EXPECT_EQ(bindings[0].spec.port, 8000);
  EXPECT_EQ(bindings[0].spec.partition, 2);
  EXPECT_EQ(bindings[0].instanceId, 77U);
}

TEST(MapRoutingTest, AReplicaWithNoNodeHasNoEndpointAndIsSkipped)
{
  StateMap map;
  Replica  r;
  r.name = "orphan";
  r.node = "gone";
  map.addReplica(r);

  EXPECT_TRUE(bindingsFrom(map).empty());
}

TEST(MapSyncTest, RoutingAndTheMapAgreeFromTheStart)
{
  Coordinator coordinator(config(), nullptr);
  coordinator.initialize();

  // The configuration is a statement that these replicas should be serving, so
  // the map says so too, and the first tick has nothing to do.
  EXPECT_EQ(coordinator.stateMap().replicas().size(), 4U);
  EXPECT_EQ(coordinator.syncedVersion(), coordinator.stateMap().version());
  EXPECT_TRUE(coordinator.syncFromMap().empty());
  EXPECT_EQ(hopsSeen(coordinator, 8), (std::vector<std::string>{"p0a", "p0b"}));
}

TEST(MapSyncTest, DrainingInTheMapTakesAReplicaOutOfRouting)
{
  Coordinator coordinator(config(), nullptr);
  coordinator.initialize();
  ASSERT_EQ(hopsSeen(coordinator, 8).size(), 2U);

  coordinator.stateMap().setPhase("p0b", Phase::Draining);
  const auto report = coordinator.syncFromMap();
  EXPECT_EQ(report.rephased, 1);

  // It keeps what it has but takes nothing new.
  EXPECT_EQ(hopsSeen(coordinator, 8), (std::vector<std::string>{"p0a"}));
}

TEST(MapSyncTest, PendingIsNotRoutableAndActivatingItIs)
{
  Coordinator coordinator(config(), nullptr);
  coordinator.initialize();

  coordinator.stateMap().setPhase("p0b", Phase::Pending);
  coordinator.syncFromMap();
  EXPECT_EQ(hopsSeen(coordinator, 8), (std::vector<std::string>{"p0a"}));

  // A replica that finished loading says so by going active.
  coordinator.stateMap().setPhase("p0b", Phase::Active);
  EXPECT_EQ(coordinator.syncFromMap().rephased, 1);
  EXPECT_EQ(hopsSeen(coordinator, 8), (std::vector<std::string>{"p0a", "p0b"}));
}

TEST(MapSyncTest, DeadTakesTheLossPathAndRevokesItsPaths)
{
  RecordingTransport transport;
  Coordinator        coordinator(config(RoutingMode::DistributedRules), &transport);
  coordinator.initialize();

  // Plan a session so there is a path to lose.
  const auto path = coordinator.routeFor("chat-1");
  ASSERT_FALSE(path.hops.empty());
  const std::string holder = path.hops.front().name;

  coordinator.stateMap().setPhase(holder, Phase::Dead);
  const auto report = coordinator.syncFromMap();

  EXPECT_EQ(report.lost, std::vector<std::string>{holder});
  // Dead is the loss path, not a flag: a replica carrying a stale rule has to
  // hear that the session moved.
  EXPECT_FALSE(transport.revoked.empty());
  // And nothing routes there any more.
  const auto seen = hopsSeen(coordinator, 8);
  EXPECT_EQ(std::find(seen.begin(), seen.end(), holder), seen.end());
}

TEST(MapSyncTest, AReplicaAddedToTheMapStartsCarryingTraffic)
{
  Coordinator coordinator(config(), nullptr);
  coordinator.initialize();
  ASSERT_EQ(hopsSeen(coordinator, 12).size(), 2U);

  Replica fresh;
  fresh.name      = "p0c";
  fresh.partition = 0;
  fresh.node      = "node-0";
  fresh.port      = 8099;
  fresh.phase     = Phase::Active;
  coordinator.stateMap().addReplica(fresh);

  EXPECT_EQ(coordinator.syncFromMap().added, 1);
  EXPECT_EQ(hopsSeen(coordinator, 24), (std::vector<std::string>{"p0a", "p0b", "p0c"}));
}

TEST(MapSyncTest, AReplicaRemovedFromTheMapStopsCarryingTraffic)
{
  Coordinator coordinator(config(), nullptr);
  coordinator.initialize();

  EXPECT_TRUE(coordinator.stateMap().removeReplica("p0b"));
  EXPECT_EQ(coordinator.syncFromMap().removed, 1);
  EXPECT_EQ(hopsSeen(coordinator, 8), (std::vector<std::string>{"p0a"}));
}

TEST(MapSyncTest, AReplicaInAPartitionRoutingDoesNotKnowIsReportedNotDropped)
{
  Coordinator coordinator(config(), nullptr);
  coordinator.initialize();

  Replica stray;
  stray.name      = "p9a";
  stray.partition = 9;
  stray.node      = "node-0";
  stray.phase     = Phase::Active;
  coordinator.stateMap().addReplica(stray);

  // Routing was built for a fixed pipeline; growing it is repartitioning, so
  // this is surfaced rather than silently ignored.
  const auto report = coordinator.syncFromMap();
  EXPECT_EQ(report.unplaceable, std::vector<std::string>{"p9a"});
  EXPECT_EQ(report.added, 0);
}

TEST(MapSyncTest, TheServiceTickAppliesAMapEditWithoutAnyoneReportingIt)
{
  Coordinator coordinator(config(), nullptr);
  coordinator.initialize();

  coordinator.stateMap().setPhase("p0b", Phase::Draining);
  // Nothing called reportReplicaLost; the tick notices the version moved.
  coordinator.service();

  EXPECT_EQ(coordinator.syncedVersion(), coordinator.stateMap().version());
  EXPECT_EQ(hopsSeen(coordinator, 8), (std::vector<std::string>{"p0a"}));
}

TEST(MapSyncTest, AnUntouchedMapCostsNothingPerTick)
{
  Coordinator coordinator(config(), nullptr);
  coordinator.initialize();
  const uint64_t version = coordinator.stateMap().version();

  for (int i = 0; i < 100; ++i) { coordinator.service(); }

  // The version is what makes a quiet tick cheap: no edit, no walk.
  EXPECT_EQ(coordinator.stateMap().version(), version);
  EXPECT_EQ(coordinator.syncedVersion(), version);
}

TEST(MapSyncTest, LosingEveryReplicaOfAPartitionLeavesThePipelineUnservable)
{
  Coordinator coordinator(config(), nullptr);
  coordinator.initialize();

  coordinator.stateMap().setPhase("p1a", Phase::Dead);
  coordinator.stateMap().setPhase("p1b", Phase::Dead);
  coordinator.syncFromMap();

  // A pipeline with a broken stage cannot serve at all, and says so rather than
  // routing to a replica that is gone.
  EXPECT_THROW((void)coordinator.routeFor("chat-1"), NoReplicaAvailable);
}

// ---------------------------------------------------------------------------
// The document as the thing two processes agree through
// ---------------------------------------------------------------------------

#include <serving/infra/document.hpp>

TEST(MapSyncTest, AdoptingADocumentRePlansRouting)
{
  Coordinator coordinator(config(), nullptr);
  coordinator.initialize();
  ASSERT_EQ(hopsSeen(coordinator, 8).size(), 2U);

  // Something else -- a scaling policy, an operator, the replica itself --
  // wrote a document saying one of them is draining. Adopting it is the whole
  // integration: the coordinator does not learn this through an API call.
  serving::infra::StateMap wanted = coordinator.stateMap();
  wanted.setPhase("p0b", Phase::Draining);
  const std::string document = serving::infra::writeDocument(wanted);

  coordinator.stateMap() = serving::infra::readDocument(document);
  coordinator.service();

  EXPECT_EQ(hopsSeen(coordinator, 8), (std::vector<std::string>{"p0a"}));
  EXPECT_EQ(coordinator.syncedVersion(), coordinator.stateMap().version());
}

TEST(MapSyncTest, ADocumentThatRemovesAReplicaTakesItOutOfRouting)
{
  Coordinator coordinator(config(), nullptr);
  coordinator.initialize();

  serving::infra::StateMap wanted = coordinator.stateMap();
  ASSERT_TRUE(wanted.removeReplica("p0b"));

  coordinator.stateMap() = serving::infra::readDocument(serving::infra::writeDocument(wanted));
  coordinator.service();

  EXPECT_EQ(hopsSeen(coordinator, 8), (std::vector<std::string>{"p0a"}));
}
