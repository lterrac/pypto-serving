#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <serving/router/strategy.hpp>

using serving::router::CoordinatorIngressStrategy;
using serving::router::makeRoutingStrategy;
using serving::router::NoReplicaAvailable;
using serving::router::parseRoutingMode;
using serving::router::ReplicaSpec;
using serving::router::RoutingConfig;
using serving::router::RoutingMode;
using serving::router::routingModeName;
using serving::router::RoutingPath;
using serving::router::RoutingPlanner;
using serving::router::SessionDirectory;

namespace
{

ReplicaSpec replica(const std::string &name, int partition, int port)
{
  ReplicaSpec s;
  s.name      = name;
  s.partition = partition;
  s.host      = "127.0.0.1";
  s.port      = port;
  return s;
}

/// Three partitions of two replicas each: a pipeline wide enough that a path is
/// a real choice rather than a formality.
std::vector<ReplicaSpec> pipeline()
{
  return {replica("p0a", 0, 8000), replica("p0b", 0, 8001), replica("p1a", 1, 8002), replica("p1b", 1, 8003), replica("p2a", 2, 8004), replica("p2b", 2, 8005)};
}

std::vector<std::string> hopNames(const RoutingPath &path)
{
  std::vector<std::string> names;
  for (const ReplicaSpec &hop : path.hops) { names.push_back(hop.name); }
  return names;
}

} // namespace

// ---------------------------------------------------------------------------
// Mode selection
// ---------------------------------------------------------------------------

TEST(RoutingModeTest, RoundTripsItsNames)
{
  EXPECT_EQ(parseRoutingMode("ingress"), RoutingMode::CoordinatorIngress);
  EXPECT_STREQ(routingModeName(RoutingMode::CoordinatorIngress), "ingress");
  EXPECT_THROW((void)parseRoutingMode("something-else"), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Planning
// ---------------------------------------------------------------------------

TEST(RoutingPlannerTest, PlansOneHopPerPartitionInOrder)
{
  SessionDirectory sessions(600.0);
  RoutingPlanner   planner(RoutingConfig{}, sessions, pipeline());

  EXPECT_EQ(planner.partitions(), (std::vector<int>{0, 1, 2}));

  const auto path = planner.plan("s1");
  ASSERT_EQ(path.hops.size(), 3u);
  EXPECT_EQ(path.hops[0].partition, 0);
  EXPECT_EQ(path.hops[1].partition, 1);
  EXPECT_EQ(path.hops[2].partition, 2);
}

TEST(RoutingPlannerTest, KeepsASessionOnTheSamePathAcrossPartitions)
{
  // The property the whole design rests on: cached data is only reused when a
  // conversation follows the same path every turn.
  SessionDirectory sessions(600.0);
  RoutingPlanner   planner(RoutingConfig{}, sessions, pipeline());

  const auto first = planner.plan("s1");
  for (int turn = 0; turn < 5; ++turn) { EXPECT_EQ(hopNames(planner.plan("s1")), hopNames(first)); }
}

TEST(RoutingPlannerTest, AffinityIsPerPartition)
{
  // Each partition holds its own KV, so the pin has to be per partition. A
  // single shared pin would have each hop overwrite the previous one's.
  SessionDirectory sessions(600.0);
  RoutingPlanner   planner(RoutingConfig{}, sessions, pipeline());

  const auto path = planner.plan("s1");
  ASSERT_EQ(path.hops.size(), 3u);
  // Distinct replicas from distinct partitions -- not the same one three times.
  EXPECT_NE(path.hops[0].name, path.hops[1].name);
  EXPECT_NE(path.hops[1].name, path.hops[2].name);
}

TEST(RoutingPlannerTest, ABrokenStageMakesThePipelineUnservable)
{
  SessionDirectory sessions(600.0);
  RoutingPlanner   planner(RoutingConfig{}, sessions, pipeline());

  planner.setReady("p1a", false);
  planner.setReady("p1b", false);
  // Partition 1 has nothing routable, so no path exists at all.
  EXPECT_THROW((void)planner.plan("s1"), NoReplicaAvailable);
}

// ---------------------------------------------------------------------------
// Option A
// ---------------------------------------------------------------------------

TEST(CoordinatorIngressTest, PlansOnEveryRequest)
{
  SessionDirectory           sessions(600.0);
  RoutingPlanner             planner(RoutingConfig{}, sessions, pipeline());
  CoordinatorIngressStrategy strategy(planner);

  for (int i = 0; i < 5; ++i) { (void)strategy.onRequest("s1"); }

  // The coordinator is on the path anyway, so it decides afresh each time --
  // that is the "fully informed load balancing" the option buys, and the cost
  // it pays on the critical path.
  EXPECT_EQ(strategy.stats().plansComputed, 5);
}

TEST(CoordinatorIngressTest, ReroutesAfterAReplicaIsLost)
{
  SessionDirectory           sessions(600.0);
  RoutingPlanner             planner(RoutingConfig{}, sessions, pipeline());
  CoordinatorIngressStrategy strategy(planner);

  const auto before = strategy.onRequest("s1");
  strategy.onReplicaLost(before.hops[1].name);

  const auto after = strategy.onRequest("s1");
  ASSERT_EQ(after.hops.size(), 3u);
  // Nothing needed invalidating: re-planning per request handles it for free.
  EXPECT_NE(after.hops[1].name, before.hops[1].name);
}

TEST(CoordinatorIngressTest, IsBuiltByTheFactory)
{
  SessionDirectory sessions(600.0);
  RoutingPlanner   planner(RoutingConfig{}, sessions, pipeline());

  const auto strategy = makeRoutingStrategy(RoutingMode::CoordinatorIngress, planner);
  ASSERT_NE(strategy, nullptr);
  EXPECT_EQ(strategy->mode(), RoutingMode::CoordinatorIngress);
}
