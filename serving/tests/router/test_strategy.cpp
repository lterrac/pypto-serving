#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <serving/router/strategy.hpp>

using serving::router::CoordinatorIngressStrategy;
using serving::router::DistributedRuleStrategy;
using serving::router::makeRoutingStrategy;
using serving::router::NoReplicaAvailable;
using serving::router::parseRoutingMode;
using serving::router::ReplicaSpec;
using serving::router::RoutingConfig;
using serving::router::RoutingMode;
using serving::router::routingModeName;
using serving::router::RoutingPath;
using serving::router::RoutingPlanner;
using serving::router::RuleTransport;
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

/// Stands in for a real transport. `ChannelRuleTransport` is the one that puts a
/// rule on a platform channel; this records instead, so the strategies can be
/// tested without standing up HiCR.
class FakeTransport : public RuleTransport
{
  public:

  struct Rule
  {
    std::string holder;
    std::string sessionId;
    std::string nextHop;
    uint64_t    generation = 0;
  };

  void publish(const ReplicaSpec &holder, const std::string &sessionId, const ReplicaSpec &nextHop, uint64_t generation) override
  {
    published.push_back(Rule{holder.name, sessionId, nextHop.name, generation});
  }

  void revoke(const std::string &sessionId, uint64_t generation) override { revoked.push_back(Rule{"", sessionId, "", generation}); }

  void onReplicaUnreachable(const std::string &replicaName) override { unreachable.push_back(replicaName); }

  std::vector<std::string> unreachable;

  [[nodiscard]] std::vector<Rule> rulesFor(const std::string &sessionId) const
  {
    std::vector<Rule> out;
    for (const Rule &rule : published)
    {
      if (rule.sessionId == sessionId) { out.push_back(rule); }
    }
    return out;
  }

  std::vector<Rule> published;
  std::vector<Rule> revoked;
};

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
  EXPECT_EQ(parseRoutingMode("rules"), RoutingMode::DistributedRules);
  EXPECT_STREQ(routingModeName(RoutingMode::CoordinatorIngress), "ingress");
  EXPECT_STREQ(routingModeName(RoutingMode::DistributedRules), "rules");
  EXPECT_THROW((void)parseRoutingMode("something-else"), std::invalid_argument);
}

TEST(RoutingModeTest, TheRulesModeNeedsATransport)
{
  SessionDirectory sessions(600.0);
  RoutingPlanner   planner(RoutingConfig{}, sessions, pipeline());

  // Option A publishes nothing, so it needs no transport.
  EXPECT_NE(makeRoutingStrategy(RoutingMode::CoordinatorIngress, planner, nullptr), nullptr);
  // Option B without one would silently never tell a replica anything.
  EXPECT_THROW((void)makeRoutingStrategy(RoutingMode::DistributedRules, planner, nullptr), std::invalid_argument);
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
  EXPECT_EQ(strategy.stats().rulesPublished, 0) << "Option A must never publish a rule";
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
  EXPECT_NE(after.hops[1].name, before.hops[1].name);
  // Nothing needed invalidating: re-planning per request handles it for free.
  EXPECT_EQ(strategy.stats().pathsInvalidated, 0);
}

// ---------------------------------------------------------------------------
// Option B
// ---------------------------------------------------------------------------

TEST(DistributedRulesTest, RoutesOncePerSessionAndPublishesTheRules)
{
  SessionDirectory        sessions(600.0);
  RoutingPlanner          planner(RoutingConfig{}, sessions, pipeline());
  FakeTransport           transport;
  DistributedRuleStrategy strategy(planner, transport);

  const auto first = strategy.onRequest("s1");
  for (int turn = 0; turn < 5; ++turn)
  {
    // Later turns reuse the standing path without consulting the planner: the
    // decision has left the critical path.
    EXPECT_EQ(hopNames(strategy.onRequest("s1")), hopNames(first));
  }
  EXPECT_EQ(strategy.stats().plansComputed, 1) << "routing must happen once per session";

  // Three partitions means two forwarding rules: the last hop has nowhere to go.
  const auto rules = transport.rulesFor("s1");
  ASSERT_EQ(rules.size(), 2u);
  EXPECT_EQ(rules[0].holder, first.hops[0].name);
  EXPECT_EQ(rules[0].nextHop, first.hops[1].name);
  EXPECT_EQ(rules[1].holder, first.hops[1].name);
  EXPECT_EQ(rules[1].nextHop, first.hops[2].name);
}

TEST(DistributedRulesTest, DistinctSessionsGetTheirOwnPaths)
{
  SessionDirectory        sessions(600.0);
  RoutingPlanner          planner(RoutingConfig{}, sessions, pipeline());
  FakeTransport           transport;
  DistributedRuleStrategy strategy(planner, transport);

  (void)strategy.onRequest("s1");
  (void)strategy.onRequest("s2");
  EXPECT_EQ(strategy.stats().plansComputed, 2);
  EXPECT_EQ(strategy.activePaths(), 2u);
  EXPECT_EQ(transport.rulesFor("s1").size(), 2u);
  EXPECT_EQ(transport.rulesFor("s2").size(), 2u);
}

TEST(DistributedRulesTest, GenerationsIncreaseSoAStaleRuleIsRecognisable)
{
  SessionDirectory        sessions(600.0);
  RoutingPlanner          planner(RoutingConfig{}, sessions, pipeline());
  FakeTransport           transport;
  DistributedRuleStrategy strategy(planner, transport);

  const auto first  = strategy.onRequest("s1");
  const auto second = strategy.onRequest("s2");
  EXPECT_GT(second.generation, first.generation);

  for (const auto &rule : transport.rulesFor("s1")) { EXPECT_EQ(rule.generation, first.generation); }
}

TEST(DistributedRulesTest, ALostReplicaRevokesAndReplansOnlyTheAffectedSessions)
{
  SessionDirectory        sessions(600.0);
  RoutingPlanner          planner(RoutingConfig{}, sessions, pipeline());
  FakeTransport           transport;
  DistributedRuleStrategy strategy(planner, transport);

  const auto s1 = strategy.onRequest("s1");
  (void)strategy.onRequest("s2");
  const size_t before = transport.published.size();

  // Kill a replica s1 depends on.
  const std::string lost = s1.hops[1].name;
  strategy.onReplicaLost(lost);

  // Its rules are withdrawn rather than left forwarding into a dead node.
  EXPECT_EQ(strategy.stats().pathsInvalidated, 1);
  ASSERT_FALSE(transport.revoked.empty());
  EXPECT_EQ(transport.revoked.front().sessionId, "s1");

  // The next request for s1 replans, at a higher generation, avoiding the loss.
  const auto replanned = strategy.onRequest("s1");
  EXPECT_NE(replanned.hops[1].name, lost);
  EXPECT_GT(replanned.generation, s1.generation);
  EXPECT_GT(transport.published.size(), before) << "the new path must be published";
  EXPECT_EQ(strategy.stats().plansComputed, 3); // s1, s2, then s1 again
}

TEST(DistributedRulesTest, ASessionNotTouchingTheLostReplicaKeepsItsPath)
{
  SessionDirectory        sessions(600.0);
  RoutingPlanner          planner(RoutingConfig{}, sessions, pipeline());
  FakeTransport           transport;
  DistributedRuleStrategy strategy(planner, transport);

  const auto s1 = strategy.onRequest("s1");

  // Lose a replica in partition 1 that s1 does not use.
  const std::string unused = s1.hops[1].name == "p1a" ? "p1b" : "p1a";
  strategy.onReplicaLost(unused);

  EXPECT_EQ(strategy.stats().pathsInvalidated, 0);
  EXPECT_EQ(hopNames(strategy.onRequest("s1")), hopNames(s1));
}

// ---------------------------------------------------------------------------
// The two options side by side
// ---------------------------------------------------------------------------

TEST(RoutingOptionsTest, AgreeOnThePathButNotOnWhereTheDecisionHappens)
{
  const auto plannedBy = [](RoutingMode mode, FakeTransport &transport, int turns) {
    SessionDirectory sessions(600.0);
    RoutingPlanner   planner(RoutingConfig{}, sessions, pipeline());
    auto             strategy = makeRoutingStrategy(mode, planner, &transport);

    RoutingPath path;
    for (int i = 0; i < turns; ++i) { path = strategy->onRequest("s1"); }
    return std::pair{hopNames(path), strategy->stats()};
  };

  FakeTransport ingressTransport;
  FakeTransport rulesTransport;
  const auto [ingressPath, ingressStats] = plannedBy(RoutingMode::CoordinatorIngress, ingressTransport, 5);
  const auto [rulesPath, rulesStats]     = plannedBy(RoutingMode::DistributedRules, rulesTransport, 5);

  // Same policy, so a healthy pipeline routes a session identically either way.
  EXPECT_EQ(ingressPath, rulesPath);

  // The difference is entirely in where the decision is applied.
  EXPECT_EQ(ingressStats.plansComputed, 5);
  EXPECT_EQ(ingressStats.rulesPublished, 0);
  EXPECT_EQ(rulesStats.plansComputed, 1);
  EXPECT_EQ(rulesStats.rulesPublished, 2);
}

TEST(DistributedRulesTest, BoundsStandingPathsAndRevokesWhatItDrops)
{
  SessionDirectory        sessions(600.0);
  RoutingPlanner          planner(RoutingConfig{}, sessions, pipeline());
  FakeTransport           transport;
  DistributedRuleStrategy strategy(planner, transport, 3);

  // The key is client-supplied, so an unbounded map grows with whatever ids a
  // caller sends.
  for (int i = 0; i < 10; ++i) { (void)strategy.onRequest("s" + std::to_string(i)); }
  EXPECT_EQ(strategy.activePaths(), 3u);

  // Dropping a path without revoking would leave replicas forwarding along it.
  EXPECT_FALSE(transport.revoked.empty());
  EXPECT_EQ(transport.revoked.front().sessionId, "s0");
}

TEST(DistributedRulesTest, ReuseKeepsAPathFromBeingEvicted)
{
  SessionDirectory        sessions(600.0);
  RoutingPlanner          planner(RoutingConfig{}, sessions, pipeline());
  FakeTransport           transport;
  DistributedRuleStrategy strategy(planner, transport, 3);

  const auto kept = strategy.onRequest("keep");
  for (int i = 0; i < 6; ++i)
  {
    (void)strategy.onRequest("other" + std::to_string(i));
    (void)strategy.onRequest("keep"); // still in use
  }
  // Reused every round, so it is never the least recently used.
  EXPECT_EQ(hopNames(strategy.onRequest("keep")), hopNames(kept));
  EXPECT_EQ(strategy.stats().plansComputed, 7) << "the kept path was replanned";
}

TEST(DistributedRulesTest, TellsTheTransportAReplicaIsUnreachable)
{
  SessionDirectory        sessions(600.0);
  RoutingPlanner          planner(RoutingConfig{}, sessions, pipeline());
  FakeTransport           transport;
  DistributedRuleStrategy strategy(planner, transport);

  const auto path = strategy.onRequest("s1");
  strategy.onReplicaLost(path.hops[1].name);
  // The transport holds a channel per replica; writing to the lost one blocks
  // once it fills, because nothing is draining it.
  ASSERT_EQ(transport.unreachable.size(), 1u);
  EXPECT_EQ(transport.unreachable.front(), path.hops[1].name);
}
