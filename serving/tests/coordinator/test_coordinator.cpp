#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <serving/coordinator/module.hpp>

using serving::coordinator::Coordinator;
using serving::coordinator::CoordinatorConfig;
using serving::coordinator::ReplicaBinding;
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
  b.instanceId     = instanceId;
  return b;
}

std::vector<ReplicaBinding> pipeline()
{
  return {binding("p0a", 0, 10), binding("p0b", 0, 11), binding("p1a", 1, 20), binding("p1b", 1, 21), binding("p2a", 2, 30), binding("p2b", 2, 31)};
}

class RecordingTransport : public RuleTransport
{
  public:

  void publish(const ReplicaSpec &holder, const std::string &sessionId, const ReplicaSpec &nextHop, uint64_t) override
  {
    published.push_back(holder.name + "->" + nextHop.name + "/" + sessionId);
  }

  void revoke(const std::string &sessionId, uint64_t) override { revoked.push_back(sessionId); }

  std::vector<std::string> published;
  std::vector<std::string> revoked;
};

CoordinatorConfig config(RoutingMode mode)
{
  CoordinatorConfig c;
  c.partition = 1;
  c.mode      = mode;
  c.replicas  = pipeline();
  return c;
}

} // namespace

TEST(CoordinatorTest, IsNotUsableBeforeInitialize)
{
  RecordingTransport transport;
  Coordinator        coordinator(config(RoutingMode::DistributedRules), &transport);

  EXPECT_FALSE(coordinator.initialized());
  // The platform's engine constructs modules before it initializes them, so a
  // request arriving in that window must fail loudly rather than route nowhere.
  EXPECT_THROW((void)coordinator.routeFor("s1"), std::runtime_error);

  coordinator.initialize();
  EXPECT_TRUE(coordinator.initialized());
  EXPECT_NO_THROW((void)coordinator.routeFor("s1"));
}

TEST(CoordinatorTest, RefusesAnEmptyDeployment)
{
  CoordinatorConfig  empty;
  RecordingTransport transport;
  Coordinator        coordinator(empty, &transport);
  EXPECT_THROW(coordinator.initialize(), std::invalid_argument);
}

TEST(CoordinatorTest, MapsRoutingNamesToPlatformInstances)
{
  // Routing works in names, the platform in instance ids; the coordinator maps
  // between them.
  RecordingTransport transport;
  Coordinator        coordinator(config(RoutingMode::DistributedRules), &transport);
  coordinator.initialize();

  EXPECT_EQ(coordinator.instanceFor("p1a"), 20u);
  EXPECT_EQ(coordinator.instanceFor("p2b"), 31u);
  EXPECT_FALSE(coordinator.instanceFor("not-ours").has_value());
}

TEST(CoordinatorTest, RoutesAPathAcrossEveryPartition)
{
  RecordingTransport transport;
  Coordinator        coordinator(config(RoutingMode::DistributedRules), &transport);
  coordinator.initialize();

  const auto path = coordinator.routeFor("chat-1");
  ASSERT_EQ(path.hops.size(), 3u);
  EXPECT_EQ(path.hops[0].partition, 0);
  EXPECT_EQ(path.hops[2].partition, 2);
  EXPECT_EQ(coordinator.partition(), 1) << "it coordinates one partition but plans the whole path";
}

TEST(CoordinatorTest, CarriesTheLaunchModeThrough)
{
  RecordingTransport ingressTransport;
  Coordinator        ingress(config(RoutingMode::CoordinatorIngress), &ingressTransport);
  ingress.initialize();
  for (int i = 0; i < 3; ++i) { (void)ingress.routeFor("s1"); }
  EXPECT_EQ(ingress.mode(), RoutingMode::CoordinatorIngress);
  EXPECT_EQ(ingress.stats().plansComputed, 3);
  EXPECT_TRUE(ingressTransport.published.empty()) << "Option A publishes nothing";

  RecordingTransport rulesTransport;
  Coordinator        rules(config(RoutingMode::DistributedRules), &rulesTransport);
  rules.initialize();
  for (int i = 0; i < 3; ++i) { (void)rules.routeFor("s1"); }
  EXPECT_EQ(rules.mode(), RoutingMode::DistributedRules);
  EXPECT_EQ(rules.stats().plansComputed, 1) << "Option B routes once per session";
  EXPECT_EQ(rulesTransport.published.size(), 2u);
}

TEST(CoordinatorTest, AppliesAReplicaLossOnTheServiceTickNotInline)
{
  RecordingTransport transport;
  Coordinator        coordinator(config(RoutingMode::DistributedRules), &transport);
  coordinator.initialize();

  const auto        before = coordinator.routeFor("chat-1");
  const std::string lost   = before.hops[1].name;

  // A heartbeat reports the loss. Nothing happens yet: running routing policy on
  // the reporting thread would put control-plane timing in the request path.
  coordinator.reportReplicaLost(lost);
  EXPECT_EQ(coordinator.lossesApplied(), 0);
  EXPECT_TRUE(transport.revoked.empty());

  coordinator.service();
  EXPECT_EQ(coordinator.lossesApplied(), 1);
  ASSERT_EQ(transport.revoked.size(), 1u);
  EXPECT_EQ(transport.revoked.front(), "chat-1");

  const auto after = coordinator.routeFor("chat-1");
  EXPECT_NE(after.hops[1].name, lost);
  EXPECT_GT(after.generation, before.generation);
}

TEST(CoordinatorTest, ServiceIsHarmlessWithNothingReported)
{
  RecordingTransport transport;
  Coordinator        coordinator(config(RoutingMode::DistributedRules), &transport);

  // The engine ticks the service whether or not the module is up yet.
  EXPECT_NO_THROW(coordinator.service());
  coordinator.initialize();
  EXPECT_NO_THROW(coordinator.service());
  EXPECT_EQ(coordinator.lossesApplied(), 0);
}

TEST(CoordinatorTest, DrainsEveryReportedLossInOneTick)
{
  RecordingTransport transport;
  Coordinator        coordinator(config(RoutingMode::DistributedRules), &transport);
  coordinator.initialize();
  (void)coordinator.routeFor("chat-1");

  coordinator.reportReplicaLost("p1a");
  coordinator.reportReplicaLost("p1b");
  coordinator.service();
  EXPECT_EQ(coordinator.lossesApplied(), 2);

  // Partition 1 has nothing left, so the pipeline cannot be served at all.
  EXPECT_THROW((void)coordinator.routeFor("chat-2"), serving::router::NoReplicaAvailable);
}

TEST(CoordinatorTest, FinalizeReleasesAndIsIdempotent)
{
  RecordingTransport transport;
  Coordinator        coordinator(config(RoutingMode::DistributedRules), &transport);
  coordinator.initialize();
  (void)coordinator.routeFor("s1");

  coordinator.finalize();
  EXPECT_FALSE(coordinator.initialized());
  EXPECT_FALSE(coordinator.instanceFor("p1a").has_value());
  EXPECT_NO_THROW(coordinator.finalize());
  // After finalize the engine may still tick once before dropping the module.
  EXPECT_NO_THROW(coordinator.service());
}

TEST(CoordinatorTest, RoutesWhileTheServiceTickApplies)
{
  // routeFor runs on request threads and service() on the platform's timer;
  // both reach the same strategy, planner and session directory. Without a lock
  // this is concurrent mutation of std::map/std::list, so it corrupts rather
  // than merely returning something stale. Run under ThreadSanitizer.
  RecordingTransport transport;
  Coordinator        coordinator(config(RoutingMode::DistributedRules), &transport);
  coordinator.initialize();

  std::atomic<bool>        stop{false};
  std::atomic<int>         routed{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < 4; ++t)
  {
    threads.emplace_back([&, t] {
      for (int i = 0; i < 200 && !stop.load(); ++i)
      {
        try
        {
          (void)coordinator.routeFor("chat-" + std::to_string(t) + "-" + std::to_string(i % 8));
          routed.fetch_add(1);
        }
        catch (const serving::router::NoReplicaAvailable &)
        {} // every replica in a partition may be gone by now
      }
    });
  }
  threads.emplace_back([&] {
    for (int i = 0; i < 50; ++i)
    {
      coordinator.reportReplicaLost(i % 2 == 0 ? "p1a" : "p2b");
      coordinator.service();
      (void)coordinator.stats();
      (void)coordinator.instanceFor("p1a");
    }
  });
  for (std::thread &thread : threads) { thread.join(); }

  EXPECT_GT(routed.load(), 0);
  EXPECT_GT(coordinator.lossesApplied(), 0);
}
