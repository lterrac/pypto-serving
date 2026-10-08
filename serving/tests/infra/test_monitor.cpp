#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <serving/infra/capacity.hpp>
#include <serving/infra/monitor.hpp>
#include <serving/infra/scaling.hpp>

using serving::infra::HealthMonitor;
using serving::infra::HealthProbe;
using serving::infra::LocalCapacity;
using serving::infra::MonitorConfig;
using serving::infra::Node;
using serving::infra::Phase;
using serving::infra::Readiness;
using serving::infra::Replica;
using serving::infra::ReplicaStarter;
using serving::infra::StateMap;
using serving::infra::Supervisor;

namespace
{

StateMap fleet(int replicas)
{
  StateMap map;
  map.addNode(Node{"hg-atlas-01", "192.168.150.11", {0, 1, 2, 3, 4, 5, 6, 7}});
  for (int i = 0; i < replicas; ++i)
  {
    Replica r;
    r.name    = "qwen-" + std::to_string(i);
    r.node    = "hg-atlas-01";
    r.devices = {i};
    r.port    = 8000 + i;
    r.phase   = Phase::Active;
    map.addReplica(r);
  }
  return map;
}

/// Answers whatever the test last said.
class ScriptedProbe final : public HealthProbe
{
  public:

  Readiness check(const Replica &replica) override
  {
    ++probes;
    const auto it = answers.find(replica.name);
    return it == answers.end() ? Readiness::Ready : it->second;
  }

  std::map<std::string, Readiness> answers;
  int                              probes = 0;
};

class CountingStarter final : public ReplicaStarter
{
  public:

  bool start(const Replica &replica) override
  {
    started.push_back(replica.name);
    return true;
  }

  void        stop(const Replica &replica) override { stopped.push_back(replica.name); }
  const char *kind() const override { return "counting"; }

  std::vector<std::string> started;
  std::vector<std::string> stopped;
};

} // namespace

TEST(MonitorTest, AHealthyFleetIsLeftAlone)
{
  StateMap      map = fleet(3);
  ScriptedProbe probe;
  HealthMonitor monitor(map, probe);

  const auto report = monitor.service();
  EXPECT_TRUE(report.empty());
  EXPECT_EQ(report.probed, 3);
}

TEST(MonitorTest, OneBlipIsNotADeath)
{
  StateMap      map = fleet(2);
  ScriptedProbe probe;
  HealthMonitor monitor(map, probe, MonitorConfig{3});

  probe.answers["qwen-0"] = Readiness::Failed;
  EXPECT_TRUE(monitor.service().lost.empty());
  EXPECT_EQ(monitor.failures("qwen-0"), 1);
  // Declaring a healthy replica dead costs a model load, so one refused
  // connection is not enough.
  EXPECT_EQ(map.replica("qwen-0")->phase, Phase::Active);
}

TEST(MonitorTest, FailuresMustBeConsecutive)
{
  StateMap      map = fleet(2);
  ScriptedProbe probe;
  HealthMonitor monitor(map, probe, MonitorConfig{3});

  probe.answers["qwen-0"] = Readiness::Failed;
  monitor.service();
  monitor.service();
  ASSERT_EQ(monitor.failures("qwen-0"), 2);

  // It answered again, so the count goes. A replica has to fail every probe in
  // a row to be declared gone.
  probe.answers["qwen-0"] = Readiness::Ready;
  EXPECT_EQ(monitor.service().recovered, std::vector<std::string>{"qwen-0"});
  EXPECT_EQ(monitor.failures("qwen-0"), 0);

  probe.answers["qwen-0"] = Readiness::Failed;
  monitor.service();
  EXPECT_EQ(monitor.failures("qwen-0"), 1);
  EXPECT_EQ(map.replica("qwen-0")->phase, Phase::Active);
}

TEST(MonitorTest, DeclaresItDeadAfterTheThirdFailureAndFreesItsCard)
{
  StateMap      map = fleet(2);
  ScriptedProbe probe;
  HealthMonitor monitor(map, probe, MonitorConfig{3});
  ASSERT_EQ(map.devicesInUse("hg-atlas-01"), (std::vector<int>{0, 1}));

  probe.answers["qwen-0"] = Readiness::Failed;
  monitor.service();
  monitor.service();
  const auto third = monitor.service();

  EXPECT_EQ(third.lost, std::vector<std::string>{"qwen-0"});
  // Dead, not draining: draining is an orderly exit that keeps its devices,
  // and this replica is not exiting, it is gone.
  EXPECT_EQ(map.replica("qwen-0")->phase, Phase::Dead);
  EXPECT_EQ(map.devicesInUse("hg-atlas-01"), std::vector<int>{1});
}

TEST(MonitorTest, OnlyWatchesWhatIsSupposedToBeServing)
{
  StateMap map = fleet(1);
  Replica  loading;
  loading.name  = "loading";
  loading.node  = "hg-atlas-01";
  loading.phase = Phase::Pending;
  map.addReplica(loading);

  Replica leaving;
  leaving.name  = "leaving";
  leaving.node  = "hg-atlas-01";
  leaving.phase = Phase::Draining;
  map.addReplica(leaving);

  ScriptedProbe probe;
  HealthMonitor monitor(map, probe, MonitorConfig{1});
  probe.answers["loading"] = Readiness::Failed;
  probe.answers["leaving"] = Readiness::Failed;

  // A pending replica is the supervisor's business and is expected not to
  // answer yet; a draining one is leaving under its own steam.
  const auto report = monitor.service();
  EXPECT_EQ(report.probed, 1);
  EXPECT_TRUE(report.lost.empty());
  EXPECT_EQ(map.replica("loading")->phase, Phase::Pending);
  EXPECT_EQ(map.replica("leaving")->phase, Phase::Draining);
}

TEST(MonitorTest, StillLoadingIsNotStillAlive)
{
  StateMap      map = fleet(1);
  ScriptedProbe probe;
  HealthMonitor monitor(map, probe, MonitorConfig{2});

  // A replica that was serving and now says "loading" is not healthy: it is
  // either restarting under us or not answering. Either way it cannot serve.
  probe.answers["qwen-0"] = Readiness::Loading;
  monitor.service();
  EXPECT_EQ(monitor.service().lost, std::vector<std::string>{"qwen-0"});
}

TEST(MonitorTest, ForgetsAReplicaThatLeftForOtherReasons)
{
  StateMap      map = fleet(1);
  ScriptedProbe probe;
  HealthMonitor monitor(map, probe, MonitorConfig{3});

  probe.answers["qwen-0"] = Readiness::Failed;
  monitor.service();
  ASSERT_EQ(monitor.failures("qwen-0"), 1);

  // Drained by a scaling decision while it was wobbling.
  map.setPhase("qwen-0", Phase::Draining);
  monitor.service();
  EXPECT_EQ(monitor.failures("qwen-0"), 0);
}

TEST(MonitorConfigTest, RefusesAThresholdOfZero) { EXPECT_THROW(MonitorConfig{0}.validate(), std::invalid_argument); }

// ---------------------------------------------------------------------------
// the loop the deck describes, end to end
// ---------------------------------------------------------------------------

TEST(RecoveryTest, ALostReplicaIsNoticedTakenOutAndRebuilt)
{
  StateMap        map = fleet(2);
  ScriptedProbe   probe;
  CountingStarter starter;
  HealthMonitor   monitor(map, probe, MonitorConfig{2});
  Supervisor      supervisor(map, starter, probe);

  // Healthy: nothing happens on either tick.
  ASSERT_TRUE(monitor.service().empty());
  ASSERT_TRUE(supervisor.service().empty());

  // One of them goes.
  probe.answers["qwen-0"] = Readiness::Failed;
  monitor.service();
  ASSERT_EQ(monitor.service().lost, std::vector<std::string>{"qwen-0"});

  // Noticed: out of routing, card released, and the deployment still wants it.
  EXPECT_EQ(map.replica("qwen-0")->phase, Phase::Dead);
  EXPECT_EQ(map.replica("qwen-0")->desiredPhase, Phase::Active);
  EXPECT_EQ(map.devicesInUse("hg-atlas-01"), std::vector<int>{1});

  // The supervisor rebuilds it without being told anything beyond the map.
  EXPECT_EQ(supervisor.service().started, std::vector<std::string>{"qwen-0"});
  EXPECT_EQ(map.replica("qwen-0")->phase, Phase::Pending);

  // And it comes back.
  probe.answers["qwen-0"] = Readiness::Ready;
  EXPECT_EQ(supervisor.service().promoted, std::vector<std::string>{"qwen-0"});
  EXPECT_EQ(map.replica("qwen-0")->phase, Phase::Active);
  // Which the monitor then watches again like any other serving replica.
  EXPECT_EQ(monitor.service().probed, 2);
}

TEST(RecoveryTest, CapacityLostToADeathCanBeRebuiltElsewhere)
{
  StateMap      map = fleet(1);
  ScriptedProbe probe;
  HealthMonitor monitor(map, probe, MonitorConfig{1});
  LocalCapacity provider(map);

  // Before: card 0 is taken and seven are free.
  ASSERT_EQ(provider.available(serving::infra::CapacityRequest{0, 1, "", 0}), 7);

  probe.answers["qwen-0"] = Readiness::Failed;
  ASSERT_EQ(monitor.service().lost.size(), 1U);

  // After: the dead replica's card is available again, which is what lets the
  // deck's "a spare replica rebuilds the lost capacity" mean anything.
  EXPECT_EQ(provider.available(serving::infra::CapacityRequest{0, 1, "", 0}), 8);
  EXPECT_TRUE(validate(map).empty());
}
