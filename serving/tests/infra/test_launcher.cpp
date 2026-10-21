#include <chrono>
#include <thread>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <serving/infra/launcher.hpp>

using serving::infra::Backoff;
using serving::infra::HealthProbe;
using serving::infra::Node;
using serving::infra::Phase;
using serving::infra::Replica;
using serving::infra::ReplicaStarter;
using serving::infra::StateMap;
using serving::infra::Supervisor;

namespace
{

StateMap atlas()
{
  StateMap map;
  map.addNode(Node{"hg-atlas-01", "192.168.150.11", {0, 1, 2, 3, 4, 5, 6, 7}});
  return map;
}

Replica wanted(const std::string &name, Phase phase = Phase::Pending)
{
  Replica r;
  r.name         = name;
  r.node         = "hg-atlas-01";
  r.devices      = {0};
  r.port         = 8000;
  r.phase        = phase;
  r.desiredPhase = Phase::Active;
  return r;
}

/// A starter that can be told to fail, and counts what it was asked to do.
class FakeStarter final : public ReplicaStarter
{
  public:

  bool start(const Replica &replica) override
  {
    starts.push_back(replica.name);
    return !refuse;
  }

  void        stop(const Replica &replica) override { stops.push_back(replica.name); }
  const char *kind() const override { return "fake"; }

  bool                     refuse = false;
  std::vector<std::string> starts;
  std::vector<std::string> stops;
};

/// Ready only once told, so a test can keep a replica loading.
class FakeProbe final : public HealthProbe
{
  public:

  serving::infra::Readiness check(const Replica &replica) override
  {
    if (healthy.count(replica.name) > 0) { return serving::infra::Readiness::Ready; }
    return failed.count(replica.name) > 0 ? serving::infra::Readiness::Failed : serving::infra::Readiness::Loading;
  }

  std::set<std::string> healthy;
  std::set<std::string> failed;
};

/// A clock a test drives by hand: the whole of backoff is time passing, and a
/// test that slept through it is one nobody runs.
struct TestClock
{
  double now = 0.0;
  double operator()() const { return now; }
};

} // namespace

TEST(BackoffTest, GrowsFromTheBaseAndStopsAtTheCeiling)
{
  const Backoff backoff(1.0, 10.0, 6);

  // The first try is not a retry.
  EXPECT_EQ(backoff.delayAfter(0), 0.0);
  EXPECT_EQ(backoff.delayAfter(1), 1.0);
  EXPECT_EQ(backoff.delayAfter(2), 2.0);
  EXPECT_EQ(backoff.delayAfter(3), 4.0);
  EXPECT_EQ(backoff.delayAfter(4), 8.0);
  EXPECT_EQ(backoff.delayAfter(5), 10.0) << "should be capped";
  EXPECT_EQ(backoff.delayAfter(500), 10.0) << "and stay capped without overflowing";
}

TEST(BackoffTest, RefusesSettingsThatCannotWork)
{
  EXPECT_THROW(Backoff(0.0, 10.0, 3), std::invalid_argument);
  EXPECT_THROW(Backoff(10.0, 1.0, 3), std::invalid_argument);
  EXPECT_THROW(Backoff(1.0, 10.0, 0), std::invalid_argument);
}

TEST(SupervisorTest, StartsWhatTheDeploymentWantsAndWaitsForItToServe)
{
  StateMap map = atlas();
  map.addReplica(wanted("qwen-0"));
  FakeStarter starter;
  FakeProbe   probe;
  TestClock   clock;
  Supervisor  supervisor(map, starter, probe, Backoff(1.0, 10.0, 3), [&clock] { return clock(); });

  const auto first = supervisor.service();
  EXPECT_EQ(first.started, std::vector<std::string>{"qwen-0"});
  // Started is not serving: the model takes minutes, and routing must not be
  // sent there meanwhile.
  EXPECT_EQ(map.replica("qwen-0")->phase, Phase::Pending);

  clock.now = 30.0;
  EXPECT_EQ(supervisor.service().waiting, 1);
  EXPECT_EQ(map.replica("qwen-0")->phase, Phase::Pending);

  // Its own /health is the authority.
  probe.healthy.insert("qwen-0");
  EXPECT_EQ(supervisor.service().promoted, std::vector<std::string>{"qwen-0"});
  EXPECT_EQ(map.replica("qwen-0")->phase, Phase::Active);
}

TEST(SupervisorTest, AServingReplicaIsLeftAlone)
{
  StateMap map = atlas();
  map.addReplica(wanted("qwen-0", Phase::Active));
  FakeStarter starter;
  FakeProbe   probe;
  Supervisor  supervisor(map, starter, probe);

  EXPECT_TRUE(supervisor.service().empty());
  EXPECT_TRUE(starter.starts.empty());
}

TEST(SupervisorTest, RetriesAFailedLaunchAfterAGrowingInterval)
{
  StateMap map = atlas();
  map.addReplica(wanted("qwen-0"));
  FakeStarter starter;
  starter.refuse = true;
  FakeProbe  probe;
  TestClock  clock;
  Supervisor supervisor(map, starter, probe, Backoff(1.0, 100.0, 5), [&clock] { return clock(); });

  EXPECT_EQ(supervisor.service().retrying.size(), 1U);
  EXPECT_EQ(supervisor.attempts("qwen-0"), 1);

  // Still inside the first interval: it is not hammered.
  clock.now = 0.5;
  EXPECT_EQ(supervisor.service().waiting, 1);
  EXPECT_EQ(starter.starts.size(), 1U);

  clock.now = 1.0;
  EXPECT_EQ(supervisor.service().retrying.size(), 1U);
  EXPECT_EQ(starter.starts.size(), 2U);

  // ...and the next interval is longer than the last.
  clock.now = 2.0;
  EXPECT_EQ(supervisor.service().waiting, 1) << "second interval should be 2s, not 1s";
  clock.now = 3.0;
  EXPECT_EQ(supervisor.service().retrying.size(), 1U);
}

TEST(SupervisorTest, GivesUpAndMarksItDeadSoTheCardComesBack)
{
  StateMap map = atlas();
  map.addReplica(wanted("qwen-0"));
  FakeStarter starter;
  starter.refuse = true;
  FakeProbe  probe;
  TestClock  clock;
  Supervisor supervisor(map, starter, probe, Backoff(1.0, 10.0, 3), [&clock] { return clock(); });

  for (int i = 0; i < 3; ++i)
  {
    supervisor.service();
    clock.now += 100.0;
  }

  // Dead rather than stuck pending: dead releases its devices, so the capacity
  // can be rebuilt somewhere that might work.
  EXPECT_EQ(map.replica("qwen-0")->phase, Phase::Dead);
  EXPECT_TRUE(map.devicesFree("hg-atlas-01").size() == 8U);
}

TEST(SupervisorTest, AReplicaThatStartsButNeverComesUpCountsAsAFailure)
{
  StateMap map = atlas();
  map.addReplica(wanted("qwen-0"));
  FakeStarter starter;
  FakeProbe   probe; // never healthy
  TestClock   clock;
  Supervisor  supervisor(map, starter, probe, Backoff(1.0, 10.0, 3), [&clock] { return clock(); });
  supervisor.setReadyTimeout(60.0);

  supervisor.service();
  ASSERT_EQ(starter.starts.size(), 1U);

  clock.now = 59.0;
  EXPECT_EQ(supervisor.service().waiting, 1);

  // Past the timeout the attempt is abandoned -- and the half-started process
  // is stopped, or the next attempt would fight it for the card.
  clock.now = 61.0;
  EXPECT_EQ(supervisor.service().retrying.size(), 1U);
  EXPECT_EQ(starter.stops, std::vector<std::string>{"qwen-0"});
  EXPECT_EQ(supervisor.attempts("qwen-0"), 1);
}

TEST(SupervisorTest, RebuildsCapacityLostToADeadReplica)
{
  StateMap map = atlas();
  map.addReplica(wanted("qwen-0", Phase::Active));
  FakeStarter starter;
  FakeProbe   probe;
  TestClock   clock;
  Supervisor  supervisor(map, starter, probe, Backoff(1.0, 10.0, 3), [&clock] { return clock(); });
  ASSERT_TRUE(supervisor.service().empty());

  // Something noticed it is gone. The deployment still wants it.
  map.setPhase("qwen-0", Phase::Dead);

  EXPECT_EQ(supervisor.service().started, std::vector<std::string>{"qwen-0"});
  // Back to pending, not straight to active: it has to come up again.
  EXPECT_EQ(map.replica("qwen-0")->phase, Phase::Pending);

  probe.healthy.insert("qwen-0");
  supervisor.service();
  EXPECT_EQ(map.replica("qwen-0")->phase, Phase::Active);
}

TEST(SupervisorTest, StopsWhatTheDeploymentNoLongerWants)
{
  StateMap map = atlas();
  map.addReplica(wanted("qwen-0", Phase::Active));
  FakeStarter starter;
  FakeProbe   probe;
  Supervisor  supervisor(map, starter, probe);

  map.setDesiredPhase("qwen-0", Phase::Dead);
  EXPECT_EQ(supervisor.service().stopped, std::vector<std::string>{"qwen-0"});
  EXPECT_EQ(starter.stops, std::vector<std::string>{"qwen-0"});
  EXPECT_EQ(map.replica("qwen-0")->phase, Phase::Dead);
}

TEST(SupervisorTest, LeavesDrainingToRouting)
{
  StateMap map = atlas();
  map.addReplica(wanted("qwen-0", Phase::Active));
  FakeStarter starter;
  FakeProbe   probe;
  Supervisor  supervisor(map, starter, probe);

  map.setDesiredPhase("qwen-0", Phase::Draining);
  // Draining keeps serving what it has and takes nothing new; no process work.
  EXPECT_TRUE(supervisor.service().empty());
  EXPECT_TRUE(starter.starts.empty());
  EXPECT_TRUE(starter.stops.empty());
}

TEST(SupervisorTest, StartsSeveralPartitionsInOneTick)
{
  StateMap map = atlas();
  Replica  a   = wanted("p0a");
  Replica  b   = wanted("p1a");
  b.partition  = 1;
  b.devices    = {1};
  b.port       = 8001;
  map.addReplica(a);
  map.addReplica(b);

  FakeStarter starter;
  FakeProbe   probe;
  Supervisor  supervisor(map, starter, probe);

  EXPECT_EQ(supervisor.service().started.size(), 2U);
}

// ---------------------------------------------------------------------------
// the real starter
// ---------------------------------------------------------------------------

#include <filesystem>
#include <fstream>
#include <csignal>
#include <unistd.h>

#include <serving/infra/process_starter.hpp>

using serving::infra::ProcessStarter;

TEST(ProcessStarterTest, SubstitutesTheReplicaIntoTheCommand)
{
  const ProcessStarter starter("serve --name {name} --device {device} --port {port} --all {devices} --on {node}");

  Replica replica = wanted("qwen-0");
  replica.devices = {4, 5};
  replica.port    = 8123;

  EXPECT_EQ(starter.commandFor(replica), "serve --name qwen-0 --device 4 --port 8123 --all 4,5 --on hg-atlas-01");
}

TEST(ProcessStarterTest, DeviceIsTheFirstCardBecauseOneIsTheCommonCase)
{
  const ProcessStarter starter("--device {device}");
  Replica              replica = wanted("qwen-0");
  replica.devices              = {6};
  EXPECT_EQ(starter.commandFor(replica), "--device 6");

  replica.devices = {};
  EXPECT_EQ(starter.commandFor(replica), "--device ");
}

TEST(ProcessStarterTest, ActuallyRunsTheCommandAndReapsIt)
{
  const auto marker = std::filesystem::temp_directory_path() / ("serving-starter-" + std::to_string(::getpid()) + ".txt");
  std::filesystem::remove(marker);

  ProcessStarter starter("echo {name} > " + marker.string());
  Replica        replica = wanted("qwen-0");

  ASSERT_TRUE(starter.start(replica));
  // It exits immediately, so running() both reports that and reaps it.
  for (int i = 0; i < 200 && starter.running("qwen-0"); ++i) { std::this_thread::sleep_for(std::chrono::milliseconds(10)); }
  EXPECT_FALSE(starter.running("qwen-0"));

  ASSERT_TRUE(std::filesystem::exists(marker));
  std::ifstream in(marker);
  std::string   wrote;
  in >> wrote;
  EXPECT_EQ(wrote, "qwen-0");
  std::filesystem::remove(marker);
}

TEST(ProcessStarterTest, RefusesToStartOneTwiceOverItsOwnPid)
{
  ProcessStarter starter("sleep 30");
  Replica        replica = wanted("qwen-0");

  ASSERT_TRUE(starter.start(replica));
  // Starting again would lose the first pid and leave it holding its card.
  EXPECT_FALSE(starter.start(replica));
  starter.stop(replica);
  EXPECT_FALSE(starter.running("qwen-0"));
}

TEST(ProcessStarterTest, StopKillsIt)
{
  ProcessStarter starter("sleep 60");
  Replica        replica = wanted("qwen-0");

  ASSERT_TRUE(starter.start(replica));
  EXPECT_TRUE(starter.running("qwen-0"));
  starter.stop(replica);
  EXPECT_FALSE(starter.running("qwen-0"));
  // Stopping something that was never started is a no-op, not a crash.
  starter.stop(replica);
}

TEST(ProcessStarterTest, StopReachesAChildThatLeftTheProcessGroup)
{
  // What simpler's per-chip workers do as they fork: os.setpgid(0, 0), which
  // makes each its own group leader (simpler/worker.py:8127). Measured on
  // hardware, the chip worker's pgid is its own pid while the wrapper's group
  // is the shell's -- so `kill(-pgid)` reaches the wrapper and the python and
  // misses the process holding the card. setpgid moves a process between
  // groups within its session and cannot leave it, so the session still holds
  // it; this is the sweep that finds it.
  //
  // The worker writes a marker while it lives and removes it on the way out, so
  // the assertion is about the process actually being gone rather than about
  // what was signalled.
  const auto marker = std::filesystem::temp_directory_path() / "serving-regrouped-child.pid";
  std::filesystem::remove(marker);

  // Regroup, record, then outlive any plausible test. A `sleep` in its own
  // group is exactly the shape that escaped the old stop().
  ProcessStarter starter("sh -c 'python3 -c \"import os,time;os.setpgid(0,0);open(\\\"" + marker.string() +
                         "\\\",\\\"w\\\").write(str(os.getpid()));time.sleep(120)\" & sleep 120'");
  Replica        replica = wanted("qwen-0");

  ASSERT_TRUE(starter.start(replica));

  // Wait for the regrouped child to exist before stopping anything.
  for (int waited = 0; waited < 100 && !std::filesystem::exists(marker); ++waited) { std::this_thread::sleep_for(std::chrono::milliseconds(50)); }
  ASSERT_TRUE(std::filesystem::exists(marker)) << "the regrouped child never started; the test proves nothing";

  std::ifstream read(marker);
  pid_t         child = 0;
  read >> child;
  ASSERT_GT(child, 0);
  // It really did leave the group it was started in.
  EXPECT_NE(::getpgid(child), ::getpgid(0));

  starter.stop(replica);

  // Gone, not merely signalled.
  EXPECT_NE(::kill(child, 0), 0) << "a worker that regrouped itself survived stop()";
  EXPECT_FALSE(starter.running("qwen-0"));
  std::filesystem::remove(marker);
}

TEST(SupervisorTest, AProcessThatDiesIsAFailedAttemptNotAnEndlessWait)
{
  StateMap map = atlas();
  map.addReplica(wanted("qwen-0"));

  ProcessStarter                    starter("true"); // exits at once
  FakeProbe                         never;
  serving::infra::LiveAndReadyProbe probe(starter, never);
  TestClock                         clock;
  Supervisor                        supervisor(map, starter, probe, Backoff(1.0, 10.0, 2), [&clock] { return clock(); });
  supervisor.setReadyTimeout(3600.0);

  ASSERT_EQ(supervisor.service().started.size(), 1U);
  for (int i = 0; i < 200 && starter.running("qwen-0"); ++i) { std::this_thread::sleep_for(std::chrono::milliseconds(10)); }

  // Health alone cannot tell "still loading" from "already exited": both refuse
  // the connection. Without the liveness half this would wait out the full
  // ready timeout -- an hour -- before retrying.
  clock.now         = 1.0;
  const auto report = supervisor.service();
  EXPECT_FALSE(report.retrying.empty() && report.gaveUp.empty()) << "a dead process should not read as still loading";
}

TEST(SupervisorTest, AProbeThatSaysFailedDoesNotWaitOutTheTimeout)
{
  StateMap map = atlas();
  map.addReplica(wanted("qwen-0"));
  FakeStarter starter;
  FakeProbe   probe;
  TestClock   clock;
  Supervisor  supervisor(map, starter, probe, Backoff(1.0, 10.0, 3), [&clock] { return clock(); });
  supervisor.setReadyTimeout(3600.0);

  ASSERT_EQ(supervisor.service().started.size(), 1U);
  probe.failed.insert("qwen-0");

  // An hour of ready timeout, but the probe already knows. The difference
  // between a slow deployment and a stuck one.
  clock.now = 1.0;
  EXPECT_EQ(supervisor.service().retrying, std::vector<std::string>{"qwen-0"});
  EXPECT_EQ(supervisor.attempts("qwen-0"), 1);
}

TEST(HealthProbeTest, ResolvesANodeNameToItsAddress)
{
  StateMap map = atlas();
  map.addReplica(wanted("qwen-0"));

  // A node name is not an address: inside a container it is whatever the
  // deployment chose to call the machine, and connecting to it would fail.
  const serving::infra::HttpHealthProbe resolving(&map);
  EXPECT_EQ(resolving.addressOf(*map.replica("qwen-0")), "192.168.150.11");

  // Without a map the name is all there is, which is right only where the two
  // coincide.
  const serving::infra::HttpHealthProbe bare;
  EXPECT_EQ(bare.addressOf(*map.replica("qwen-0")), "hg-atlas-01");
}

TEST(ProcessStarterTest, StopReachesWhatTheWrapperSpawned)
{
  const auto marker = std::filesystem::temp_directory_path() / ("serving-grandchild-" + std::to_string(::getpid()) + ".txt");
  std::filesystem::remove(marker);

  // A template that spawns rather than execs, which is what a wrapper script
  // does. Signalling only the shell would leave the grandchild running --
  // holding a card, in production.
  ProcessStarter starter("( sleep 30; echo leaked > " + marker.string() + " ) & wait");
  Replica        replica = wanted("qwen-0");

  ASSERT_TRUE(starter.start(replica));
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  starter.stop(replica);

  // Give the grandchild longer than it would have needed to write.
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  EXPECT_FALSE(starter.running("qwen-0"));
  EXPECT_FALSE(std::filesystem::exists(marker)) << "the spawned grandchild outlived stop()";
  std::filesystem::remove(marker);
}

TEST(SupervisorTest, DoesNotAdoptSomethingElseListeningOnAPendingReplicasPort)
{
  // Found on two cards. A previous run leaked a replica, it was still serving
  // on that port, and the supervisor promoted the new one without starting it
  // -- so the deployment reported as healthy a process it had never started,
  // could not stop, and did not account for. Only a reserved spare is already
  // up; anything else gets launched.
  StateMap map = atlas();
  map.addReplica(wanted("r0", Phase::Pending));

  FakeStarter starter;
  FakeProbe   probe;
  probe.healthy.insert("r0"); // something is listening on that port
  Supervisor supervisor(map, starter, probe);

  const auto report = supervisor.service();
  EXPECT_EQ(starter.starts.size(), 1U) << "a pending replica must be launched, not adopted";
  EXPECT_TRUE(report.promoted.empty());
  EXPECT_EQ(map.replica("r0")->phase, Phase::Pending) << "it cannot be serving before it has started";
}

TEST(SupervisorTest, AReservedSpareIsStillActivatedWithoutALaunch)
{
  // The case the shortcut exists for, unchanged: the spare is already loaded,
  // so activating it is a phase change rather than a model load.
  StateMap map   = atlas();
  Replica  spare = wanted("r0", Phase::Reserved);
  map.addReplica(spare);

  FakeStarter starter;
  FakeProbe   probe;
  probe.healthy.insert("r0");
  Supervisor supervisor(map, starter, probe);

  const auto report = supervisor.service();
  EXPECT_TRUE(starter.starts.empty()) << "activating a spare must not cost a model load";
  EXPECT_EQ(map.replica("r0")->phase, Phase::Active);
  EXPECT_EQ(report.promoted.size(), 1U);
}
