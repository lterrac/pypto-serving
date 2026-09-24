#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <serving/router/routing.hpp>

using serving::router::newSessionId;
using serving::router::NoReplicaAvailable;
using serving::router::ReplicaRegistry;
using serving::router::ReplicaSpec;
using serving::router::RoutingConfig;
using serving::router::SessionDirectory;

namespace
{

/// A clock the test drives, so expiry is exercised without sleeping -- the
/// Python takes an injectable clock for the same reason.
struct FakeClock
{
  double now = 0.0;

  [[nodiscard]] SessionDirectory::Clock fn()
  {
    return [this] { return now; };
  }
};

ReplicaSpec spec(const std::string &name, int port)
{
  ReplicaSpec s;
  s.name = name;
  s.host = "127.0.0.1";
  s.port = port;
  return s;
}

std::vector<ReplicaSpec> twoReplicas() { return {spec("a", 8000), spec("b", 8001)}; }

} // namespace

// ---------------------------------------------------------------------------
// ReplicaSpec
// ---------------------------------------------------------------------------

TEST(ReplicaSpecTest, BuildsItsBaseUrl)
{
  EXPECT_EQ(spec("a", 8000).baseUrl(), "http://127.0.0.1:8000");

  ReplicaSpec tls = spec("a", 443);
  tls.scheme      = "https";
  // Per-replica scheme: prompts cross this hop in the clear otherwise.
  EXPECT_EQ(tls.baseUrl(), "https://127.0.0.1:443");
}

TEST(SessionIdTest, IsThirtyTwoHexCharactersAndUnique)
{
  std::set<std::string> seen;
  for (int i = 0; i < 200; ++i)
  {
    const auto id = newSessionId();
    EXPECT_EQ(id.size(), 32u);
    EXPECT_EQ(id.find_first_not_of("0123456789abcdef"), std::string::npos) << id;
    seen.insert(id);
  }
  EXPECT_EQ(seen.size(), 200u);
}

// ---------------------------------------------------------------------------
// SessionDirectory
// ---------------------------------------------------------------------------

TEST(SessionDirectoryTest, RoundTripsAPin)
{
  FakeClock        clock;
  SessionDirectory sessions(60.0, clock.fn());
  sessions.pin("s1", "a");
  EXPECT_EQ(sessions.lookup("s1"), "a");
  EXPECT_EQ(sessions.size(), 1u);
  EXPECT_FALSE(sessions.lookup("missing").has_value());
}

TEST(SessionDirectoryTest, RejectsNonsenseConstruction)
{
  EXPECT_THROW(SessionDirectory(0.0), std::invalid_argument);
  EXPECT_THROW(SessionDirectory(-1.0), std::invalid_argument);
  EXPECT_THROW(SessionDirectory(60.0, {}, 0), std::invalid_argument);
}

TEST(SessionDirectoryTest, ExpiresAPinPastItsTtl)
{
  FakeClock        clock;
  SessionDirectory sessions(60.0, clock.fn());
  sessions.pin("s1", "a");

  clock.now = 59.0;
  EXPECT_EQ(sessions.lookup("s1"), "a");

  clock.now = 61.0;
  EXPECT_FALSE(sessions.lookup("s1").has_value());
  // Lookup drops what it touches, so the entry is gone rather than stale.
  EXPECT_EQ(sessions.size(), 0u);
}

TEST(SessionDirectoryTest, ForgetDropsOnePin)
{
  FakeClock        clock;
  SessionDirectory sessions(60.0, clock.fn());
  sessions.pin("s1", "a");
  sessions.pin("s2", "a");
  sessions.forget("s1");
  EXPECT_FALSE(sessions.lookup("s1").has_value());
  EXPECT_EQ(sessions.lookup("s2"), "a");
  EXPECT_NO_THROW(sessions.forget("never-existed"));
}

TEST(SessionDirectoryTest, ForgetReplicaDropsOnlyItsOwnPins)
{
  FakeClock        clock;
  SessionDirectory sessions(60.0, clock.fn());
  sessions.pin("s1", "a");
  sessions.pin("s2", "b");
  sessions.pin("s3", "a");

  EXPECT_EQ(sessions.forgetReplica("a"), 2);
  EXPECT_FALSE(sessions.lookup("s1").has_value());
  EXPECT_FALSE(sessions.lookup("s3").has_value());
  EXPECT_EQ(sessions.lookup("s2"), "b");
}

TEST(SessionDirectoryTest, SweepDropsOnlyExpiredPins)
{
  FakeClock        clock;
  SessionDirectory sessions(60.0, clock.fn());
  sessions.pin("old", "a");
  clock.now = 50.0;
  sessions.pin("new", "b");

  clock.now = 70.0; // old is 70s stale, new is 20s
  EXPECT_EQ(sessions.sweep(), 1);
  EXPECT_EQ(sessions.size(), 1u);
  EXPECT_EQ(sessions.lookup("new"), "b");
}

TEST(SessionDirectoryTest, EvictsTheLeastRecentlyUsedPinWhenFull)
{
  // Session ids are client-supplied, so this map is keyed by untrusted input.
  // Unbounded growth costs the process; a lost pin costs one prefill.
  FakeClock        clock;
  SessionDirectory sessions(600.0, clock.fn(), 3);
  sessions.pin("s1", "a");
  sessions.pin("s2", "a");
  sessions.pin("s3", "a");

  // Touch s1 so s2 becomes the least recent.
  sessions.pin("s1", "a");
  sessions.pin("s4", "a");

  EXPECT_EQ(sessions.size(), 3u);
  EXPECT_FALSE(sessions.lookup("s2").has_value()) << "the least-recently pinned entry should have gone";
  EXPECT_TRUE(sessions.lookup("s1").has_value());
  EXPECT_TRUE(sessions.lookup("s3").has_value());
  EXPECT_TRUE(sessions.lookup("s4").has_value());
}

TEST(SessionDirectoryTest, ReclaimsExpiredPinsBeforeEvictingLiveOnes)
{
  FakeClock        clock;
  SessionDirectory sessions(60.0, clock.fn(), 2);
  sessions.pin("stale", "a");
  clock.now = 100.0; // stale is now expired
  sessions.pin("fresh", "a");
  sessions.pin("newest", "a");

  // The expired pin is what goes, not the live one.
  EXPECT_TRUE(sessions.lookup("fresh").has_value());
  EXPECT_TRUE(sessions.lookup("newest").has_value());
  EXPECT_FALSE(sessions.lookup("stale").has_value());
}

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------

TEST(ReplicaRegistryTest, SendsTheSameSessionToTheSameReplica)
{
  // The whole point: the prefix cache is per replica, so turn N+1 re-prefills
  // only the new text when it lands where turn N did.
  FakeClock        clock;
  SessionDirectory sessions(600.0, clock.fn());
  ReplicaRegistry  registry(RoutingConfig{}, sessions, twoReplicas());

  const auto first = registry.select("session-1");
  for (int i = 0; i < 5; ++i)
  {
    const auto again = registry.select("session-1");
    EXPECT_EQ(again.replica.name, first.replica.name);
    EXPECT_TRUE(again.affinityHit);
  }
  EXPECT_FALSE(first.affinityHit) << "the first route cannot be an affinity hit";
  EXPECT_EQ(registry.affinityHits(), 5);
}

TEST(ReplicaRegistryTest, SpreadsNewSessionsAcrossReplicas)
{
  FakeClock        clock;
  SessionDirectory sessions(600.0, clock.fn());
  ReplicaRegistry  registry(RoutingConfig{}, sessions, twoReplicas());

  const auto first  = registry.select("s1");
  const auto second = registry.select("s2");
  // Equal load, so the rotating tiebreak must not hand both to the same replica.
  EXPECT_NE(first.replica.name, second.replica.name);
  EXPECT_EQ(registry.totalRouted(), 2);
}

TEST(ReplicaRegistryTest, AffinitySurvivesModerateLoadButYieldsPastTheSlack)
{
  FakeClock        clock;
  SessionDirectory sessions(600.0, clock.fn());
  RoutingConfig    config;
  config.affinitySlack = 2;
  ReplicaRegistry registry(config, sessions, twoReplicas());

  const auto        first      = registry.select("s1");
  const std::string pinnedName = first.replica.name;

  // Two outstanding: within the slack, so affinity holds.
  registry.acquire(pinnedName);
  registry.acquire(pinnedName);
  EXPECT_TRUE(registry.select("s1").affinityHit);

  // A third tips it past the slack and load wins.
  registry.acquire(pinnedName);
  const auto moved = registry.select("s1");
  EXPECT_FALSE(moved.affinityHit);
  EXPECT_NE(moved.replica.name, pinnedName);
}

TEST(ReplicaRegistryTest, ZeroSlackDisablesAffinityUnderAnyImbalance)
{
  FakeClock        clock;
  SessionDirectory sessions(600.0, clock.fn());
  RoutingConfig    config;
  config.affinitySlack = 0;
  ReplicaRegistry registry(config, sessions, twoReplicas());

  const auto first = registry.select("s1");
  registry.acquire(first.replica.name);
  const auto moved = registry.select("s1");
  EXPECT_FALSE(moved.affinityHit);
  EXPECT_NE(moved.replica.name, first.replica.name);
}

TEST(ReplicaRegistryTest, AnExpiredPinReRoutesByLoad)
{
  FakeClock        clock;
  SessionDirectory sessions(60.0, clock.fn());
  ReplicaRegistry  registry(RoutingConfig{}, sessions, twoReplicas());

  registry.select("s1");
  clock.now = 120.0;
  EXPECT_FALSE(registry.select("s1").affinityHit) << "an expired pin is not an affinity hit";
}

TEST(ReplicaRegistryTest, AnUnroutablePinnedReplicaFallsBack)
{
  FakeClock        clock;
  SessionDirectory sessions(600.0, clock.fn());
  ReplicaRegistry  registry(RoutingConfig{}, sessions, twoReplicas());

  const auto first = registry.select("s1");
  registry.setReady(first.replica.name, false);

  const auto moved = registry.select("s1");
  EXPECT_NE(moved.replica.name, first.replica.name);
  EXPECT_FALSE(moved.affinityHit);
}

TEST(ReplicaRegistryTest, RoutesToTheLeastLoadedReplica)
{
  FakeClock        clock;
  SessionDirectory sessions(600.0, clock.fn());
  ReplicaRegistry  registry(RoutingConfig{}, sessions, twoReplicas());

  for (int i = 0; i < 5; ++i) { registry.acquire("a"); }
  EXPECT_EQ(registry.select("fresh").replica.name, "b");
}

TEST(ReplicaRegistryTest, RejectsWhenNothingIsRoutable)
{
  FakeClock        clock;
  SessionDirectory sessions(600.0, clock.fn());
  ReplicaRegistry  registry(RoutingConfig{}, sessions, twoReplicas());

  registry.setReady("a", false);
  registry.setReady("b", false);
  EXPECT_THROW((void)registry.select("s1"), NoReplicaAvailable);
  EXPECT_EQ(registry.rejected(), 1);
  EXPECT_EQ(registry.readyCount(), 0);
}

// ---------------------------------------------------------------------------
// Membership
// ---------------------------------------------------------------------------

TEST(ReplicaRegistryTest, AddsAReplicaUnreadyByDefault)
{
  FakeClock        clock;
  SessionDirectory sessions(600.0, clock.fn());
  ReplicaRegistry  registry(RoutingConfig{}, sessions, {spec("a", 8000)});

  const auto &added = registry.add(spec("b", 8001));
  // A launched replica cannot serve for the minutes its model takes to load;
  // the health poller is what promotes it.
  EXPECT_FALSE(added.ready);
  EXPECT_EQ(registry.readyCount(), 1);
  EXPECT_EQ(registry.select("s1").replica.name, "a");

  registry.setReady("b", true);
  EXPECT_EQ(registry.readyCount(), 2);
}

TEST(ReplicaRegistryTest, RefusesADuplicateName)
{
  FakeClock        clock;
  SessionDirectory sessions(600.0, clock.fn());
  ReplicaRegistry  registry(RoutingConfig{}, sessions, twoReplicas());
  EXPECT_THROW((void)registry.add(spec("a", 9999)), std::invalid_argument);
}

TEST(ReplicaRegistryTest, RemovingReindexesTheSurvivors)
{
  // Leaving gaps makes (index - counter) % count alias two replicas onto one
  // tiebreak slot: unfair rather than wrong, but silently so.
  FakeClock        clock;
  SessionDirectory sessions(600.0, clock.fn());
  ReplicaRegistry  registry(RoutingConfig{}, sessions, {spec("a", 1), spec("b", 2), spec("c", 3)});

  EXPECT_TRUE(registry.remove("b"));
  ASSERT_EQ(registry.states().size(), 2u);
  EXPECT_EQ(registry.states()[0].name(), "a");
  EXPECT_EQ(registry.states()[0].index, 0);
  EXPECT_EQ(registry.states()[1].name(), "c");
  EXPECT_EQ(registry.states()[1].index, 1) << "indices must be contiguous after a removal";

  EXPECT_FALSE(registry.remove("b"));
}

TEST(ReplicaRegistryTest, RemovingDropsItsPins)
{
  FakeClock        clock;
  SessionDirectory sessions(600.0, clock.fn());
  ReplicaRegistry  registry(RoutingConfig{}, sessions, twoReplicas());

  const auto first = registry.select("s1");
  EXPECT_TRUE(sessions.lookup("s1").has_value());
  registry.remove(first.replica.name);
  EXPECT_FALSE(sessions.lookup("s1").has_value());
}

TEST(ReplicaRegistryTest, DrainingReleasesPinsAndStopsNewRoutes)
{
  FakeClock        clock;
  SessionDirectory sessions(600.0, clock.fn());
  ReplicaRegistry  registry(RoutingConfig{}, sessions, twoReplicas());

  const auto first = registry.select("s1");
  registry.setDraining(first.replica.name, true);

  // A session whose replica is going away re-routes on its next turn rather
  // than waiting out the TTL.
  EXPECT_FALSE(sessions.lookup("s1").has_value());
  const auto moved = registry.select("s1");
  EXPECT_NE(moved.replica.name, first.replica.name);
  EXPECT_EQ(registry.readyCount(), 1);

  registry.setDraining(first.replica.name, false);
  EXPECT_EQ(registry.readyCount(), 2);
}

TEST(ReplicaRegistryTest, TracksOutstandingWithoutGoingNegative)
{
  FakeClock        clock;
  SessionDirectory sessions(600.0, clock.fn());
  ReplicaRegistry  registry(RoutingConfig{}, sessions, twoReplicas());

  registry.acquire("a");
  registry.acquire("a");
  registry.release("a");
  EXPECT_EQ(registry.state("a")->outstanding, 1);

  registry.release("a");
  registry.release("a"); // one too many
  EXPECT_EQ(registry.state("a")->outstanding, 0);

  EXPECT_NO_THROW(registry.acquire("nonexistent"));
  EXPECT_NO_THROW(registry.release("nonexistent"));
}

TEST(ReplicaRegistryTest, TheRotatingTiebreakStaysInRangeAfterRemoval)
{
  // Python's % is non-negative, C++'s is not; the rotation must not go negative.
  FakeClock        clock;
  SessionDirectory sessions(600.0, clock.fn());
  ReplicaRegistry  registry(RoutingConfig{}, sessions, {spec("a", 1), spec("b", 2), spec("c", 3)});

  for (int i = 0; i < 6; ++i) { registry.select("s" + std::to_string(i)); }
  registry.remove("a");

  std::set<std::string> reached;
  for (int i = 0; i < 12; ++i) { reached.insert(registry.select("t" + std::to_string(i)).replica.name); }
  EXPECT_EQ(reached, (std::set<std::string>{"b", "c"})) << "the rotation must still reach every survivor";
}
