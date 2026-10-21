#include <algorithm>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <serving/infra/state.hpp>

using serving::infra::Action;
using serving::infra::Channel;
using serving::infra::Node;
using serving::infra::PendingAction;
using serving::infra::Phase;
using serving::infra::Replica;
using serving::infra::StateMap;

namespace
{

/// One node with eight cards, the shape of hg-atlas-01.
Node atlas(std::string name = "hg-atlas-01") { return Node{std::move(name), "192.168.150.11", {0, 1, 2, 3, 4, 5, 6, 7}}; }

Replica qwen(const std::string &name, int partition, std::vector<int> devices, int port)
{
  Replica r;
  r.name      = name;
  r.partition = partition;
  r.node      = "hg-atlas-01";
  r.devices   = std::move(devices);
  r.port      = port;
  r.phase     = Phase::Active;
  return r;
}

/// The problems `validate` reported, joined, so a test can ask whether one is in there.
std::string joined(const std::vector<std::string> &problems)
{
  std::string out;
  for (const auto &p : problems) { out += p + "\n"; }
  return out;
}

bool mentions(const std::vector<std::string> &problems, const std::string &needle) { return joined(problems).find(needle) != std::string::npos; }

} // namespace

TEST(StateMapTest, EveryMutationMovesTheVersion)
{
  StateMap map;
  EXPECT_EQ(map.version(), 0U);

  map.addNode(atlas());
  const uint64_t afterNode = map.version();
  EXPECT_GT(afterNode, 0U);

  map.addReplica(qwen("r0", 0, {4}, 8000));
  EXPECT_GT(map.version(), afterNode);

  // A reader that cached a routing decision can tell the ground moved without
  // diffing the document, which is the point of versioning it.
  const uint64_t before = map.version();
  EXPECT_TRUE(map.setPhase("r0", Phase::Draining));
  EXPECT_GT(map.version(), before);
}

TEST(StateMapTest, AMissingReplicaIsNotAnEditAndDoesNotMoveTheVersion)
{
  StateMap map;
  map.addNode(atlas());
  const uint64_t before = map.version();

  EXPECT_FALSE(map.setPhase("nobody", Phase::Dead));
  EXPECT_FALSE(map.setDesiredPhase("nobody", Phase::Dead));
  EXPECT_FALSE(map.place("nobody", "hg-atlas-01", {0}));
  EXPECT_FALSE(map.removeReplica("nobody"));
  EXPECT_EQ(map.version(), before);
}

TEST(StateMapTest, AddingAReplicaTwiceReplacesIt)
{
  StateMap map;
  map.addNode(atlas());
  map.addReplica(qwen("r0", 0, {4}, 8000));
  map.addReplica(qwen("r0", 0, {5}, 8000));

  ASSERT_EQ(map.replicas().size(), 1U);
  EXPECT_EQ(map.replica("r0")->devices, std::vector<int>{5});
}

TEST(StateMapTest, TracksWhichCardsAreTakenOnANode)
{
  StateMap map;
  map.addNode(atlas());
  map.addReplica(qwen("r0", 0, {0, 1}, 8000));
  map.addReplica(qwen("r1", 0, {4}, 8001));

  EXPECT_EQ(map.devicesInUse("hg-atlas-01"), (std::vector<int>{0, 1, 4}));
  EXPECT_EQ(map.devicesFree("hg-atlas-01"), (std::vector<int>{2, 3, 5, 6, 7}));

  // A drained replica is still serving what it has, so it still holds its card.
  map.setPhase("r1", Phase::Draining);
  EXPECT_EQ(map.devicesInUse("hg-atlas-01"), (std::vector<int>{0, 1, 4}));

  // A dead one has released it, and the capacity is free to be rebuilt on.
  map.setPhase("r1", Phase::Dead);
  EXPECT_EQ(map.devicesInUse("hg-atlas-01"), (std::vector<int>{0, 1}));
  EXPECT_NE(std::find(map.devicesFree("hg-atlas-01").begin(), map.devicesFree("hg-atlas-01").end(), 4), map.devicesFree("hg-atlas-01").end());
}

TEST(StateMapTest, DevicesOfAnUnknownNodeAreEmptyRatherThanEverything)
{
  StateMap map;
  EXPECT_TRUE(map.devicesFree("not-a-node").empty());
  EXPECT_TRUE(map.devicesInUse("not-a-node").empty());
}

TEST(StateMapTest, RemovingAReplicaTakesItsChannelsWithIt)
{
  StateMap map;
  map.addNode(atlas());
  map.addReplica(qwen("p0a", 0, {0}, 8000));
  map.addReplica(qwen("p1a", 1, {1}, 8001));
  map.addChannel({"p0a", "p1a"});
  ASSERT_EQ(map.channels().size(), 1U);

  EXPECT_TRUE(map.removeReplica("p1a"));
  // A channel to a replica that is gone would be a dangling edge.
  EXPECT_TRUE(map.channels().empty());
}

// ---------------------------------------------------------------------------
// validate: "does this partition fit that node?"
// ---------------------------------------------------------------------------

TEST(ValidateTest, ACoherentDeploymentHasNothingToSay)
{
  StateMap map;
  map.addNode(atlas());
  map.addReplica(qwen("p0a", 0, {0, 1}, 8000));
  map.addReplica(qwen("p1a", 1, {2, 3}, 8001));
  map.addChannel({"p0a", "p1a"});

  EXPECT_TRUE(validate(map).empty()) << joined(validate(map));
}

TEST(ValidateTest, CatchesAReplicaOnANodeThatIsNotThere)
{
  StateMap map;
  Replica  r = qwen("r0", 0, {0}, 8000);
  r.node     = "some-other-host";
  map.addReplica(r);

  EXPECT_TRUE(mentions(validate(map), "node 'some-other-host' is not in the map"));
}

TEST(ValidateTest, CatchesAReplicaClaimingACardItsNodeDoesNotHave)
{
  StateMap map;
  map.addNode(Node{"small", "10.0.0.1", {0, 1}});
  Replica r = qwen("r0", 0, {0, 7}, 8000);
  r.node    = "small";
  map.addReplica(r);

  // This is the deployment-validation question: the partition wants a card the
  // node does not own, and the map says so before anything is started.
  EXPECT_TRUE(mentions(validate(map), "device 7 is not on node 'small'"));
}

TEST(ValidateTest, CatchesTwoLiveReplicasOnTheSameCard)
{
  StateMap map;
  map.addNode(atlas());
  map.addReplica(qwen("r0", 0, {4}, 8000));
  map.addReplica(qwen("r1", 0, {4}, 8001));

  const auto problems = validate(map);
  EXPECT_TRUE(mentions(problems, "device 4: claimed by r0, r1")) << joined(problems);
}

TEST(ValidateTest, ADeadReplicaDoesNotClashOverTheCardItReleased)
{
  StateMap map;
  map.addNode(atlas());
  map.addReplica(qwen("r0", 0, {4}, 8000));
  Replica replacement = qwen("r1", 0, {4}, 8001);
  map.addReplica(replacement);
  ASSERT_FALSE(validate(map).empty());

  // Rebuilding lost capacity on the same card is exactly what M3 does, so the
  // replaced replica must stop claiming it.
  map.setPhase("r0", Phase::Dead);
  EXPECT_TRUE(validate(map).empty()) << joined(validate(map));
}

TEST(ValidateTest, CatchesDuplicateEndpointsAndNames)
{
  StateMap map;
  map.addNode(atlas());
  map.addReplica(qwen("r0", 0, {0}, 8000));
  Replica clash = qwen("r1", 0, {1}, 8000);
  map.addReplica(clash);

  EXPECT_TRUE(mentions(validate(map), "endpoint hg-atlas-01:8000 is already taken"));
}

TEST(ValidateTest, CatchesAPipelineWithAHole)
{
  StateMap map;
  map.addNode(atlas());
  map.addReplica(qwen("p0a", 0, {0}, 8000));
  map.addReplica(qwen("p2a", 2, {1}, 8001));

  // A request reaching partition 1 would have nowhere to go.
  EXPECT_TRUE(mentions(validate(map), "partitions must run from 0 without gaps"));
}

TEST(ValidateTest, CatchesAChannelToNowhere)
{
  StateMap map;
  map.addNode(atlas());
  map.addReplica(qwen("p0a", 0, {0}, 8000));
  map.addChannel({"p0a", "ghost"});

  EXPECT_TRUE(mentions(validate(map), "channel to 'ghost': no such replica"));
}

// ---------------------------------------------------------------------------
// reconcile: the declarative half
// ---------------------------------------------------------------------------

TEST(ReconcileTest, AMapInItsDesiredStateNeedsNothing)
{
  StateMap map;
  map.addNode(atlas());
  map.addReplica(qwen("r0", 0, {4}, 8000));

  EXPECT_TRUE(reconcile(map).empty());
}

TEST(ReconcileTest, AsksToStartWhatIsWantedAndNotYetRunning)
{
  StateMap map;
  map.addNode(atlas());
  Replica pending      = qwen("r0", 0, {4}, 8000);
  pending.phase        = Phase::Pending;
  pending.desiredPhase = Phase::Active;
  map.addReplica(pending);

  EXPECT_EQ(reconcile(map), (std::vector<PendingAction>{{"r0", Action::Start}}));
}

TEST(ReconcileTest, AsksToRebuildCapacityLostToADeadReplica)
{
  StateMap map;
  map.addNode(atlas());
  map.addReplica(qwen("r0", 0, {4}, 8000));
  map.setPhase("r0", Phase::Dead);

  // The deployment still wants it active, so the gap is a Start. Nothing here
  // executes it; that is the launcher's job and there is no launcher yet.
  EXPECT_EQ(reconcile(map), (std::vector<PendingAction>{{"r0", Action::Start}}));
}

TEST(ReconcileTest, DrainsAndStops)
{
  StateMap map;
  map.addNode(atlas());
  map.addReplica(qwen("r0", 0, {4}, 8000));
  map.addReplica(qwen("r1", 0, {5}, 8001));

  map.setDesiredPhase("r0", Phase::Draining);
  map.setDesiredPhase("r1", Phase::Dead);

  EXPECT_EQ(reconcile(map), (std::vector<PendingAction>{{"r0", Action::Drain}, {"r1", Action::Stop}}));
}

TEST(ReconcileTest, ADrainingReplicaIsNotAskedToDrainAgain)
{
  StateMap map;
  map.addNode(atlas());
  map.addReplica(qwen("r0", 0, {4}, 8000));
  map.setPhase("r0", Phase::Draining);
  map.setDesiredPhase("r0", Phase::Draining);

  EXPECT_TRUE(reconcile(map).empty());
}

TEST(PhaseTest, NamesRoundTrip)
{
  for (const Phase phase : {Phase::Pending, Phase::Active, Phase::Draining, Phase::Dead}) { EXPECT_EQ(serving::infra::parsePhase(serving::infra::phaseName(phase)), phase); }
  EXPECT_FALSE(serving::infra::parsePhase("zombie").has_value());
}

// ---------------------------------------------------------------------------
// discovery
// ---------------------------------------------------------------------------

#include <filesystem>
#include <fstream>

#include <serving/infra/discovery.hpp>

namespace
{

/// A directory shaped like /dev on a node with `count` cards.
class FakeDev
{
  public:

  explicit FakeDev(int count)
    : _dir(std::filesystem::temp_directory_path() / ("serving-dev-" + std::to_string(::getpid()) + "-" + std::to_string(++_seq)))
  {
    std::filesystem::create_directories(_dir);
    for (int i = 0; i < count; ++i) { touch("davinci" + std::to_string(i)); }
    // The driver puts these next to the cards; they are not cards.
    touch("davinci_manager");
    touch("devmm_svm");
    touch("hisi_hdc");
  }

  ~FakeDev()
  {
    std::error_code ec;
    std::filesystem::remove_all(_dir, ec);
  }

  void touch(const std::string &name) const { std::ofstream(_dir / name).put('\0'); }

  [[nodiscard]] std::string path() const { return _dir.string(); }

  private:

  std::filesystem::path _dir;
  static inline int     _seq = 0;
};

} // namespace

TEST(DiscoveryTest, FindsTheCardsAndIgnoresTheControlNodes)
{
  const FakeDev dev(8);
  EXPECT_EQ(serving::infra::discoverDevices(dev.path()), (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7}));
}

TEST(DiscoveryTest, AMachineWithNoAcceleratorsIsNotAnError)
{
  const FakeDev dev(0);
  EXPECT_TRUE(serving::infra::discoverDevices(dev.path()).empty());
  // A CPU-only build walks this path every time, so a missing /dev must not throw.
  EXPECT_TRUE(serving::infra::discoverDevices("/no/such/directory").empty());
}

TEST(DiscoveryTest, BuildsANodeTheMapAccepts)
{
  const FakeDev dev(8);
  StateMap      map;
  map.addNode(serving::infra::localNode("hg-atlas-01", "192.168.150.11", dev.path()));

  ASSERT_NE(map.node("hg-atlas-01"), nullptr);
  EXPECT_EQ(map.node("hg-atlas-01")->devices.size(), 8U);
  EXPECT_TRUE(map.node("hg-atlas-01")->owns(4));
  EXPECT_FALSE(map.node("hg-atlas-01")->owns(8));

  // And a replica placed on the card the broker granted validates against it.
  Replica r;
  r.name    = "qwen-0";
  r.node    = "hg-atlas-01";
  r.devices = {4};
  r.port    = 8000;
  r.phase   = Phase::Active;
  map.addReplica(r);
  EXPECT_TRUE(validate(map).empty()) << joined(validate(map));
  EXPECT_EQ(map.devicesFree("hg-atlas-01"), (std::vector<int>{0, 1, 2, 3, 5, 6, 7}));
}

TEST(StateMapTest, ARejectedNameIsOneThatCouldReachAShell)
{
  using serving::infra::isUsableName;

  EXPECT_TRUE(isUsableName("qwen-0"));
  EXPECT_TRUE(isUsableName("p0.auto_3"));

  // These are the ones that matter: the name is interpolated into a pid-file
  // path that a shell on another node evaluates, so a space splits the command
  // and a separator runs a second one.
  EXPECT_FALSE(isUsableName("qwen 0"));
  EXPECT_FALSE(isUsableName("qwen;rm -rf /"));
  EXPECT_FALSE(isUsableName("qwen$(id)"));
  EXPECT_FALSE(isUsableName("qwen`id`"));
  EXPECT_FALSE(isUsableName("../../etc/passwd"));
  EXPECT_FALSE(isUsableName(""));
  EXPECT_FALSE(isUsableName(std::string(65, 'a')));
}

TEST(StateMapTest, ValidateRefusesAReplicaNameAShellWouldSplit)
{
  StateMap map;
  map.addNode(atlas());

  Replica r;
  r.name    = "qwen 0; rm -rf /";
  r.node    = "hg-atlas-01";
  r.devices = {4};
  r.phase   = Phase::Active;
  map.addReplica(r);

  const auto problems = validate(map);
  ASSERT_FALSE(problems.empty());
  EXPECT_NE(joined(problems).find("letters, digits"), std::string::npos) << joined(problems);
}

TEST(StateMapTest, APendingReplicaStillHoldsTheCardItIsLoadingOnto)
{
  StateMap map;
  map.addNode(atlas());

  // Pending is the whole of a model load -- minutes. Treating the card as free
  // during it let a scale-up land a second model on top of the first.
  Replica loading;
  loading.name    = "qwen-0";
  loading.node    = "hg-atlas-01";
  loading.devices = {4};
  loading.phase   = Phase::Pending;
  map.addReplica(loading);

  EXPECT_EQ(map.devicesInUse("hg-atlas-01"), (std::vector<int>{4}));
  EXPECT_EQ(map.devicesFree("hg-atlas-01"), (std::vector<int>{0, 1, 2, 3, 5, 6, 7}));

  // And it is given back once the replica is gone for good.
  map.setPhase("qwen-0", Phase::Dead);
  EXPECT_TRUE(map.devicesInUse("hg-atlas-01").empty());
}
