#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <serving/infra/capacity.hpp>

using serving::infra::CapacityProvider;
using serving::infra::CapacityRequest;
using serving::infra::Grant;
using serving::infra::LocalCapacity;
using serving::infra::Node;
using serving::infra::Phase;
using serving::infra::Replica;
using serving::infra::StateMap;

namespace
{

/// One node with eight cards, the shape of hg-atlas-01.
StateMap atlas()
{
  StateMap map;
  map.addNode(Node{"hg-atlas-01", "192.168.150.11", {0, 1, 2, 3, 4, 5, 6, 7}});
  return map;
}

Replica running(const std::string &name, std::vector<int> devices)
{
  Replica r;
  r.name    = name;
  r.node    = "hg-atlas-01";
  r.devices = std::move(devices);
  r.phase   = Phase::Active;
  return r;
}

CapacityRequest one(int devices = 1)
{
  CapacityRequest request;
  request.devices = devices;
  return request;
}

} // namespace

// ---------------------------------------------------------------------------
// LocalCapacity
// ---------------------------------------------------------------------------

TEST(LocalCapacityTest, PlacesOnTheFreeCardsTheMapKnowsAbout)
{
  StateMap      map = atlas();
  LocalCapacity local(map);

  const auto grant = local.acquire(one());
  ASSERT_TRUE(grant.has_value());
  EXPECT_EQ(grant->node, "hg-atlas-01");
  EXPECT_EQ(grant->devices.size(), 1U);
}

TEST(LocalCapacityTest, WillNotHandOutACardAReplicaAlreadyHolds)
{
  StateMap map = atlas();
  map.addReplica(running("qwen-0", {4}));
  LocalCapacity local(map);

  // Eight cards, one taken.
  EXPECT_EQ(local.available(one()), 7);
  for (int i = 0; i < 7; ++i)
  {
    const auto grant = local.acquire(one());
    ASSERT_TRUE(grant.has_value()) << "at " << i;
    EXPECT_NE(grant->devices.front(), 4) << "handed out the card qwen-0 is on";
  }
}

TEST(LocalCapacityTest, TwoAcquiresInARowDoNotGetTheSameCard)
{
  StateMap      map = atlas();
  LocalCapacity local(map);

  // The map does not know about a grant until the caller places it, so without
  // tracking what is outstanding both of these would be handed card 0.
  const auto first  = local.acquire(one());
  const auto second = local.acquire(one());
  ASSERT_TRUE(first.has_value() && second.has_value());
  EXPECT_NE(first->devices, second->devices);
}

TEST(LocalCapacityTest, RunsOutRatherThanOverbooking)
{
  StateMap      map = atlas();
  LocalCapacity local(map);

  for (int i = 0; i < 8; ++i) { ASSERT_TRUE(local.acquire(one()).has_value()) << "at " << i; }
  // Asking for more than exists is a normal answer, not a failure: a policy
  // that asked too eagerly should back off.
  EXPECT_FALSE(local.acquire(one()).has_value());
  EXPECT_EQ(local.available(one()), 0);
}

TEST(LocalCapacityTest, ReleasingPutsTheCardBack)
{
  StateMap      map = atlas();
  LocalCapacity local(map);

  const auto grant = local.acquire(one());
  ASSERT_TRUE(grant.has_value());
  EXPECT_EQ(local.available(one()), 7);

  local.release(grant->replica);
  EXPECT_EQ(local.available(one()), 8);
  // Releasing something never granted is a no-op, so a caller recovering from
  // a half-finished scale-up need not track what stuck.
  local.release("never-existed");
  EXPECT_EQ(local.available(one()), 8);
}

TEST(LocalCapacityTest, AMultiCardReplicaNeedsThemAllOnOneNode)
{
  StateMap map;
  map.addNode(Node{"small-a", "10.0.0.1", {0, 1}});
  map.addNode(Node{"small-b", "10.0.0.2", {0, 1}});
  LocalCapacity local(map);

  // Four cards in total, but a four-card replica fits on neither node: the
  // deck's DSpark case, where a model needs a whole machine.
  EXPECT_FALSE(local.acquire(one(4)).has_value());
  EXPECT_EQ(local.available(one(4)), 0);
  // Two fits, twice.
  EXPECT_EQ(local.available(one(2)), 2);
}

TEST(LocalCapacityTest, ADeadReplicasCardsComeBack)
{
  StateMap map = atlas();
  map.addReplica(running("qwen-0", {0, 1, 2, 3, 4, 5, 6, 7}));
  LocalCapacity local(map);
  ASSERT_EQ(local.available(one()), 0);

  // Which is what lets lost capacity be rebuilt where it was.
  map.setPhase("qwen-0", Phase::Dead);
  EXPECT_EQ(local.available(one()), 8);
}
