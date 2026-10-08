#include <atomic>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include <serving/coordinator/rule_message.hpp>
#include <serving/replica/rule_table.hpp>
#include <serving/router/strategy.hpp>

using serving::coordinator::RulePublish;
using serving::coordinator::RuleRevoke;
using serving::replica::RuleOutcome;
using serving::replica::RuleTable;
using serving::router::DistributedRuleStrategy;

namespace
{

RulePublish rule(const std::string &session, const std::string &nextHop, uint64_t generation, uint64_t instance = 0) { return RulePublish{session, nextHop, instance, generation}; }

} // namespace

TEST(RuleTableTest, HoldsWhereASessionGoesNext)
{
  RuleTable table;
  EXPECT_EQ(table.apply(rule("chat-1", "p1a", 10, 77)), RuleOutcome::Applied);

  const auto hop = table.lookup("chat-1");
  ASSERT_TRUE(hop.has_value());
  EXPECT_EQ(hop->name, "p1a");
  // Carried so the replica does not have to resolve the name itself.
  EXPECT_EQ(hop->instance, 77U);
}

TEST(RuleTableTest, AnUnknownSessionIsNotAGuess)
{
  RuleTable table;
  // Nothing held is the replica's signal to ask the coordinator rather than
  // pick somewhere plausible.
  EXPECT_FALSE(table.lookup("never-seen").has_value());
}

TEST(RuleTableTest, ANewerGenerationReplacesTheRule)
{
  RuleTable table;
  table.apply(rule("chat-1", "p1a", 10));
  EXPECT_EQ(table.apply(rule("chat-1", "p1b", 11)), RuleOutcome::Applied);
  EXPECT_EQ(table.peek("chat-1")->name, "p1b");
}

TEST(RuleTableTest, ALateMessageCannotResurrectAReplannedPath)
{
  RuleTable table;
  table.apply(rule("chat-1", "p1b", 11));

  // Channels are per replica and nothing orders them against each other, so an
  // older message can arrive after a newer one. Applying it would send the
  // session back to a replica the coordinator has already moved it off.
  EXPECT_EQ(table.apply(rule("chat-1", "p1a", 10)), RuleOutcome::Stale);
  EXPECT_EQ(table.peek("chat-1")->name, "p1b");
}

TEST(RuleTableTest, RepublishingTheSamePlanIsNotAConflict)
{
  RuleTable table;
  table.apply(rule("chat-1", "p1a", 10));
  // Equal generations apply: a coordinator repeating itself is not an error.
  EXPECT_EQ(table.apply(rule("chat-1", "p1a", 10)), RuleOutcome::Applied);
  EXPECT_EQ(table.peek("chat-1")->name, "p1a");
}

TEST(RuleTableTest, RevokingForgetsTheSession)
{
  RuleTable table;
  table.apply(rule("chat-1", "p1a", 10));

  EXPECT_EQ(table.revoke(RuleRevoke{"chat-1", 10}), RuleOutcome::Revoked);
  EXPECT_FALSE(table.lookup("chat-1").has_value());
  EXPECT_EQ(table.revoke(RuleRevoke{"chat-1", 10}), RuleOutcome::Unknown);
}

TEST(RuleTableTest, ARevokeRacingItsReplacementDoesNotUndoIt)
{
  RuleTable table;
  table.apply(rule("chat-1", "p1a", 10));
  // The coordinator replanned: the new rule arrives, then the revoke that
  // belonged to the old one. Dropping the session here would strand it.
  table.apply(rule("chat-1", "p1b", 11));

  EXPECT_EQ(table.revoke(RuleRevoke{"chat-1", 10}), RuleOutcome::Stale);
  ASSERT_TRUE(table.peek("chat-1").has_value());
  EXPECT_EQ(table.peek("chat-1")->name, "p1b");
}

TEST(RuleTableTest, ARestartedCoordinatorsRulesAreNotMistakenForOldOnes)
{
  RuleTable table;
  // The epoch is the high half, so a coordinator that restarts and begins
  // counting at 1 again still issues generations above the ones a surviving
  // replica holds. Without it this second apply would read as stale.
  table.apply(rule("chat-1", "p1a", DistributedRuleStrategy::makeGeneration(100, 5000)));
  EXPECT_EQ(table.apply(rule("chat-1", "p1b", DistributedRuleStrategy::makeGeneration(101, 1))), RuleOutcome::Applied);
  EXPECT_EQ(table.peek("chat-1")->name, "p1b");
}

TEST(RuleTableTest, BoundedSoALongTailOfConversationsCannotGrowItForEver)
{
  RuleTable table(3);
  for (int i = 0; i < 5; ++i) { table.apply(rule("chat-" + std::to_string(i), "p1a", 1)); }

  EXPECT_EQ(table.size(), 3U);
  EXPECT_EQ(table.evicted(), 2);
  // The oldest went; a session whose rule is gone is a cache miss the
  // coordinator can replan.
  EXPECT_FALSE(table.peek("chat-0").has_value());
  EXPECT_TRUE(table.peek("chat-4").has_value());
}

TEST(RuleTableTest, LookingUpASessionKeepsItFromBeingEvicted)
{
  RuleTable table(2);
  table.apply(rule("old", "p1a", 1));
  table.apply(rule("new", "p1a", 1));

  // Used, so it is no longer the least recently used.
  ASSERT_TRUE(table.lookup("old").has_value());
  table.apply(rule("newest", "p1a", 1));

  EXPECT_TRUE(table.peek("old").has_value());
  EXPECT_FALSE(table.peek("new").has_value());
}

TEST(RuleTableTest, RefusesACapacityOfZero) { EXPECT_THROW(RuleTable(0), std::invalid_argument); }

// ---------------------------------------------------------------------------
// the wire, end to end
// ---------------------------------------------------------------------------

TEST(RuleTableTest, AppliesWhatTheCoordinatorPutOnTheWire)
{
  const RulePublish sent  = rule("chat-1", "p1a", 42, 77);
  const auto        frame = serving::coordinator::encodeRulePublish(sent);

  // What the replica actually receives is bytes.
  const auto decoded = serving::coordinator::decodeRulePublish(frame.data(), frame.size());
  ASSERT_TRUE(decoded.has_value());

  RuleTable table;
  ASSERT_EQ(table.apply(*decoded), RuleOutcome::Applied);
  const auto hop = table.lookup("chat-1");
  ASSERT_TRUE(hop.has_value());
  EXPECT_EQ(hop->name, "p1a");
  EXPECT_EQ(hop->instance, 77U);
  EXPECT_EQ(hop->generation, 42U);
}

TEST(RuleTableTest, ATruncatedFrameIsRejectedRatherThanHalfApplied)
{
  const auto frame = serving::coordinator::encodeRulePublish(rule("chat-1", "p1a", 42));
  EXPECT_FALSE(serving::coordinator::decodeRulePublish(frame.data(), frame.size() / 2).has_value());
}

// ---------------------------------------------------------------------------
// two writers
// ---------------------------------------------------------------------------

TEST(RuleTableTest, SurvivesAChannelWritingWhileHandlersRead)
{
  // The deployment this guards: rules arrive on the rule module's service
  // thread while HTTP handler threads look sessions up to forward them. Both
  // mutate the table -- `lookup` reorders the LRU list -- so neither side can
  // be left unlocked. Run under tsan this is the test that catches it.
  RuleTable table(64);

  std::atomic<bool> stop{false};
  std::atomic<int>  found{0};

  std::thread reader([&] {
    while (!stop.load())
    {
      for (int i = 0; i < 128; ++i)
      {
        if (table.lookup("chat-" + std::to_string(i % 32)).has_value()) { found.fetch_add(1); }
      }
    }
  });

  for (int generation = 1; generation <= 200; ++generation)
  {
    for (int i = 0; i < 32; ++i)
    {
      const std::string session = "chat-" + std::to_string(i);
      table.apply(rule(session, "p1a", static_cast<uint64_t>(generation), 77));
      if (generation % 7 == 0) { table.revoke(RuleRevoke{session, static_cast<uint64_t>(generation)}); }
    }
  }

  stop.store(true);
  reader.join();

  // The point is that it finished without tearing; what the reader saw depends
  // on timing, but it must have seen something.
  EXPECT_GT(found.load(), 0);
  EXPECT_LE(table.size(), table.capacity());
}
