#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <serving/coordinator/rule_message.hpp>

using serving::coordinator::decodeRulePublish;
using serving::coordinator::decodeRuleRevoke;
using serving::coordinator::encodeRulePublish;
using serving::coordinator::encodeRuleRevoke;
using serving::coordinator::RulePublish;
using serving::coordinator::RuleRevoke;

namespace
{

RulePublish samplePublish()
{
  RulePublish rule;
  rule.sessionId       = "chat-1";
  rule.nextHopName     = "p1b";
  rule.nextHopInstance = 21;
  rule.generation      = 7;
  return rule;
}

} // namespace

TEST(RuleMessageTest, PublishRoundTrips)
{
  const auto rule    = samplePublish();
  const auto encoded = encodeRulePublish(rule);
  const auto decoded = decodeRulePublish(encoded.data(), encoded.size());

  ASSERT_TRUE(decoded.has_value());
  EXPECT_EQ(decoded->sessionId, rule.sessionId);
  EXPECT_EQ(decoded->nextHopName, rule.nextHopName);
  EXPECT_EQ(decoded->nextHopInstance, rule.nextHopInstance);
  EXPECT_EQ(decoded->generation, rule.generation);
}

TEST(RuleMessageTest, RevokeRoundTrips)
{
  const RuleRevoke revoke{"chat-1", 9};
  const auto       encoded = encodeRuleRevoke(revoke);
  const auto       decoded = decodeRuleRevoke(encoded.data(), encoded.size());

  ASSERT_TRUE(decoded.has_value());
  EXPECT_EQ(decoded->sessionId, revoke.sessionId);
  EXPECT_EQ(decoded->generation, revoke.generation);
}

TEST(RuleMessageTest, HandlesEmptyAndUnicodeSessionIds)
{
  RulePublish rule = samplePublish();
  rule.sessionId   = "";
  auto encoded     = encodeRulePublish(rule);
  auto decoded     = decodeRulePublish(encoded.data(), encoded.size());
  ASSERT_TRUE(decoded.has_value());
  EXPECT_EQ(decoded->sessionId, "");

  rule.sessionId = "caf\xC3\xA9-\xE2\x98\x85";
  encoded        = encodeRulePublish(rule);
  decoded        = decodeRulePublish(encoded.data(), encoded.size());
  ASSERT_TRUE(decoded.has_value());
  EXPECT_EQ(decoded->sessionId, rule.sessionId);
}

TEST(RuleMessageTest, RejectsATruncatedFrame)
{
  // A coordinator and its replicas restart independently, so a reader has to
  // reject a short frame rather than read past the end of it.
  const auto encoded = encodeRulePublish(samplePublish());
  for (size_t cut = 0; cut < encoded.size(); ++cut) { EXPECT_FALSE(decodeRulePublish(encoded.data(), cut).has_value()) << "accepted a frame truncated to " << cut; }
  EXPECT_TRUE(decodeRulePublish(encoded.data(), encoded.size()).has_value());
}

TEST(RuleMessageTest, RejectsTrailingBytes)
{
  auto encoded = encodeRulePublish(samplePublish());
  encoded.push_back(0);
  // Extra bytes mean the frame is not what this version expects; guessing would
  // be worse than refusing.
  EXPECT_FALSE(decodeRulePublish(encoded.data(), encoded.size()).has_value());
}

TEST(RuleMessageTest, RejectsALengthThatWouldReadPastTheEnd)
{
  // The length prefix arrives over a channel, so it is not to be trusted. A
  // huge one must not wrap the bounds check and walk off the buffer.
  auto encoded = encodeRulePublish(samplePublish());
  for (int i = 0; i < 8; ++i) { encoded[8 + static_cast<size_t>(i)] = 0xFF; }
  EXPECT_FALSE(decodeRulePublish(encoded.data(), encoded.size()).has_value());
}

TEST(RuleMessageTest, RejectsNullAndEmpty)
{
  EXPECT_FALSE(decodeRulePublish(nullptr, 0).has_value());
  EXPECT_FALSE(decodeRuleRevoke(nullptr, 0).has_value());

  const std::vector<uint8_t> empty;
  EXPECT_FALSE(decodeRulePublish(empty.data(), 0).has_value());
  EXPECT_FALSE(decodeRuleRevoke(empty.data(), 0).has_value());
}

TEST(RuleMessageTest, TheEncodingIsEndianIndependent)
{
  // Written byte by byte, so the frame does not depend on the host's endianness
  // or on struct padding -- a coordinator and a replica may not be the same build.
  RulePublish rule;
  rule.sessionId       = "";
  rule.nextHopName     = "";
  rule.nextHopInstance = 0;
  rule.generation      = 0x0102030405060708ULL;

  const auto encoded = encodeRulePublish(rule);
  ASSERT_GE(encoded.size(), 8u);
  EXPECT_EQ(encoded[0], 0x08);
  EXPECT_EQ(encoded[7], 0x01);
}
