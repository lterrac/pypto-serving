#pragma once

/**
 * Wire format of the coordinator's messages to replicas: a routing rule
 * (session, next hop, generation) and its revocation.
 *
 * Little-endian, length-prefixed, HiCR-free. Decoders return nullopt on a
 * malformed or truncated frame. channel_transport.hpp puts them on a channel.
 */

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace serving::coordinator
{

/// Names registered with the platform's MessageTypeRegistry.
inline constexpr const char *RULE_PUBLISH_MESSAGE = "serving.routing.rule.publish";
inline constexpr const char *RULE_REVOKE_MESSAGE  = "serving.routing.rule.revoke";

/// "This session continues at that replica."
struct RulePublish
{
  std::string sessionId;
  std::string nextHopName;
  /// The HiCR instance behind `nextHopName`, so the receiving replica does not
  /// have to resolve the name itself.
  uint64_t nextHopInstance = 0;
  /// Bumped on every replan, so a replica holding an older rule can tell.
  uint64_t generation = 0;
};

/// "Forget this session's rule; it is being replanned."
struct RuleRevoke
{
  std::string sessionId;
  uint64_t    generation = 0;
};

[[nodiscard]] std::vector<uint8_t>       encodeRulePublish(const RulePublish &rule);
[[nodiscard]] std::optional<RulePublish> decodeRulePublish(const uint8_t *data, size_t size);

[[nodiscard]] std::vector<uint8_t>      encodeRuleRevoke(const RuleRevoke &revoke);
[[nodiscard]] std::optional<RuleRevoke> decodeRuleRevoke(const uint8_t *data, size_t size);

} // namespace serving::coordinator
