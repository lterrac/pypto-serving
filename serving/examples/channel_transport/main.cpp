/**
 * Wire check for ChannelRuleTransport without a HiCR deployment: the message
 * types register and a rule round-trips.
 */

#include <cstdio>
#include <map>
#include <string>

#include <serving/coordinator/channel_transport.hpp>

int main()
{
  using namespace serving;

  system::channels::MessageTypeRegistry registry;

  // Publishing to an unwired replica is a no-op.
  std::map<std::string, system::channels::Output *> channels;
  std::map<std::string, uint64_t>                   instances{{"p1a", 20}, {"p1b", 21}};

  coordinator::ChannelRuleTransport transport(registry, channels, instances);

  router::ReplicaSpec holder;
  holder.name = "p0a";
  router::ReplicaSpec nextHop;
  nextHop.name = "p1b";

  transport.publish(holder, "chat-1", nextHop, 3);
  transport.revoke("chat-1", 3);
  std::printf("transport: publishing to an unwired replica is a no-op, as intended\n");

  std::printf("message types registered: %s=%u  %s=%u\n",
              coordinator::RULE_PUBLISH_MESSAGE,
              registry.getType(coordinator::RULE_PUBLISH_MESSAGE),
              coordinator::RULE_REVOKE_MESSAGE,
              registry.getType(coordinator::RULE_REVOKE_MESSAGE));

  // The frame a replica receives and decodes on the far side.
  coordinator::RulePublish rule;
  rule.sessionId       = "chat-1";
  rule.nextHopName     = "p1b";
  rule.nextHopInstance = 21;
  rule.generation      = 3;

  const auto encoded = coordinator::encodeRulePublish(rule);
  const auto decoded = coordinator::decodeRulePublish(encoded.data(), encoded.size());
  std::printf("rule round trip: %s (%zu bytes) -> session %s continues at %s on instance %lu, generation %lu\n",
              decoded.has_value() ? "ok" : "FAILED",
              encoded.size(),
              decoded->sessionId.c_str(),
              decoded->nextHopName.c_str(),
              static_cast<unsigned long>(decoded->nextHopInstance),
              static_cast<unsigned long>(decoded->generation));
  return decoded.has_value() ? 0 : 1;
}
