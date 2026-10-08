/**
 * Wire check for the rule path without a HiCR deployment: the message types
 * register, a rule round-trips, and the replica's own handlers put the
 * publisher's bytes into a replica's table -- both ends of Option B against
 * one frame.
 *
 * What is not checked here is a live channel. Standing an Input/Output pair up
 * takes the platform engine exchanging memory slots between two instances,
 * which is a deployment, not an example.
 */

#include <cstdio>
#include <map>
#include <string>

#include <serving/coordinator/channel_transport.hpp>
#include <serving/replica/channel_rules.hpp>
#include <serving/replica/rule_module.hpp>

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
  if (!decoded.has_value()) { return 1; }

  // The consumer half: the same frames, into a replica's table, through the
  // handlers a platform module subscribes with.
  replica::RuleTable    table;
  replica::ChannelDrain tally;
  const auto            onPublish = replica::rulePublishHandler(table, &tally);
  const auto            onRevoke  = replica::ruleRevokeHandler(table, &tally);

  system::channels::Message::metadata_t metadata;
  metadata.type = registry.getType(coordinator::RULE_PUBLISH_MESSAGE);
  onPublish(nullptr, system::channels::Message(encoded.data(), encoded.size(), metadata));

  const auto held = table.peek("chat-1");
  std::printf("replica applied it: %s -> %s on instance %lu, generation %lu\n",
              held.has_value() ? "yes" : "no",
              held.has_value() ? held->name.c_str() : "-",
              static_cast<unsigned long>(held.has_value() ? held->instance : 0),
              static_cast<unsigned long>(held.has_value() ? held->generation : 0));
  if (!held.has_value() || held->name != "p1b") { return 1; }

  // A message from an older generation must not undo a path already replanned.
  coordinator::RulePublish late = rule;
  late.generation               = 1;
  late.nextHopName              = "p1a";
  const auto lateFrame          = coordinator::encodeRulePublish(late);
  onPublish(nullptr, system::channels::Message(lateFrame.data(), lateFrame.size(), metadata));
  std::printf("generation 1 arriving after generation 3: stale=%d, still routed to %s\n", tally.stale, table.peek("chat-1")->name.c_str());
  if (tally.stale != 1 || table.peek("chat-1")->name != "p1b") { return 1; }

  // A frame that does not decode is counted, never applied.
  const uint8_t garbage[] = {0xFF, 0x00, 0x11};
  onPublish(nullptr, system::channels::Message(garbage, sizeof(garbage), metadata));

  coordinator::RuleRevoke revoke;
  revoke.sessionId       = "chat-1";
  revoke.generation      = 3;
  const auto revokeFrame = coordinator::encodeRuleRevoke(revoke);
  metadata.type          = registry.getType(coordinator::RULE_REVOKE_MESSAGE);
  onRevoke(nullptr, system::channels::Message(revokeFrame.data(), revokeFrame.size(), metadata));
  std::printf("after the revoke the replica holds: %s\n", table.peek("chat-1").has_value() ? "a rule" : "nothing");

  std::printf("tally: applied=%d revoked=%d stale=%d unknown=%d rejected=%d\n", tally.applied, tally.revoked, tally.stale, tally.unknown, tally.rejected);
  const bool ok = !table.peek("chat-1").has_value() && tally.applied == 1 && tally.revoked == 1 && tally.stale == 1 && tally.rejected == 1;
  std::printf("%s\n", ok ? "rule path ok on both ends" : "FAILED");
  return ok ? 0 : 1;
}
