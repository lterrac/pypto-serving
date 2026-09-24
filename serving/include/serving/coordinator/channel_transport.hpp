#pragma once

/**
 * RuleTransport over platform channels.
 *
 * Compiled only when the platform's channel headers are complete; they need
 * modules/configuration/edge.hpp (see serving/meson.build).
 */

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>

#include <system/channels/message.hpp>
#include <system/channels/messageTypeRegistry.hpp>
#include <system/channels/output.hpp>

#include <serving/coordinator/rule_message.hpp>
#include <serving/router/strategy.hpp>

namespace serving::coordinator
{

/**
 * Publishes routing rules onto one outbound channel per replica.
 *
 * The coordinator owns the name-to-instance map and the channels; this only
 * turns a routing decision into a control message and pushes it.
 */
class ChannelRuleTransport final : public router::RuleTransport
{
  public:

  /// `channels` is keyed by the routing name of the replica each one reaches.
  ChannelRuleTransport(system::channels::MessageTypeRegistry &registry, std::map<std::string, system::channels::Output *> channels, std::map<std::string, uint64_t> instanceByName)
    : _publishType(registry.registerType(RULE_PUBLISH_MESSAGE)),
      _revokeType(registry.registerType(RULE_REVOKE_MESSAGE)),
      _channels(std::move(channels)),
      _instanceByName(std::move(instanceByName))
  {}

  void publish(const router::ReplicaSpec &holder, const std::string &sessionId, const router::ReplicaSpec &nextHop, uint64_t generation) override
  {
    system::channels::Output *channel = channelFor(holder.name);
    if (channel == nullptr) { return; }

    RulePublish rule;
    rule.sessionId       = sessionId;
    rule.nextHopName     = nextHop.name;
    rule.nextHopInstance = instanceFor(nextHop.name);
    rule.generation      = generation;

    const auto payload = encodeRulePublish(rule);
    send(*channel, _publishType, generation, payload);
  }

  void revoke(const std::string &sessionId, uint64_t generation) override
  {
    RuleRevoke revoke;
    revoke.sessionId   = sessionId;
    revoke.generation  = generation;
    const auto payload = encodeRuleRevoke(revoke);

    // A revoke has no single holder: every replica that might be carrying the
    // rule has to hear about it, or one of them keeps forwarding into a replica
    // that is going away.
    for (auto &[name, channel] : _channels)
    {
      if (channel != nullptr && _unreachable.count(name) == 0) { send(*channel, _revokeType, generation, payload); }
    }
  }

  /// Pushing to a channel whose consumer is gone blocks once it fills, which
  /// would hang the service tick that is applying the loss.
  void onReplicaUnreachable(const std::string &replicaName) override { _unreachable.insert(replicaName); }

  private:

  [[nodiscard]] system::channels::Output *channelFor(const std::string &replicaName) const
  {
    const auto it = _channels.find(replicaName);
    return it == _channels.end() ? nullptr : it->second;
  }

  [[nodiscard]] uint64_t instanceFor(const std::string &replicaName) const
  {
    const auto it = _instanceByName.find(replicaName);
    return it == _instanceByName.end() ? 0 : it->second;
  }

  void send(system::channels::Output &channel, system::channels::Message::messageType_t type, uint64_t generation, const std::vector<uint8_t> &payload)
  {
    system::channels::Message::metadata_t metadata;
    metadata.type    = type;
    metadata.groupId = 0;
    // The channel's own ordering, distinct from a path's generation: a sequence
    // is per channel, a generation is per session.
    metadata.sequenceId = ++_sequence;
    (void)generation;

    // Locking push: a coordinator may publish from its request path and from its
    // service tick, and those are different threads.
    channel.pushMessageLocking(system::channels::Message(payload.data(), payload.size(), metadata));
  }

  system::channels::Message::messageType_t _publishType;
  system::channels::Message::messageType_t _revokeType;

  std::map<std::string, system::channels::Output *> _channels;

  std::set<std::string>           _unreachable;
  std::map<std::string, uint64_t> _instanceByName;
  uint64_t                        _sequence = 0;
};

} // namespace serving::coordinator
