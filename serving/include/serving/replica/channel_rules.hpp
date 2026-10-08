#pragma once

/**
 * Taking routing rules off a channel and into the replica's table.
 *
 * The consumer side of `coordinator/channel_transport.hpp`. The coordinator
 * publishes where a session goes next; this reads those messages and applies
 * them, so a replica learns its forwarding rules over the control plane's own
 * fabric rather than over HTTP.
 *
 * That matters for an MPI deployment: the replicas already have channels to
 * each other, and an HTTP hop for control traffic would be a second transport
 * to configure, secure and keep alive for no reason.
 *
 * Header-only and compiled only where the platform channel headers are, the
 * way `coordinator/channel_transport.hpp` is: `libserving` links no HiCR, and
 * only a binary that already has it pays for this.
 */

#include <cstdint>
#include <string>

#include <modules/subscription.hpp>
#include <system/channels/input.hpp>
#include <system/channels/messageTypeRegistry.hpp>

#include <serving/coordinator/rule_message.hpp>
#include <serving/replica/rule_table.hpp>

namespace serving::replica
{

/// What one drain of the channel did, so a replica can log something better
/// than silence and a test can see the tick worked.
struct ChannelDrain
{
  int applied = 0;
  int revoked = 0;
  /// Older than what was held: a late message, ignored. Expected, not an error.
  int stale = 0;
  /// Revokes for sessions this replica never held.
  int unknown = 0;
  /// Frames that did not decode, or carried a type this replica does not know.
  int rejected = 0;

  [[nodiscard]] int  total() const { return applied + revoked + stale + unknown + rejected; }
  [[nodiscard]] bool empty() const { return total() == 0; }
};

/**
 * Drains rule messages from one channel into one table.
 *
 * Does not own either. A replica has one table and one inbound channel from
 * its coordinator, and both outlive a drain.
 */
class ChannelRuleConsumer
{
  public:

  /// Registers the two message types so the ids match the publisher's. Both
  /// ends must register against the same registry contents, or a publish
  /// arrives with a type this replica cannot name.
  ChannelRuleConsumer(system::channels::MessageTypeRegistry &registry, system::channels::Input &channel, RuleTable &table)
    : _publishType(registry.registerType(coordinator::RULE_PUBLISH_MESSAGE)),
      _revokeType(registry.registerType(coordinator::RULE_REVOKE_MESSAGE)),
      _channel(channel),
      _table(table)
  {}

  /// Apply everything waiting. Returns without blocking when nothing is.
  ChannelDrain service()
  {
    ChannelDrain drain;

    while (_channel.hasMessage())
    {
      const auto message = _channel.getMessage();
      _received += 1;

      const auto type = message.getMetadata().type;
      if (type == _publishType)
      {
        const auto rule = coordinator::decodeRulePublish(message.getData(), message.getSize());
        if (!rule.has_value()) { drain.rejected += 1; }
        else
        {
          // No endpoint: a channel deployment addresses the next hop by
          // instance, which is what the rule already carries.
          switch (_table.apply(*rule))
          {
          case RuleOutcome::Applied: drain.applied += 1; break;
          case RuleOutcome::Stale: drain.stale += 1; break;
          default: drain.rejected += 1; break;
          }
        }
      }
      else if (type == _revokeType)
      {
        const auto revoke = coordinator::decodeRuleRevoke(message.getData(), message.getSize());
        if (!revoke.has_value()) { drain.rejected += 1; }
        else
        {
          switch (_table.revoke(*revoke))
          {
          case RuleOutcome::Revoked: drain.revoked += 1; break;
          case RuleOutcome::Stale: drain.stale += 1; break;
          case RuleOutcome::Unknown: drain.unknown += 1; break;
          default: drain.rejected += 1; break;
          }
        }
      }
      else
      {
        // Someone else's traffic on this channel, or a registry the two ends do
        // not agree on. Counted rather than dropped silently: a replica quietly
        // ignoring its routing rules is the hardest kind of fault to find.
        drain.rejected += 1;
      }

      // Popped whatever happened: a frame this replica cannot use must not block
      // the ones behind it.
      _channel.popMessage();
    }

    return drain;
  }

  /// Messages seen since construction, whatever became of them.
  [[nodiscard]] int received() const { return _received; }

  private:

  system::channels::Message::messageType_t _publishType;
  system::channels::Message::messageType_t _revokeType;

  system::channels::Input &_channel;
  RuleTable               &_table;
  int                      _received = 0;
};

/**
 * The same thing as a handler for `modules/subscription.hpp`.
 *
 * A `Subscription` binds one message type to one channel and one handler, so a
 * replica running as a platform module consumes its rules this way rather than
 * draining a channel itself. A subscription already knows which type it is
 * bound to, hence one handler per message rather than one that dispatches.
 *
 * `tally` is optional and, when given, is added to -- a module that wants to
 * report what it applied has somewhere to keep it.
 */
[[nodiscard]] inline modules::messageHandler_t rulePublishHandler(RuleTable &table, ChannelDrain *tally = nullptr)
{
  return [&table, tally](const std::shared_ptr<system::channels::Input>, const system::channels::Message &message) {
    const auto rule = coordinator::decodeRulePublish(message.getData(), message.getSize());
    if (!rule.has_value())
    {
      if (tally != nullptr) { tally->rejected += 1; }
      return;
    }
    // No endpoint: a channel deployment addresses the next hop by instance,
    // which the rule already carries.
    const auto outcome = table.apply(*rule);
    if (tally == nullptr) { return; }
    if (outcome == RuleOutcome::Applied) { tally->applied += 1; }
    else if (outcome == RuleOutcome::Stale) { tally->stale += 1; }
    else { tally->rejected += 1; }
  };
}

[[nodiscard]] inline modules::messageHandler_t ruleRevokeHandler(RuleTable &table, ChannelDrain *tally = nullptr)
{
  return [&table, tally](const std::shared_ptr<system::channels::Input>, const system::channels::Message &message) {
    const auto revoke = coordinator::decodeRuleRevoke(message.getData(), message.getSize());
    if (!revoke.has_value())
    {
      if (tally != nullptr) { tally->rejected += 1; }
      return;
    }
    const auto outcome = table.revoke(*revoke);
    if (tally == nullptr) { return; }
    if (outcome == RuleOutcome::Revoked) { tally->revoked += 1; }
    else if (outcome == RuleOutcome::Stale) { tally->stale += 1; }
    else if (outcome == RuleOutcome::Unknown) { tally->unknown += 1; }
    else { tally->rejected += 1; }
  };
}

} // namespace serving::replica
