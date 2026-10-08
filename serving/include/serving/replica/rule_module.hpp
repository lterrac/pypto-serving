#pragma once

/**
 * The replica side of Option B as a platform module.
 *
 * `CoordinatorModule` runs the coordinator that plans a session's path; this
 * runs the replica that receives it. Every service tick drains the inbound
 * rule channel into the replica's table, which the HTTP server then reads to
 * forward a request to the next partition without asking anyone.
 *
 * Nothing here is on the per-token path: a rule arrives once per session, and
 * the forwarding decision afterwards is a table lookup.
 *
 * Compiled only with the platform feature, like everything else that includes
 * the channel headers.
 */

#include <cstddef>

#include <modules/module.hpp>

#include <serving/replica/channel_rules.hpp>
#include <serving/replica/rule_table.hpp>

namespace serving::replica
{

/// Default service period. Rules arrive once per session and a few tens of
/// milliseconds of delay on the first message of a conversation is invisible,
/// so this is set to not busy-wait rather than to be fast.
inline constexpr size_t DEFAULT_RULE_SERVICE_INTERVAL_MS = 50;

class RuleModule final : public modules::Module
{
  public:

  /// Neither the channel nor the table is owned: the replica has one of each
  /// and both outlive this module.
  RuleModule(system::channels::MessageTypeRegistry &registry,
             system::channels::Input               &channel,
             RuleTable                             &table,
             const size_t                           intervalMs = DEFAULT_RULE_SERVICE_INTERVAL_MS)
    : modules::Module(intervalMs),
      _consumer(registry, channel, table)
  {}

  void initialize() override {}

  /// Nothing to start: the module is driven entirely by its service tick.
  void run() override {}

  void terminate() override {}

  void await() override {}

  /// One last drain, so a rule that landed between the final tick and shutdown
  /// is not reported as lost.
  void finalize() override { accumulate(_consumer.service()); }

  /// Everything applied since the module started.
  [[nodiscard]] const ChannelDrain &applied() const { return _total; }

  /// Messages seen, whatever became of them.
  [[nodiscard]] int received() const { return _consumer.received(); }

  protected:

  void service() override { accumulate(_consumer.service()); }

  private:

  void accumulate(const ChannelDrain &drain)
  {
    _total.applied += drain.applied;
    _total.revoked += drain.revoked;
    _total.stale += drain.stale;
    _total.unknown += drain.unknown;
    _total.rejected += drain.rejected;
  }

  ChannelRuleConsumer _consumer;
  ChannelDrain        _total;
};

} // namespace serving::replica
