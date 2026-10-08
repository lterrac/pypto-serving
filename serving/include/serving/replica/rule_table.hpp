#pragma once

/**
 * The replica's half of Option B (`scaling-pypto-serving.pptx`, slides 9-10).
 *
 * The coordinator plans a session's path once and publishes a forwarding rule
 * to each replica on it; the replicas then forward directly, which is what
 * keeps the coordinator off the per-token path. This is the table a replica
 * keeps so it knows where the next partition is -- the consumer the publisher
 * in `coordinator/channel_transport.hpp` was written against.
 *
 * Generations are what make this safe. A path carries one, bumped on every
 * replan, and its high half is the coordinator's epoch. A replica applies a
 * rule only if it is at least as new as the one it holds, so a message that
 * arrives late -- channels are per-replica and nothing orders them against
 * each other -- cannot resurrect a path that has already been replanned.
 *
 * HiCR-free: the table is fed decoded messages, by a channel in production and
 * directly in a test.
 *
 * Thread-safe: in a channel deployment the coordinator's rules arrive on the
 * module's service thread while HTTP handler threads are reading the table to
 * route requests, so the table locks itself rather than trusting every caller
 * to agree on an external mutex.
 */

#include <cstdint>
#include <list>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

#include <serving/coordinator/rule_message.hpp>

namespace serving::replica
{

/// Where a session goes after this replica.
struct NextHop
{
  std::string name;
  /// The HiCR instance behind `name`, so the replica does not resolve it.
  uint64_t instance   = 0;
  uint64_t generation = 0;

  /**
   * Where to reach it over HTTP, when that is how this deployment talks.
   *
   * The binary rule message addresses the next hop by instance, because that
   * is what a control plane routes on. A deployment whose replicas talk HTTP
   * needs an address instead, and nothing on the wire carries one -- so it is
   * supplied alongside. Empty means "addressed by instance", which a forward
   * over HTTP cannot use.
   */
  std::string endpoint;

  bool operator==(const NextHop &) const = default;
};

/// Why an applied message did or did not change anything, so a replica can log
/// something better than silence.
enum class RuleOutcome
{
  Applied,
  /// Older than what is held: a late message, ignored.
  Stale,
  /// A revoke for a session that was not held.
  Unknown,
  Revoked,
};

[[nodiscard]] const char *ruleOutcomeName(RuleOutcome outcome);

/**
 * A replica's forwarding rules, newest-used kept.
 *
 * Bounded, like the coordinator's own path table: a replica serving a long tail
 * of conversations would otherwise hold a rule per session for ever. Evicting
 * the least recently used is safe in a way dropping an arbitrary one is not --
 * a session whose rule is gone is a cache miss the coordinator can replan, and
 * the least recently used is the one least likely to come back.
 */
class RuleTable
{
  public:

  explicit RuleTable(size_t maxSessions = 65536);

  /// Apply a published rule. Older generations are ignored.
  ///
  /// `endpoint` is where the hop is reachable over HTTP; the wire format
  /// carries an instance id instead, so a deployment that forwards over HTTP
  /// supplies it here.
  RuleOutcome apply(const coordinator::RulePublish &rule, std::string endpoint = {});

  /// Withdraw one. A revoke older than the rule held is ignored, so a revoke
  /// racing the replan that replaced it cannot undo the new rule.
  RuleOutcome revoke(const coordinator::RuleRevoke &revoke);

  /// Where this session goes next, or nullopt when nothing is held -- which is
  /// the replica's signal to ask the coordinator rather than guess.
  [[nodiscard]] std::optional<NextHop> lookup(const std::string &sessionId);

  /// Without marking it used, for tests and for reporting.
  [[nodiscard]] std::optional<NextHop> peek(const std::string &sessionId) const;

  [[nodiscard]] size_t size() const;
  [[nodiscard]] size_t capacity() const { return _max; }

  /// Rules dropped because the table was full.
  [[nodiscard]] int evicted() const;

  void clear();

  private:

  using Order = std::list<std::string>;

  void touch(Order::iterator it);

  /// Guards everything below. Mutable because `peek` reorders nothing but
  /// still reads state another thread may be rewriting.
  mutable std::mutex _mutex;

  size_t                                                               _max;
  Order                                                                _order;
  std::unordered_map<std::string, std::pair<NextHop, Order::iterator>> _index;
  int                                                                  _evicted = 0;
};

} // namespace serving::replica
