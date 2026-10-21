#include <serving/replica/rule_table.hpp>

#include <stdexcept>

namespace serving::replica
{

const char *ruleOutcomeName(RuleOutcome outcome)
{
  switch (outcome)
  {
  case RuleOutcome::Applied: return "applied";
  case RuleOutcome::Stale: return "stale";
  case RuleOutcome::Unknown: return "unknown";
  case RuleOutcome::Revoked: return "revoked";
  }
  return "applied";
}

RuleTable::RuleTable(size_t maxSessions)
  : _rules(maxSessions)
{
  if (maxSessions < 1) { throw std::invalid_argument("max_sessions must be positive"); }
}

RuleOutcome RuleTable::apply(const coordinator::RulePublish &rule, std::string endpoint)
{
  const std::lock_guard<std::mutex> lock(_mutex);

  // Channels are per replica and nothing orders them against each other, so a
  // message can arrive after a newer one. Equal generations are applied too: a
  // republish of the same plan is not a conflict.
  if (const NextHop *held = _rules.peek(rule.sessionId); held != nullptr && rule.generation < held->generation) { return RuleOutcome::Stale; }

  _rules.put(rule.sessionId, NextHop{rule.nextHopName, rule.nextHopInstance, rule.generation, std::move(endpoint)});
  // A session whose rule is gone is a cache miss the coordinator can replan;
  // the least recently used is the one least likely to come back, and this
  // table has nothing to withdraw for what it drops.
  _rules.trim([](const std::string &, const NextHop &) {});
  return RuleOutcome::Applied;
}

RuleOutcome RuleTable::revoke(const coordinator::RuleRevoke &message)
{
  const std::lock_guard<std::mutex> lock(_mutex);

  const NextHop *held = _rules.peek(message.sessionId);
  if (held == nullptr) { return RuleOutcome::Unknown; }

  // A revoke racing the replan that replaced the rule must not undo the new
  // one: the revoke belongs to the generation being withdrawn, so anything
  // strictly newer survives it.
  if (message.generation < held->generation) { return RuleOutcome::Stale; }

  _rules.erase(message.sessionId);
  return RuleOutcome::Revoked;
}

std::optional<NextHop> RuleTable::lookup(const std::string &sessionId)
{
  const std::lock_guard<std::mutex> lock(_mutex);

  const NextHop *held = _rules.find(sessionId);
  return held == nullptr ? std::nullopt : std::optional{*held};
}

std::optional<NextHop> RuleTable::peek(const std::string &sessionId) const
{
  const std::lock_guard<std::mutex> lock(_mutex);

  const NextHop *held = _rules.peek(sessionId);
  return held == nullptr ? std::nullopt : std::optional{*held};
}

void RuleTable::clear()
{
  const std::lock_guard<std::mutex> lock(_mutex);

  _rules.clear();
}

size_t RuleTable::size() const
{
  const std::lock_guard<std::mutex> lock(_mutex);
  return _rules.size();
}

int RuleTable::evicted() const
{
  const std::lock_guard<std::mutex> lock(_mutex);
  return static_cast<int>(_rules.evicted());
}

} // namespace serving::replica
