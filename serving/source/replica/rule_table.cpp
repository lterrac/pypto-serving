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
  : _max(maxSessions)
{
  if (_max < 1) { throw std::invalid_argument("max_sessions must be positive"); }
}

void RuleTable::touch(Order::iterator it) { _order.splice(_order.end(), _order, it); }

RuleOutcome RuleTable::apply(const coordinator::RulePublish &rule, std::string endpoint)
{
  const std::lock_guard<std::mutex> lock(_mutex);

  const auto it = _index.find(rule.sessionId);
  if (it != _index.end())
  {
    // Channels are per replica and nothing orders them against each other, so
    // a message can arrive after a newer one. Equal generations are applied
    // too: a republish of the same plan is not a conflict.
    if (rule.generation < it->second.first.generation) { return RuleOutcome::Stale; }
    it->second.first = NextHop{rule.nextHopName, rule.nextHopInstance, rule.generation, std::move(endpoint)};
    touch(it->second.second);
    return RuleOutcome::Applied;
  }

  _order.push_back(rule.sessionId);
  _index.emplace(rule.sessionId, std::pair{NextHop{rule.nextHopName, rule.nextHopInstance, rule.generation, std::move(endpoint)}, std::prev(_order.end())});

  while (_index.size() > _max)
  {
    // A session whose rule is gone is a cache miss the coordinator can replan;
    // the least recently used is the one least likely to come back.
    const std::string evicted = _order.front();
    _order.pop_front();
    _index.erase(evicted);
    _evicted += 1;
  }
  return RuleOutcome::Applied;
}

RuleOutcome RuleTable::revoke(const coordinator::RuleRevoke &message)
{
  const std::lock_guard<std::mutex> lock(_mutex);

  const auto it = _index.find(message.sessionId);
  if (it == _index.end()) { return RuleOutcome::Unknown; }

  // A revoke racing the replan that replaced the rule must not undo the new
  // one: the revoke belongs to the generation being withdrawn, so anything
  // strictly newer survives it.
  if (message.generation < it->second.first.generation) { return RuleOutcome::Stale; }

  _order.erase(it->second.second);
  _index.erase(it);
  return RuleOutcome::Revoked;
}

std::optional<NextHop> RuleTable::lookup(const std::string &sessionId)
{
  const std::lock_guard<std::mutex> lock(_mutex);

  const auto it = _index.find(sessionId);
  if (it == _index.end()) { return std::nullopt; }
  touch(it->second.second);
  return it->second.first;
}

std::optional<NextHop> RuleTable::peek(const std::string &sessionId) const
{
  const std::lock_guard<std::mutex> lock(_mutex);

  const auto it = _index.find(sessionId);
  return it == _index.end() ? std::nullopt : std::optional{it->second.first};
}

void RuleTable::clear()
{
  const std::lock_guard<std::mutex> lock(_mutex);

  _order.clear();
  _index.clear();
}

size_t RuleTable::size() const
{
  const std::lock_guard<std::mutex> lock(_mutex);
  return _index.size();
}

int RuleTable::evicted() const
{
  const std::lock_guard<std::mutex> lock(_mutex);
  return _evicted;
}

} // namespace serving::replica
