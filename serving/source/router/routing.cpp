#include <serving/router/routing.hpp>

#include <algorithm>
#include <iomanip>
#include <random>
#include <sstream>

namespace serving::router
{

namespace
{

/// Python's `%` is always non-negative; C++'s is not. The tiebreak below
/// subtracts a counter from an index, so a naive `%` would give a negative key
/// and silently reorder the rotation.
int pythonMod(int value, int modulus)
{
  if (modulus <= 0) { return 0; }
  return ((value % modulus) + modulus) % modulus;
}

double monotonicSeconds() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

} // namespace

std::string newSessionId()
{
  // uuid4().hex: 32 hex characters from a random 128-bit value.
  static thread_local std::mt19937_64 rng{std::random_device{}()};
  std::ostringstream                  out;
  out << std::hex << std::setfill('0') << std::setw(16) << rng() << std::setw(16) << rng();
  return out.str();
}

// ---------------------------------------------------------------------------
// SessionDirectory
// ---------------------------------------------------------------------------

SessionDirectory::SessionDirectory(double ttlSeconds, Clock clock, size_t maxSessions)
  : _ttl(ttlSeconds),
    _clock(clock ? std::move(clock) : Clock{monotonicSeconds}),
    _max(maxSessions)
{
  if (ttlSeconds <= 0) { throw std::invalid_argument("ttl_seconds must be positive"); }
  if (maxSessions < 1) { throw std::invalid_argument("max_sessions must be positive"); }
}

void SessionDirectory::touch(Order::iterator it)
{
  // Move to the back: most recently pinned.
  _order.splice(_order.end(), _order, it);
}

std::optional<std::string> SessionDirectory::lookup(const std::string &sessionId)
{
  const auto it = _index.find(sessionId);
  if (it == _index.end()) { return std::nullopt; }
  if (_clock() - it->second.first.lastSeen > _ttl)
  {
    _order.erase(it->second.second);
    _index.erase(it);
    return std::nullopt;
  }
  return it->second.first.replicaName;
}

void SessionDirectory::pin(const std::string &sessionId, const std::string &replicaName)
{
  const auto it = _index.find(sessionId);
  if (it != _index.end())
  {
    it->second.first = Entry{replicaName, _clock()};
    touch(it->second.second);
  }
  else
  {
    _order.push_back(sessionId);
    auto position     = std::prev(_order.end());
    _index[sessionId] = {Entry{replicaName, _clock()}, position};
  }

  // Expired pins are reclaimed before anything live is evicted.
  if (_index.size() > _max) { sweep(); }
  while (_index.size() > _max)
  {
    const std::string evicted = _order.front();
    _order.pop_front();
    _index.erase(evicted);
  }
}

void SessionDirectory::forget(const std::string &sessionId)
{
  const auto it = _index.find(sessionId);
  if (it == _index.end()) { return; }
  _order.erase(it->second.second);
  _index.erase(it);
}

int SessionDirectory::forgetReplica(const std::string &replicaName)
{
  int dropped = 0;
  for (auto it = _index.begin(); it != _index.end();)
  {
    if (it->second.first.replicaName == replicaName)
    {
      _order.erase(it->second.second);
      it = _index.erase(it);
      dropped += 1;
    }
    else { ++it; }
  }
  return dropped;
}

int SessionDirectory::sweep()
{
  const double now     = _clock();
  int          expired = 0;
  for (auto it = _index.begin(); it != _index.end();)
  {
    if (now - it->second.first.lastSeen > _ttl)
    {
      _order.erase(it->second.second);
      it = _index.erase(it);
      expired += 1;
    }
    else { ++it; }
  }
  return expired;
}

// ---------------------------------------------------------------------------
// ReplicaRegistry
// ---------------------------------------------------------------------------

ReplicaRegistry::ReplicaRegistry(RoutingConfig config, SessionDirectory &sessions, const std::vector<ReplicaSpec> &replicas)
  : _config(config),
    _sessions(sessions)
{
  _states.reserve(replicas.size());
  for (size_t i = 0; i < replicas.size(); ++i)
  {
    ReplicaState state;
    state.spec  = replicas[i];
    state.index = static_cast<int>(i);
    _states.push_back(std::move(state));
  }
}

ReplicaState *ReplicaRegistry::state(const std::string &name)
{
  const auto it = std::find_if(_states.begin(), _states.end(), [&](const ReplicaState &s) { return s.name() == name; });
  return it == _states.end() ? nullptr : &*it;
}

int ReplicaRegistry::readyCount() const
{
  return static_cast<int>(std::count_if(_states.begin(), _states.end(), [](const ReplicaState &s) { return s.routable(); }));
}

int ReplicaRegistry::totalRouted() const
{
  int total = 0;
  for (const ReplicaState &s : _states) { total += s.routed; }
  return total;
}

ReplicaState &ReplicaRegistry::add(const ReplicaSpec &spec, bool ready, bool owned)
{
  if (state(spec.name) != nullptr) { throw std::invalid_argument("replica '" + spec.name + "' is already registered"); }
  ReplicaState fresh;
  fresh.spec  = spec;
  fresh.index = static_cast<int>(_states.size());
  fresh.ready = ready;
  fresh.owned = owned;
  _states.push_back(std::move(fresh));
  return _states.back();
}

bool ReplicaRegistry::remove(const std::string &name)
{
  const auto it = std::find_if(_states.begin(), _states.end(), [&](const ReplicaState &s) { return s.name() == name; });
  if (it == _states.end()) { return false; }
  _states.erase(it);
  for (size_t position = 0; position < _states.size(); ++position) { _states[position].index = static_cast<int>(position); }
  _routeCounter = _states.empty() ? 0 : pythonMod(_routeCounter, static_cast<int>(_states.size()));
  _sessions.forgetReplica(name);
  return true;
}

void ReplicaRegistry::setDraining(const std::string &name, bool draining)
{
  ReplicaState *target = state(name);
  if (target == nullptr || target->draining == draining) { return; }
  target->draining = draining;
  if (draining) { _sessions.forgetReplica(name); }
}

void ReplicaRegistry::setReady(const std::string &name, bool ready)
{
  ReplicaState *target = state(name);
  if (target == nullptr || target->ready == ready) { return; }
  target->ready = ready;
}

void ReplicaRegistry::acquire(const std::string &name)
{
  ReplicaState *target = state(name);
  if (target != nullptr) { target->outstanding += 1; }
}

void ReplicaRegistry::release(const std::string &name)
{
  ReplicaState *target = state(name);
  if (target != nullptr) { target->outstanding = std::max(0, target->outstanding - 1); }
}

ReplicaState &ReplicaRegistry::leastLoaded(const std::vector<ReplicaState *> &candidates)
{
  const int  count = static_cast<int>(_states.size());
  const auto key   = [&](const ReplicaState *s) { return std::tuple{s->outstanding, pythonMod(s->index - _routeCounter, count), s->index}; };
  auto       best  = candidates.front();
  for (ReplicaState *candidate : candidates)
  {
    if (key(candidate) < key(best)) { best = candidate; }
  }
  return *best;
}

RoutingDecision ReplicaRegistry::select(const std::string &sessionId)
{
  std::vector<ReplicaState *> candidates;
  for (ReplicaState &s : _states)
  {
    if (s.routable()) { candidates.push_back(&s); }
  }
  if (candidates.empty())
  {
    _rejected += 1;
    throw NoReplicaAvailable("no routable replica");
  }

  ReplicaState &least       = leastLoaded(candidates);
  const auto    pinned      = _sessions.lookup(sessionId);
  ReplicaState *pinnedState = pinned.has_value() ? state(*pinned) : nullptr;

  ReplicaState *chosen      = &least;
  bool          affinityHit = false;

  if (pinnedState != nullptr && pinnedState->routable())
  {
    // Past the slack, load wins: one saved prefill is worth less than an
    // unbounded wait behind a saturated replica.
    if (pinnedState->outstanding <= least.outstanding + _config.affinitySlack)
    {
      chosen      = pinnedState;
      affinityHit = true;
    }
  }

  if (!affinityHit)
  {
    // Only advance the rotation when load decided the route, so a stream of
    // affine requests does not skew the tiebreak.
    _routeCounter = pythonMod(chosen->index + 1, static_cast<int>(_states.size()));
  }

  _sessions.pin(sessionId, chosen->name());
  chosen->routed += 1;
  if (affinityHit) { _affinityHits += 1; }
  return RoutingDecision{chosen->spec, affinityHit};
}

} // namespace serving::router
