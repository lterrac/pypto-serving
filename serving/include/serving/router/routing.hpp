#pragma once

/**
 * Session affinity and replica selection.
 *
 * A conversation is pinned to the replica holding its prefix cache: prefix
 * blocks cover the prompt at admission, so the next turn re-prefills only the
 * new text. Affinity is a preference bounded by affinitySlack -- prefix blocks
 * are evictable, and one saved prefill is worth less than waiting behind a
 * saturated replica.
 */

#include <chrono>
#include <functional>
#include <list>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace serving::router
{

/// Consecutive failed probes before a replica leaves rotation. One blip should
/// not depin every session; recovery is immediate on the first success.
inline constexpr int UNHEALTHY_THRESHOLD = 2;

/// Session ids are client-supplied, so the directory is an unbounded map keyed
/// by untrusted input until its entries expire. Cap it and evict
/// least-recently-used pins: losing a pin costs one prefill, unbounded growth
/// costs the process.
inline constexpr size_t DEFAULT_MAX_SESSIONS = 100000;

/// Raised when every replica in the table is unroutable.
class NoReplicaAvailable : public std::runtime_error
{
  public:

  explicit NoReplicaAvailable(const std::string &what)
    : std::runtime_error(what)
  {}
};

[[nodiscard]] std::string newSessionId();

/// Which pipeline stage a replica belongs to. A model split across nodes runs
/// as a chain of partitions; each has its own coordinator and its own replicas,
/// and each scales independently of the others.
using PartitionId = int;

/// One serving replica reachable over HTTP.
struct ReplicaSpec
{
  std::string name;
  /// The partition this replica serves. Single-partition deployments leave it 0.
  PartitionId partition = 0;
  std::string host;
  int         port = 0;
  /// Per replica so a deployment whose replicas are not on a trusted network can
  /// terminate TLS in front of them; prompts and generated text cross this hop
  /// in the clear otherwise.
  std::string scheme = "http";

  [[nodiscard]] std::string baseUrl() const { return scheme + "://" + host + ":" + std::to_string(port); }
};

/// The routing-relevant slice of `RouterConfig`.
struct RoutingConfig
{
  double sessionTtlSeconds = 1800.0;
  int    affinitySlack     = 8;
};

/**
 * Maps a conversation id to the replica holding its KV.
 *
 * The clock is injectable so expiry is testable without sleeping -- the Python
 * takes the same argument for the same reason.
 */
class SessionDirectory
{
  public:

  using Clock = std::function<double()>;

  explicit SessionDirectory(double ttlSeconds, Clock clock = {}, size_t maxSessions = DEFAULT_MAX_SESSIONS);

  [[nodiscard]] size_t size() const { return _index.size(); }

  /// The pinned replica name, dropping the pin if it has expired.
  [[nodiscard]] std::optional<std::string> lookup(const std::string &sessionId);

  void pin(const std::string &sessionId, const std::string &replicaName);

  void forget(const std::string &sessionId);

  /**
   * Drop every pin to one replica, returning how many were dropped.
   *
   * Used when a replica starts draining: its sessions should re-route on their
   * next turn instead of waiting out the TTL pointing at something that is
   * going away.
   */
  int forgetReplica(const std::string &replicaName);

  /// Drop every expired pin. Lookup only expires pins it touches, so this bounds
  /// the memory held by sessions that are never seen again.
  int sweep();

  private:

  struct Entry
  {
    std::string replicaName;
    double      lastSeen = 0.0;
  };

  /// Most-recently pinned at the back, so eviction pops the front.
  using Order = std::list<std::string>;

  void touch(Order::iterator it);

  double _ttl;
  Clock  _clock;
  size_t _max;

  Order                                                              _order;
  std::unordered_map<std::string, std::pair<Entry, Order::iterator>> _index;
};

/// Mutable routing state for one replica.
struct ReplicaState
{
  ReplicaSpec spec;
  /// Carried explicitly: the rotating tiebreak needs it, and looking it up by
  /// value would rely on structural equality.
  int index       = 0;
  int outstanding = 0;
  int routed      = 0;
  /// Statically configured replicas start routable: one that is still loading
  /// refuses the connection anyway, and the first probe corrects an optimistic
  /// guess quickly. A replica the router launches is added with ready=false,
  /// because it provably cannot serve for the minutes its model takes to load.
  bool ready    = true;
  int  failures = 0;
  /// Excluded from routing while it finishes what it already has.
  bool draining = false;
  /// True only for replicas this router started, and so may stop.
  bool owned = false;

  [[nodiscard]] const std::string &name() const { return spec.name; }
  [[nodiscard]] bool               routable() const { return ready && !draining; }
};

struct RoutingDecision
{
  ReplicaSpec replica;
  bool        affinityHit = false;
};

/// Owns replica state, picks a replica per request, and counts what happened.
class ReplicaRegistry
{
  public:

  ReplicaRegistry(RoutingConfig config, SessionDirectory &sessions, const std::vector<ReplicaSpec> &replicas);

  [[nodiscard]] const std::vector<ReplicaState> &states() const { return _states; }
  [[nodiscard]] ReplicaState                    *state(const std::string &name);

  [[nodiscard]] int readyCount() const;
  [[nodiscard]] int totalRouted() const;

  /**
   * Register a replica at runtime.
   *
   * Appending keeps the rotating tiebreak honest for free: both sites that use
   * `index` recompute the modulus from the live list, so an index of
   * `states().size()` stays inside it.
   *
   * Defaults to not-ready: a launched replica cannot serve until its model is
   * loaded, and the health poller is what promotes it.
   */
  ReplicaState &add(const ReplicaSpec &spec, bool ready = false, bool owned = false);

  /**
   * Drop a replica and reindex the rest.
   *
   * Reindexing is the point: leaving gaps makes `(index - counter) % count`
   * alias two replicas onto the same tiebreak slot, which is unfair rather than
   * wrong, but silently so.
   */
  bool remove(const std::string &name);

  /// Take a replica out of routing without dropping its in-flight work. Pins go
  /// at the same moment, so a session re-routes on its next turn rather than
  /// waiting out the TTL.
  void setDraining(const std::string &name, bool draining);

  void setReady(const std::string &name, bool ready);

  void acquire(const std::string &name);
  void release(const std::string &name);

  /// Pick a replica for this session and record the pin.
  [[nodiscard]] RoutingDecision select(const std::string &sessionId);

  [[nodiscard]] int affinityHits() const { return _affinityHits; }
  [[nodiscard]] int rejected() const { return _rejected; }

  private:

  [[nodiscard]] ReplicaState &leastLoaded(const std::vector<ReplicaState *> &candidates);

  RoutingConfig             _config;
  SessionDirectory         &_sessions;
  std::vector<ReplicaState> _states;

  /// Rotating tiebreak, matching AsyncLLMEngine._select_replica: equal-load
  /// replicas are taken in turn instead of always the lowest index.
  int _routeCounter = 0;
  int _affinityHits = 0;
  int _rejected     = 0;
};

} // namespace serving::router
