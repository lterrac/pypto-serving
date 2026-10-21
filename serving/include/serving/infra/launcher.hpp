#pragma once

/**
 * Starting replicas, and retrying when they do not come up.
 *
 * `reconcile()` says what the deployment is missing; this acts on it. The two
 * are split because deciding is cheap and idempotent while starting a process
 * is neither, and because the brief asks for the retry to behave the way
 * Kubernetes does: a failed launch is tried again after an interval that grows.
 *
 * A started replica is not a serving one. The model takes minutes to load, so
 * the supervisor starts it `pending` -- which routing already refuses to send
 * to -- and promotes it to `active` only when its own health endpoint says it
 * is ready. That is the brief's "route requests to the replica only when it's
 * ready", enforced here rather than hoped for.
 *
 * Nothing in here knows how a replica is actually started. `ReplicaStarter` is
 * a process on this node, an RPC to a control plane, or a fake in a test.
 */

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include <serving/infra/state.hpp>

namespace serving::infra
{

/**
 * Exponential backoff with a ceiling, per the brief's "exponential backoff like
 * k8s does".
 *
 * The delay before attempt `n` is `base * 2^n`, capped. Attempt 0 is immediate:
 * the first try is not a retry.
 */
class Backoff
{
  public:

  Backoff(double baseSeconds = 1.0, double maxSeconds = 60.0, int maxAttempts = 5);

  /// Seconds to wait after `attempts` failures before trying again.
  [[nodiscard]] double delayAfter(int attempts) const;

  /// Whether this many failures is enough to stop trying.
  [[nodiscard]] bool exhausted(int attempts) const { return attempts >= _maxAttempts; }

  [[nodiscard]] int maxAttempts() const { return _maxAttempts; }

  private:

  double _base;
  double _max;
  int    _maxAttempts;
};

/// How a replica is actually started. The supervisor never does this itself.
class ReplicaStarter
{
  public:

  virtual ~ReplicaStarter() = default;

  /// Begin starting it. False means it could not even be launched, which is a
  /// failed attempt; true only means the attempt began, not that it serves.
  [[nodiscard]] virtual bool start(const Replica &replica) = 0;

  /// Take it down. Called when the deployment no longer wants it, and when it
  /// failed often enough to give up on.
  virtual void stop(const Replica &replica) = 0;

  [[nodiscard]] virtual const char *kind() const = 0;
};

/**
 * Where a started replica is between launched and serving.
 *
 * Three states, not two, because "not serving" covers two cases a supervisor
 * must treat differently: one is worth waiting minutes for, the other will
 * never resolve. Both look identical over HTTP -- a refused connection.
 */
enum class Readiness
{
  /// Not serving yet, and might be: keep waiting.
  Loading,
  Ready,
  /// Will not become ready: the attempt has already failed.
  Failed,
};

[[nodiscard]] const char *readinessName(Readiness readiness);

/// Whether a started replica is serving yet. In production this is its own
/// `/health`, which is 503 until the engine loop is live.
class HealthProbe
{
  public:

  virtual ~HealthProbe() = default;

  [[nodiscard]] virtual Readiness check(const Replica &replica) = 0;
};

/// What one tick did, so a caller and a test can see it without reading phases.
struct SupervisionReport
{
  std::vector<std::string> started;
  /// Came up and is now serving.
  std::vector<std::string> promoted;
  /// The attempt failed; it will be tried again after the backoff.
  std::vector<std::string> retrying;
  /// Tried enough times. Marked dead, so the capacity can be rebuilt elsewhere.
  std::vector<std::string> gaveUp;
  std::vector<std::string> stopped;
  /// Waiting out a backoff interval, or still loading.
  int waiting = 0;

  [[nodiscard]] bool empty() const { return started.empty() && promoted.empty() && retrying.empty() && gaveUp.empty() && stopped.empty(); }
};

/**
 * Drives the map towards what it says it wants.
 *
 * One `service()` per tick. The clock is injected because the whole of the
 * backoff is about time passing, and a test that had to sleep through it would
 * be a test nobody runs.
 */
class Supervisor
{
  public:

  using Clock = std::function<double()>;

  Supervisor(StateMap &map, ReplicaStarter &starter, HealthProbe &probe, Backoff backoff = {}, Clock clock = {});

  /// How long a replica may take to answer its health probe before the attempt
  /// counts as failed. Longer than a model load, or every launch gives up.
  void setReadyTimeout(double seconds) { _readyTimeout = seconds; }

  /// How long a draining replica is given to finish what it is holding before
  /// it is stopped. Routing has already stopped sending it anything new, so
  /// this bounds the wait rather than measuring it: a drain with no deadline
  /// never converges, and the card never comes back.
  void setDrainTimeout(double seconds) { _drainTimeout = seconds; }

  SupervisionReport service();

  /// Failed attempts recorded for a replica, for a caller that wants to report
  /// why something is taking so long.
  [[nodiscard]] int attempts(const std::string &replica) const;

  private:

  struct Attempt
  {
    int    failures  = 0;
    double startedAt = 0.0;
    bool   inFlight  = false;
    double nextTryAt = 0.0;
  };

  StateMap       &_map;
  ReplicaStarter &_starter;
  HealthProbe    &_probe;
  Backoff         _backoff;
  Clock           _clock;
  double          _readyTimeout = 600.0;
  double          _drainTimeout = 120.0;

  std::map<std::string, Attempt> _attempts;
  /// When each replica was first seen draining, so the deadline is measured
  /// from the drain rather than from the deployer's start.
  std::map<std::string, double> _drainingSince;
};

} // namespace serving::infra
