#include <serving/infra/launcher.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace serving::infra
{

namespace
{

double monotonicSeconds() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

} // namespace

const char *readinessName(Readiness readiness)
{
  switch (readiness)
  {
  case Readiness::Loading: return "loading";
  case Readiness::Ready: return "ready";
  case Readiness::Failed: return "failed";
  }
  return "loading";
}

Backoff::Backoff(double baseSeconds, double maxSeconds, int maxAttempts)
  : _base(baseSeconds),
    _max(maxSeconds),
    _maxAttempts(maxAttempts)
{
  if (_base <= 0.0) { throw std::invalid_argument("backoff base must be positive"); }
  if (_max < _base) { throw std::invalid_argument("backoff ceiling must not be below its base"); }
  if (_maxAttempts < 1) { throw std::invalid_argument("backoff must allow at least one attempt"); }
}

double Backoff::delayAfter(int attempts) const
{
  if (attempts <= 0) { return 0.0; }
  // 2^(n-1) so the wait after the first failure is the base, not twice it.
  // ldexp rather than pow: the exponent grows fast enough to overflow a double
  // for a long-running deployment, and the ceiling is what matters anyway.
  const double grown = std::ldexp(_base, std::min(attempts - 1, 60));
  return std::min(grown, _max);
}

Supervisor::Supervisor(StateMap &map, ReplicaStarter &starter, HealthProbe &probe, Backoff backoff, Clock clock)
  : _map(map),
    _starter(starter),
    _probe(probe),
    _backoff(backoff),
    _clock(clock ? std::move(clock) : Clock{monotonicSeconds})
{}

int Supervisor::attempts(const std::string &replica) const
{
  const auto it = _attempts.find(replica);
  return it == _attempts.end() ? 0 : it->second.failures;
}

SupervisionReport Supervisor::service()
{
  SupervisionReport report;
  const double      now = _clock();

  // reconcile() says what is missing; this decides nothing beyond when to try.
  for (const PendingAction &action : reconcile(_map))
  {
    const Replica *replica = _map.replica(action.replica);
    if (replica == nullptr) { continue; }

    if (action.action == Action::Stop)
    {
      _starter.stop(*replica);
      _map.setPhase(action.replica, Phase::Dead);
      _attempts.erase(action.replica);
      report.stopped.push_back(action.replica);
      continue;
    }

    // Draining is routing's business: it keeps serving what it has and takes
    // nothing new, and nothing needs starting or stopping for that.
    if (action.action == Action::Drain) { continue; }

    Attempt &attempt = _attempts[action.replica];

    if (attempt.inFlight)
    {
      // Started, waiting for it to say it is serving. Its own /health is the
      // authority: a process that exists is not a replica that can serve.
      const Readiness readiness = _probe.check(*replica);
      if (readiness == Readiness::Ready)
      {
        // To the phase the deployment asked for, not always Active: a spare is
        // started so it can wait, and promoting it to serving would defeat it.
        _map.setPhase(action.replica, replica->desiredPhase == Phase::Reserved ? Phase::Reserved : Phase::Active);
        _attempts.erase(action.replica);
        report.promoted.push_back(action.replica);
        continue;
      }

      // Loading is worth the wait; failed never will be, and waiting out a
      // ready timeout measured in minutes for a process that has already
      // exited is the difference between a slow deployment and a stuck one.
      if (readiness == Readiness::Loading && now - attempt.startedAt < _readyTimeout)
      {
        report.waiting += 1;
        continue;
      }

      // It never came up. Count it and wait before trying again.
      _starter.stop(*replica);
      attempt.inFlight = false;
      attempt.failures += 1;
      attempt.nextTryAt = now + _backoff.delayAfter(attempt.failures);

      if (_backoff.exhausted(attempt.failures))
      {
        // Dead rather than stuck pending: dead releases its devices, so the
        // capacity can be rebuilt somewhere that might work.
        _map.setPhase(action.replica, Phase::Dead);
        report.gaveUp.push_back(action.replica);
        _attempts.erase(action.replica);
      }
      else { report.retrying.push_back(action.replica); }
      continue;
    }

    // Already serving: a reserved spare being activated, which is the whole
    // point of keeping one. No launch, no model load -- one phase change.
    //
    // Only from reserved. The probe was once the only question asked here, on
    // the reasoning that a replica which was never started cannot answer --
    // and that is false. Something else listening on the port answers too: a
    // run that leaked a replica left one holding a card, and the next
    // deployment adopted it, reporting a replica as serving that it had never
    // started and could not stop. A pending or dead replica gets launched.
    if (replica->phase == Phase::Reserved && _probe.check(*replica) == Readiness::Ready)
    {
      _map.setPhase(action.replica, replica->desiredPhase == Phase::Reserved ? Phase::Reserved : Phase::Active);
      _attempts.erase(action.replica);
      report.promoted.push_back(action.replica);
      continue;
    }

    if (now < attempt.nextTryAt)
    {
      report.waiting += 1;
      continue;
    }

    if (!_starter.start(*replica))
    {
      attempt.failures += 1;
      attempt.nextTryAt = now + _backoff.delayAfter(attempt.failures);
      if (_backoff.exhausted(attempt.failures))
      {
        _map.setPhase(action.replica, Phase::Dead);
        report.gaveUp.push_back(action.replica);
        _attempts.erase(action.replica);
      }
      else { report.retrying.push_back(action.replica); }
      continue;
    }

    attempt.inFlight  = true;
    attempt.startedAt = now;
    // Pending until its health says otherwise. A replica that was dead and is
    // being rebuilt goes back to pending rather than straight to active.
    if (replica->phase != Phase::Pending) { _map.setPhase(action.replica, Phase::Pending); }
    report.started.push_back(action.replica);
  }

  return report;
}

} // namespace serving::infra
