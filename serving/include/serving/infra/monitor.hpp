#pragma once

/**
 * Noticing that a replica has gone (`scaling-pypto-serving.pptx`, M3).
 *
 * The deck's complaint about the Python stack was that nothing noticed: a dead
 * worker surfaced only after a twenty-minute timeout, while the server kept
 * accepting requests, returned truncated answers marked as normal completions,
 * and `/health` stayed green. Everything downstream of noticing already works
 * here -- routing drops a dead replica and the supervisor rebuilds it -- so
 * this is the piece that closes the loop.
 *
 * It only writes a phase. Taking the replica out of routing, revoking the paths
 * through it and starting a replacement are all consequences of the map saying
 * `dead`, which is what makes one map the single source of truth rather than
 * three components each with an opinion.
 */

#include <map>
#include <string>
#include <vector>

#include <serving/infra/launcher.hpp>
#include <serving/infra/state.hpp>

namespace serving::infra
{

struct MonitorConfig
{
  /// Consecutive failed probes before a replica is declared dead. More than
  /// one, because a single refused connection is as likely to be a blip as a
  /// death, and declaring a healthy replica dead costs a model load.
  int failuresBeforeDead = 3;

  /// Throws std::invalid_argument on a configuration that cannot work.
  void validate() const;
};

struct MonitorReport
{
  /// Probed and found gone: marked dead, which releases their devices and
  /// takes them out of routing.
  std::vector<std::string> lost;
  /// Answered again before reaching the threshold. Their count is forgotten,
  /// so a replica has to fail `failuresBeforeDead` times *in a row*.
  std::vector<std::string> recovered;
  int                      probed = 0;

  [[nodiscard]] bool empty() const { return lost.empty() && recovered.empty(); }
};

/**
 * Polls the replicas that are supposed to be serving.
 *
 * Only `active` ones: a `pending` replica is the supervisor's business and is
 * expected not to answer yet, a `draining` one is leaving under its own steam,
 * and a `dead` one has already been noticed.
 */
class HealthMonitor
{
  public:

  HealthMonitor(StateMap &map, HealthProbe &probe, MonitorConfig config = {});

  MonitorReport service();

  /// Consecutive failures recorded for a replica, so a caller can report that
  /// something is wobbling before it is declared gone.
  [[nodiscard]] int failures(const std::string &replica) const;

  private:

  StateMap                  &_map;
  HealthProbe               &_probe;
  MonitorConfig              _config;
  std::map<std::string, int> _failures;
};

} // namespace serving::infra
