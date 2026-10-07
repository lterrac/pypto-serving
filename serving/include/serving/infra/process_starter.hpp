#pragma once

/**
 * Starting a replica as a process on this node, and asking it whether it is up.
 *
 * The production half of `launcher.hpp`. Split out because it is the only part
 * that touches the operating system and the network, which is what makes the
 * supervisor itself testable without either.
 */

#include <map>
#include <string>
#include <vector>

#include <serving/infra/launcher.hpp>

namespace serving::infra
{

/**
 * Launches a command per replica.
 *
 * The command is a template; `{name}`, `{node}`, `{device}`, `{devices}` and
 * `{port}` are replaced from the replica's own record, so one line describes a
 * whole fleet:
 *
 *     python -m pypto_serving.cpp --model /models/Qwen3-14B --device {device} --port {port}
 *
 * Uses `posix_spawn` rather than fork and exec: a supervisor has threads, and
 * forking a threaded process to exec from the child is the kind of thing that
 * works until it does not.
 */
class ProcessStarter final : public ReplicaStarter
{
  public:

  /// `command` is run through `/bin/sh -c` after substitution, so a template
  /// can carry environment assignments and redirection the way an operator
  /// would write them by hand.
  explicit ProcessStarter(std::string commandTemplate);
  ~ProcessStarter() override;

  ProcessStarter(const ProcessStarter &)            = delete;
  ProcessStarter &operator=(const ProcessStarter &) = delete;

  [[nodiscard]] bool        start(const Replica &replica) override;
  void                      stop(const Replica &replica) override;
  [[nodiscard]] const char *kind() const override { return "process"; }

  /// The command this replica would be started with. Exposed so a deployment
  /// can print what it is about to run before letting it run.
  [[nodiscard]] std::string commandFor(const Replica &replica) const;

  /// Whether a replica's process is still alive, reaping it if it is not.
  /// A replica whose process died has failed, however healthy it looked.
  [[nodiscard]] bool running(const std::string &replica);

  private:

  std::string                _template;
  std::map<std::string, int> _pids;
};

/**
 * Asks a replica's own `/health`, which is 503 until its engine loop is live.
 *
 * A replica names its node, and a node name is not an address -- inside a
 * container it is whatever the deployment chose to call the machine. Give the
 * probe the map and it resolves the name to the node's host; without one it
 * treats the name as an address, which is right only where they coincide.
 */
class HttpHealthProbe final : public HealthProbe
{
  public:

  explicit HttpHealthProbe(const StateMap *map = nullptr, double connectTimeoutSeconds = 2.0)
    : _map(map),
      _timeout(connectTimeoutSeconds)
  {}

  [[nodiscard]] Readiness check(const Replica &replica) override;

  /// Where this replica would be probed, so a deployment can see what it is
  /// about to call before it fails to reach it.
  [[nodiscard]] std::string addressOf(const Replica &replica) const;

  private:

  const StateMap *_map;
  double          _timeout;
};

/**
 * A probe that is only satisfied when the process is alive *and* serving.
 *
 * Health alone cannot tell a replica that is still loading from one whose
 * process has already exited: both refuse the connection. Pairing them turns
 * the second case into a failed attempt instead of a wait that runs to timeout.
 */
class LiveAndReadyProbe final : public HealthProbe
{
  public:

  LiveAndReadyProbe(ProcessStarter &starter, HealthProbe &health)
    : _starter(starter),
      _health(health)
  {}

  /// A replica whose process has gone is Failed, however the endpoint looks:
  /// that is the case a bare health check cannot distinguish from loading.
  [[nodiscard]] Readiness check(const Replica &replica) override
  {
    if (!_starter.running(replica.name)) { return Readiness::Failed; }
    return _health.check(replica);
  }

  private:

  ProcessStarter &_starter;
  HealthProbe    &_health;
};

} // namespace serving::infra
