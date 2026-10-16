#pragma once

/**
 * Starting a replica on a node that is not this one.
 *
 * `ProcessStarter` spawns a command here. That is all a single-machine
 * deployment needs, and it is why the state map's nodes have never meant
 * anything to the launcher: every replica was started locally whatever node it
 * claimed to be on. A deployment described across machines has to be able to
 * act on that claim.
 *
 * This wraps the local starter and reaches other nodes through a command
 * template, so the mechanism is a property of the deployment rather than
 * something compiled in. The default is ssh; a cluster with its own launcher
 * substitutes it without touching this code.
 *
 * **A remote replica needs more bookkeeping than a local one.** The pid here
 * belongs to the reach command, not to the replica, so killing it leaves the
 * replica running and holding its card on the far side -- the same failure
 * `ProcessStarter` fixed locally by signalling the process group, one machine
 * further away. The remote wrapper therefore records its process group in a
 * file on that node, and stopping signals the group through a second reach.
 */

#include <map>
#include <memory>
#include <string>

#include <serving/infra/process_starter.hpp>
#include <serving/infra/state.hpp>

namespace serving::infra
{

struct RemoteLaunch
{
  /// How to run a command on another node. `{host}` is that node's address and
  /// `{command}` the already-quoted shell command to run there.
  std::string reach = "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new {host} {command}";

  /// Where the remote wrapper records the replica's process group. It must be
  /// writable on every node the deployment reaches.
  std::string stateDirectory = "/tmp";

  /// Seconds the remote stop waits between SIGTERM and SIGKILL. A model
  /// mid-load does not always die promptly, and a replica still alive on the
  /// far side keeps its card out of the pool for good.
  int graceSeconds = 10;
};

class RemoteStarter final : public ReplicaStarter
{
  public:

  /// `localNode` is this process's entry in the map: replicas on it are
  /// started directly, everything else is reached. Empty means every replica
  /// is remote, which is what a deployer that runs no replicas itself wants.
  ///
  /// The map is borrowed and must outlive this, like `LocalCapacity`'s.
  RemoteStarter(const StateMap &map, std::string localNode, std::string commandTemplate, RemoteLaunch launch = {});

  [[nodiscard]] bool        start(const Replica &replica) override;
  void                      stop(const Replica &replica) override;
  [[nodiscard]] const char *kind() const override { return "remote"; }

  /// True when this replica would be started here rather than reached.
  [[nodiscard]] bool isLocal(const Replica &replica) const;

  /// Exactly what would be run for this replica, reach command and all, so a
  /// deployment can print it before letting it run.
  [[nodiscard]] std::string commandFor(const Replica &replica) const;

  /// Whether the replica's process is still alive. For a remote one this is
  /// the reach command's liveness: ssh exits when the command it ran does, so
  /// it answers the question without a round trip per tick.
  [[nodiscard]] bool running(const std::string &replica);

  private:

  /// The node's address, or the node name when the map does not have it --
  /// which is right only where the two coincide, and wrong loudly rather than
  /// silently when they do not.
  [[nodiscard]] std::string hostOf(const Replica &replica) const;

  [[nodiscard]] std::string pidFileFor(const Replica &replica) const;

  /// The command as it is run on the far side: record the group, then become
  /// the replica, so the file holds the replica's own group and not a wrapper's.
  [[nodiscard]] std::string remoteCommand(const Replica &replica) const;

  const StateMap &_map;
  std::string     _localNode;
  std::string     _template;
  RemoteLaunch    _launch;

  /// One per replica, built with the fully composed command. ProcessStarter
  /// takes its template at construction, and a reach command differs per
  /// replica, so there is one of them per replica rather than one shared.
  std::map<std::string, std::unique_ptr<ProcessStarter>> _starters;
};

} // namespace serving::infra
