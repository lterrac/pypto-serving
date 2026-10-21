#pragma once

/**
 * The infrastructure state map: what runs where, holding which NPUs, in which
 * state (`scaling-pypto-serving.pptx`, M4).
 *
 * One versioned document, and the only source of truth routing, scaling and
 * monitoring read. It is deliberately plain data with no HiCR in it, the way
 * `coordinator/module.hpp` is: the platform hosts and distributes it, but a
 * test, the router and a single-node deployment can all hold one without a
 * control plane underneath.
 *
 * Declarative, per the deck: a record carries both the phase it is observed in
 * and the phase it is wanted in, and `reconcile()` says what would close the
 * gap. Nothing here executes those actions -- that is the launcher's job, and
 * there is no launcher yet.
 *
 * What a device is matters here. Serving has treated a device as an integer
 * with no way to say which host owns it; a `Device` is that integer plus the
 * node it belongs to, which is what makes "does this partition fit that node"
 * a question the map can answer.
 */

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace serving::infra
{

using PartitionId = int;

/// A physical machine and the accelerators it owns.
struct Node
{
  /// Stable identity, not necessarily a hostname: it is what a replica names.
  std::string name;
  /// Where replicas on this node are reachable.
  std::string host;
  /// Physical device ids this node owns, as the driver numbers them.
  std::vector<int> devices;

  [[nodiscard]] bool owns(int device) const;
  bool               operator==(const Node &) const = default;
};

/// Where a replica is in its life. `Pending` is a replica the deployment wants
/// but nothing has started yet; `Dead` is one that was lost rather than drained.
enum class Phase
{
  Pending,
  Active,
  /**
   * Up, loaded and healthy, deliberately not routed to.
   *
   * The deck's warm capacity: *spare worker processes start at launch and sit
   * idle; activating one is a single message*. The message is one edit to this
   * phase -- the process is already there, so activation costs no model load,
   * which is the entire reason the spare exists.
   *
   * It holds its devices while it waits, because it really is holding them.
   */
  Reserved,
  Draining,
  Dead,
};

[[nodiscard]] const char *phaseName(Phase phase);
/// Parses the names `phaseName` produces; nullopt for anything else.
[[nodiscard]] std::optional<Phase> parsePhase(const std::string &text);

/// Whether a replica in this phase is holding its devices.
[[nodiscard]] bool holdsDevices(Phase phase);

/// One replica, as the deployment describes it and as it is observed.
struct Replica
{
  std::string name;
  PartitionId partition = 0;
  /// The `Node::name` this replica runs on.
  std::string node;
  /// Which of that node's devices it holds. Empty until it is placed.
  std::vector<int> devices;
  int              port   = 0;
  std::string      scheme = "http";
  /// HiCR instance id once the platform runs it, 0 while it does not.
  uint64_t instance = 0;

  /// What it is doing now, and what the deployment wants it to be doing.
  Phase phase        = Phase::Pending;
  Phase desiredPhase = Phase::Active;

  bool operator==(const Replica &) const = default;
};

/// A control-plane channel between two replicas, by name.
struct Channel
{
  std::string from;
  std::string to;
  bool        operator==(const Channel &) const = default;
};

/// What would have to happen for a replica's phase to match the one wanted.
enum class Action
{
  Start,
  Drain,
  Stop,
};

[[nodiscard]] const char *actionName(Action action);

struct PendingAction
{
  std::string replica;
  Action      action;
  bool        operator==(const PendingAction &) const = default;
};

/**
 * The map.
 *
 * Every mutation bumps `version()`, so a reader that cached a decision can tell
 * the ground moved under it without diffing the document.
 */
class StateMap
{
  public:

  /// Bumped by every mutation, including one that changes nothing observable.
  [[nodiscard]] uint64_t version() const { return _version; }

  [[nodiscard]] const std::vector<Node>    &nodes() const { return _nodes; }
  [[nodiscard]] const std::vector<Replica> &replicas() const { return _replicas; }
  [[nodiscard]] const std::vector<Channel> &channels() const { return _channels; }

  [[nodiscard]] const Node    *node(const std::string &name) const;
  [[nodiscard]] const Replica *replica(const std::string &name) const;

  [[nodiscard]] std::vector<const Replica *> replicasOn(const std::string &nodeName) const;
  [[nodiscard]] std::vector<const Replica *> replicasIn(PartitionId partition) const;
  /// Partitions that have at least one replica, ascending.
  [[nodiscard]] std::vector<PartitionId> partitions() const;

  /// Devices on `nodeName` held by a replica that has not released them.
  [[nodiscard]] std::vector<int> devicesInUse(const std::string &nodeName) const;
  /// Devices the node owns that no replica is holding.
  [[nodiscard]] std::vector<int> devicesFree(const std::string &nodeName) const;

  /// Replaces a node of the same name.
  void addNode(Node node);
  /// Replaces a replica of the same name.
  void addReplica(Replica replica);
  void addChannel(Channel channel);

  /// False if there is no such replica; the version does not move then.
  bool setPhase(const std::string &name, Phase phase);
  bool setDesiredPhase(const std::string &name, Phase phase);
  bool place(const std::string &name, const std::string &nodeName, std::vector<int> devices);
  bool removeReplica(const std::string &name);

  /// Put the version back where a loaded document left it.
  ///
  /// Building a map replica by replica bumps the version once per edit, so a
  /// freshly loaded document would come back with a version counting the load
  /// rather than the edits it records -- and a reader that cached a version
  /// would draw the wrong conclusion from it. Only a loader should call this.
  void restoreVersion(uint64_t version) { _version = version; }

  private:

  void bump() { _version += 1; }

  uint64_t             _version = 0;
  std::vector<Node>    _nodes;
  std::vector<Replica> _replicas;
  std::vector<Channel> _channels;
};

/**
 * Whether a node or replica name is safe to carry through the control plane.
 *
 * Names are not decoration: they are interpolated into pid-file paths that a
 * shell on another node evaluates (`infra/remote_starter.cpp`), into process
 * command lines, and into routing keys. A document arrives over RPC from a
 * deployer this process did not write, so the name is checked once, here, at
 * every point one enters the map rather than at each place one is used.
 *
 * Letters, digits, dot, underscore and dash, 1..64 characters. Deliberately
 * narrower than a hostname: anything a shell, a URL or a JSON key treats
 * specially is out.
 */
[[nodiscard]] bool isUsableName(const std::string &name);

/**
 * Everything wrong with the deployment the map describes, one message per
 * problem, each naming its subject.
 *
 * Empty means the deployment is coherent: every replica sits on a node that
 * exists and owns the devices it claims, no two live replicas claim the same
 * device, names and endpoints are unique, partitions are contiguous from zero,
 * and channels join replicas that exist. This is slide 6's "does this partition
 * fit that node", answered before anything is started rather than after.
 */
[[nodiscard]] std::vector<std::string> validate(const StateMap &map);

/// The actions that would bring every replica to the phase it is wanted in, in
/// map order. A replica already in its desired phase contributes nothing.
[[nodiscard]] std::vector<PendingAction> reconcile(const StateMap &map);

} // namespace serving::infra
