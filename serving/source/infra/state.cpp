#include <serving/infra/state.hpp>

#include <algorithm>
#include <map>
#include <set>

namespace serving::infra
{

namespace
{

template <typename T>
const T *findByName(const std::vector<T> &items, const std::string &name)
{
  const auto it = std::find_if(items.begin(), items.end(), [&](const T &item) { return item.name == name; });
  return it == items.end() ? nullptr : &*it;
}

} // namespace

bool Node::owns(int device) const { return std::find(devices.begin(), devices.end(), device) != devices.end(); }

const char *phaseName(Phase phase)
{
  switch (phase)
  {
  case Phase::Pending: return "pending";
  case Phase::Active: return "active";
  case Phase::Reserved: return "reserved";
  case Phase::Draining: return "draining";
  case Phase::Dead: return "dead";
  }
  return "pending";
}

std::optional<Phase> parsePhase(const std::string &text)
{
  if (text == "pending") { return Phase::Pending; }
  if (text == "active") { return Phase::Active; }
  if (text == "reserved") { return Phase::Reserved; }
  if (text == "draining") { return Phase::Draining; }
  if (text == "dead") { return Phase::Dead; }
  return std::nullopt;
}

// A draining replica is still serving what it has and a reserved one is loaded
// and waiting, so both are holding their devices; only pending and dead have
// released them.
bool holdsDevices(Phase phase) { return phase == Phase::Active || phase == Phase::Draining || phase == Phase::Reserved; }

const char *actionName(Action action)
{
  switch (action)
  {
  case Action::Start: return "start";
  case Action::Drain: return "drain";
  case Action::Stop: return "stop";
  }
  return "start";
}

// ---------------------------------------------------------------------------
// StateMap
// ---------------------------------------------------------------------------

const Node *StateMap::node(const std::string &name) const { return findByName(_nodes, name); }

const Replica *StateMap::replica(const std::string &name) const { return findByName(_replicas, name); }

std::vector<const Replica *> StateMap::replicasOn(const std::string &nodeName) const
{
  std::vector<const Replica *> out;
  for (const Replica &r : _replicas)
  {
    if (r.node == nodeName) { out.push_back(&r); }
  }
  return out;
}

std::vector<const Replica *> StateMap::replicasIn(PartitionId partition) const
{
  std::vector<const Replica *> out;
  for (const Replica &r : _replicas)
  {
    if (r.partition == partition) { out.push_back(&r); }
  }
  return out;
}

std::vector<PartitionId> StateMap::partitions() const
{
  std::set<PartitionId> seen;
  for (const Replica &r : _replicas) { seen.insert(r.partition); }
  return {seen.begin(), seen.end()};
}

std::vector<int> StateMap::devicesInUse(const std::string &nodeName) const
{
  std::set<int> used;
  for (const Replica &r : _replicas)
  {
    if (r.node != nodeName || !holdsDevices(r.phase)) { continue; }
    used.insert(r.devices.begin(), r.devices.end());
  }
  return {used.begin(), used.end()};
}

std::vector<int> StateMap::devicesFree(const std::string &nodeName) const
{
  const Node *owner = node(nodeName);
  if (owner == nullptr) { return {}; }
  const auto       used = devicesInUse(nodeName);
  std::vector<int> free;
  for (const int device : owner->devices)
  {
    if (std::find(used.begin(), used.end(), device) == used.end()) { free.push_back(device); }
  }
  return free;
}

void StateMap::addNode(Node value)
{
  const auto it = std::find_if(_nodes.begin(), _nodes.end(), [&](const Node &n) { return n.name == value.name; });
  if (it == _nodes.end()) { _nodes.push_back(std::move(value)); }
  else { *it = std::move(value); }
  bump();
}

void StateMap::addReplica(Replica value)
{
  const auto it = std::find_if(_replicas.begin(), _replicas.end(), [&](const Replica &r) { return r.name == value.name; });
  if (it == _replicas.end()) { _replicas.push_back(std::move(value)); }
  else { *it = std::move(value); }
  bump();
}

void StateMap::addChannel(Channel value)
{
  if (std::find(_channels.begin(), _channels.end(), value) == _channels.end()) { _channels.push_back(std::move(value)); }
  bump();
}

bool StateMap::setPhase(const std::string &name, Phase phase)
{
  const auto it = std::find_if(_replicas.begin(), _replicas.end(), [&](const Replica &r) { return r.name == name; });
  if (it == _replicas.end()) { return false; }
  it->phase = phase;
  bump();
  return true;
}

bool StateMap::setDesiredPhase(const std::string &name, Phase phase)
{
  const auto it = std::find_if(_replicas.begin(), _replicas.end(), [&](const Replica &r) { return r.name == name; });
  if (it == _replicas.end()) { return false; }
  it->desiredPhase = phase;
  bump();
  return true;
}

bool StateMap::place(const std::string &name, const std::string &nodeName, std::vector<int> devices)
{
  const auto it = std::find_if(_replicas.begin(), _replicas.end(), [&](const Replica &r) { return r.name == name; });
  if (it == _replicas.end()) { return false; }
  it->node    = nodeName;
  it->devices = std::move(devices);
  bump();
  return true;
}

bool StateMap::removeReplica(const std::string &name)
{
  const auto it = std::find_if(_replicas.begin(), _replicas.end(), [&](const Replica &r) { return r.name == name; });
  if (it == _replicas.end()) { return false; }
  _replicas.erase(it);
  // A channel to a replica that is gone would be a dangling edge in the map.
  std::erase_if(_channels, [&](const Channel &c) { return c.from == name || c.to == name; });
  bump();
  return true;
}

// ---------------------------------------------------------------------------
// validate
// ---------------------------------------------------------------------------

std::vector<std::string> validate(const StateMap &map)
{
  std::vector<std::string> problems;

  std::set<std::string> nodeNames;
  for (const Node &n : map.nodes())
  {
    if (n.name.empty()) { problems.emplace_back("node: name is empty"); }
    else if (!nodeNames.insert(n.name).second) { problems.emplace_back("node '" + n.name + "': declared twice"); }

    std::set<int> devices;
    for (const int device : n.devices)
    {
      if (!devices.insert(device).second) { problems.emplace_back("node '" + n.name + "': device " + std::to_string(device) + " listed twice"); }
    }
  }

  std::set<std::string> replicaNames;
  std::set<std::string> endpoints;
  // Device to the live replicas claiming it, per node, so a clash names both.
  std::map<std::pair<std::string, int>, std::vector<std::string>> claims;

  for (const Replica &r : map.replicas())
  {
    const std::string who = "replica '" + r.name + "'";
    if (r.name.empty()) { problems.emplace_back("replica: name is empty"); }
    else if (!replicaNames.insert(r.name).second) { problems.emplace_back(who + ": declared twice"); }

    if (r.partition < 0) { problems.emplace_back(who + ": partition is negative"); }

    const Node *owner = map.node(r.node);
    if (r.node.empty()) { problems.emplace_back(who + ": names no node"); }
    else if (owner == nullptr) { problems.emplace_back(who + ": node '" + r.node + "' is not in the map"); }

    for (const int device : r.devices)
    {
      if (owner != nullptr && !owner->owns(device)) { problems.emplace_back(who + ": device " + std::to_string(device) + " is not on node '" + r.node + "'"); }
      // Only a replica that still holds its devices can clash with another.
      if (holdsDevices(r.phase)) { claims[{r.node, device}].push_back(r.name); }
    }

    if (r.port != 0 && !r.node.empty())
    {
      const std::string endpoint = r.node + ":" + std::to_string(r.port);
      if (!endpoints.insert(endpoint).second) { problems.emplace_back(who + ": endpoint " + endpoint + " is already taken"); }
    }
  }

  for (const auto &[where, holders] : claims)
  {
    if (holders.size() < 2) { continue; }
    std::string names;
    for (const std::string &holder : holders) { names += (names.empty() ? "" : ", ") + holder; }
    problems.emplace_back("node '" + where.first + "' device " + std::to_string(where.second) + ": claimed by " + names);
  }

  // A pipeline with a hole cannot serve: a request has no replica for that hop.
  const auto partitions = map.partitions();
  for (size_t i = 0; i < partitions.size(); ++i)
  {
    if (partitions[i] != static_cast<PartitionId>(i))
    {
      problems.emplace_back("partitions must run from 0 without gaps; found " + std::to_string(partitions[i]) + " where " + std::to_string(i) + " was expected");
      break;
    }
  }

  for (const Channel &c : map.channels())
  {
    if (map.replica(c.from) == nullptr) { problems.emplace_back("channel from '" + c.from + "': no such replica"); }
    if (map.replica(c.to) == nullptr) { problems.emplace_back("channel to '" + c.to + "': no such replica"); }
  }

  return problems;
}

// ---------------------------------------------------------------------------
// reconcile
// ---------------------------------------------------------------------------

std::vector<PendingAction> reconcile(const StateMap &map)
{
  std::vector<PendingAction> actions;
  for (const Replica &r : map.replicas())
  {
    if (r.phase == r.desiredPhase) { continue; }
    switch (r.desiredPhase)
    {
    case Phase::Reserved:
    case Phase::Active:
      // Also the path from reserved to active: the supervisor finds the
      // process already up and only has to change the phase, which is what
      // makes activating a spare a message rather than a model load.
      actions.push_back({r.name, Action::Start});
      break;
    case Phase::Draining:
      if (r.phase == Phase::Active) { actions.push_back({r.name, Action::Drain}); }
      break;
    case Phase::Dead:
    case Phase::Pending: actions.push_back({r.name, Action::Stop}); break;
    }
  }
  return actions;
}

} // namespace serving::infra
