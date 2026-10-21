#include <serving/infra/capacity.hpp>

#include <algorithm>
#include <set>

namespace serving::infra
{

namespace
{

/// A name for a replica the caller did not name, stable enough to be unique
/// within one provider's lifetime.
std::string autoName(PartitionId partition, int sequence) { return "p" + std::to_string(partition) + "-auto-" + std::to_string(sequence); }

} // namespace

// ---------------------------------------------------------------------------
// LocalCapacity
// ---------------------------------------------------------------------------

std::optional<Grant> LocalCapacity::acquire(const CapacityRequest &request)
{
  const int wanted = std::max(1, request.devices);

  // A grant stops being outstanding the moment the map knows about the replica
  // it was issued for: from then on the map's own device accounting covers it,
  // and keeping the grant would exclude the card twice. Pruning here is what
  // makes the list self-healing -- nothing ever called release(), so every card
  // a scale-up was ever granted stayed withheld for the life of the process,
  // even after its replica had drained or died.
  std::erase_if(_outstanding, [&](const Grant &grant) { return _map.replica(grant.replica) != nullptr; });

  for (const Node &node : _map.nodes())
  {
    std::vector<int> free = _map.devicesFree(node.name);

    // The map does not know about grants the caller has not placed yet, so two
    // acquires in a row would otherwise be handed the same card.
    for (const Grant &outstanding : _outstanding)
    {
      if (outstanding.node != node.name) { continue; }
      for (const int device : outstanding.devices) { std::erase(free, device); }
    }

    if (static_cast<int>(free.size()) < wanted) { continue; }

    Grant grant;
    grant.replica = request.name.empty() ? autoName(request.partition, nextAutoIndex(request.partition)) : request.name;
    grant.node    = node.name;
    grant.devices.assign(free.begin(), free.begin() + wanted);
    grant.port = request.port;
    _outstanding.push_back(grant);
    return grant;
  }
  return std::nullopt;
}

int LocalCapacity::nextAutoIndex(PartitionId partition) const
{
  // Counted from what exists, not from how many grants this process has issued.
  // The old counter restarted with the process while the document it wrote into
  // persisted, so the first scale-up after a restart generated a name that was
  // already taken -- and addReplica replaces on a matching name, so it silently
  // overwrote a live replica's record and orphaned its process.
  for (int index = 0;; ++index)
  {
    const std::string candidate = autoName(partition, index);
    if (_map.replica(candidate) != nullptr) { continue; }
    const bool granted = std::any_of(_outstanding.begin(), _outstanding.end(), [&](const Grant &grant) { return grant.replica == candidate; });
    if (!granted) { return index; }
  }
}

void LocalCapacity::release(const std::string &replica)
{
  std::erase_if(_outstanding, [&](const Grant &grant) { return grant.replica == replica; });
}

int LocalCapacity::available(const CapacityRequest &request) const
{
  const int wanted = std::max(1, request.devices);

  int total = 0;
  for (const Node &node : _map.nodes())
  {
    std::vector<int> free = _map.devicesFree(node.name);
    for (const Grant &outstanding : _outstanding)
    {
      if (outstanding.node != node.name) { continue; }
      for (const int device : outstanding.devices) { std::erase(free, device); }
    }
    total += static_cast<int>(free.size()) / wanted;
  }
  return total;
}

} // namespace serving::infra
