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
    grant.replica = request.name.empty() ? autoName(request.partition, static_cast<int>(_outstanding.size())) : request.name;
    grant.node    = node.name;
    grant.devices.assign(free.begin(), free.begin() + wanted);
    grant.port = request.port;
    _outstanding.push_back(grant);
    return grant;
  }
  return std::nullopt;
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
