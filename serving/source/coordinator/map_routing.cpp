#include <serving/coordinator/map_routing.hpp>

namespace serving::coordinator
{

std::vector<ReplicaBinding> bindingsFrom(const infra::StateMap &map)
{
  std::vector<ReplicaBinding> bindings;
  bindings.reserve(map.replicas().size());

  for (const infra::Replica &replica : map.replicas())
  {
    const infra::Node *node = map.node(replica.node);
    // No node, no address: a replica the map cannot place has no endpoint to
    // route to, and `validate()` already reports it as a broken deployment.
    if (node == nullptr) { continue; }

    ReplicaBinding binding;
    binding.spec.name      = replica.name;
    binding.spec.partition = replica.partition;
    binding.spec.host      = node->host;
    binding.spec.port      = replica.port;
    binding.spec.scheme    = replica.scheme;
    binding.instanceId     = replica.instance;
    bindings.push_back(std::move(binding));
  }
  return bindings;
}

} // namespace serving::coordinator
