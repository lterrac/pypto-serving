#include <serving/infra/cloudr_capacity.hpp>

#include <algorithm>

namespace serving::infra
{

CloudrCapacity::CloudrCapacity(HiCR::backend::cloudr::InstanceManager &instances, int devicesPerNode, std::string namePrefix)
  : _instances(instances),
    _devicesPerNode(std::max(1, devicesPerNode)),
    _prefix(std::move(namePrefix))
{}

std::optional<Grant> CloudrCapacity::acquire(const CapacityRequest &request)
{
  if (static_cast<int>(_held.size()) >= _limit) { return std::nullopt; }
  // A node that cannot hold the replica is not worth creating.
  if (request.devices > _devicesPerNode) { return std::nullopt; }

  std::shared_ptr<HiCR::Instance> instance;
  try
  {
    // An empty template: the emulator clones the requesting topology, which is
    // what "a machine like this one" means here.
    const auto blueprint = _instances.createInstanceTemplate();
    instance             = _instances.createInstance(*blueprint);
  }
  catch (const std::exception &)
  {
    // The emulator refusing is the ordinary "no capacity" answer, not an error
    // to propagate: a policy that asked too eagerly should back off.
    return std::nullopt;
  }
  if (instance == nullptr) { return std::nullopt; }

  const std::string nodeName = _prefix + "-" + std::to_string(_sequence++);

  Node node;
  node.name = nodeName;
  // The emulated instance is addressed by id, not by hostname; the instance id
  // on the grant is what the control plane actually routes control traffic to.
  node.host = nodeName;
  node.devices.resize(static_cast<size_t>(_devicesPerNode));
  for (int i = 0; i < _devicesPerNode; ++i) { node.devices[static_cast<size_t>(i)] = i; }

  Grant grant;
  grant.replica = request.name.empty() ? nodeName + "-r0" : request.name;
  grant.node    = nodeName;
  grant.devices.assign(node.devices.begin(), node.devices.begin() + std::max(1, request.devices));
  grant.port     = request.port;
  grant.instance = instance->getId();
  grant.newNode  = node;

  _held.emplace(grant.replica, instance);
  return grant;
}

void CloudrCapacity::release(const std::string &replica)
{
  const auto it = _held.find(replica);
  if (it == _held.end()) { return; }

  try
  {
    _instances.terminateInstance(it->second);
  }
  catch (const std::exception &)
  {
    // Already gone, or the emulator is shutting down. Either way there is
    // nothing left to give back, and forgetting it is the right end state.
  }
  _held.erase(it);
}

int CloudrCapacity::available(const CapacityRequest &request) const
{
  if (request.devices > _devicesPerNode) { return 0; }
  return std::max(0, _limit - static_cast<int>(_held.size()));
}

} // namespace serving::infra
