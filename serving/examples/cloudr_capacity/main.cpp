/**
 * Scaling onto nodes that did not exist, against the CloudR emulator.
 *
 * The deck's second source of capacity -- *beyond the local pool, ask the
 * provider for additional nodes* -- needs more than one machine to be real.
 * CloudR emulates instances on one, so the control plane can be shown taking
 * delivery of a node it did not have, placing a replica on it, and giving it
 * back.
 *
 * Run it the way CloudR's own examples run:
 *
 *     mpirun --allow-run-as-root --oversubscribe -np 4 ./cloudr-capacity
 *
 * One instance is the root and does the scaling; the others are the pool the
 * emulator hands out.
 */

#include <cstdio>
#include <string>

#include <hicr/backends/cloudr/instanceManager.hpp>
#include <hicr/backends/hwloc/topologyManager.hpp>
#include <hicr/backends/mpi/communicationManager.hpp>
#include <hicr/backends/mpi/instanceManager.hpp>
#include <hicr/backends/mpi/memoryManager.hpp>
#include <hicr/backends/pthreads/computeManager.hpp>
#include <hicr/frontends/RPCEngine/RPCEngine.hpp>

#include <serving/infra/cloudr_capacity.hpp>
#include <serving/infra/scaling.hpp>

namespace
{

using namespace serving::infra;

void report(const char *label, const StateMap &map)
{
  std::printf("\n%s: version %llu, %zu node(s), %zu replica(s)\n", label, static_cast<unsigned long long>(map.version()), map.nodes().size(), map.replicas().size());
  for (const Node &node : map.nodes()) { std::printf("  node %-12s host=%-12s cards=%zu\n", node.name.c_str(), node.host.c_str(), node.devices.size()); }
  for (const Replica &replica : map.replicas())
  {
    std::printf("  replica %-16s partition=%d node=%-12s instance=%llu phase=%s\n",
                replica.name.c_str(),
                replica.partition,
                replica.node.c_str(),
                static_cast<unsigned long long>(replica.instance),
                phaseName(replica.phase));
  }
  const auto problems = validate(map);
  std::printf("  validate: %s\n", problems.empty() ? "coherent" : problems.front().c_str());
}

/// Load high enough that the policy always wants another replica.
LoadSnapshot saturated(const StateMap &map)
{
  LoadSnapshot load;
  for (const Replica &replica : map.replicas())
  {
    if (replica.phase == Phase::Active) { load.replicas.push_back(ReplicaLoad{replica.name, 99, 0, 0}); }
  }
  return load;
}

void scaleOntoTheCloud(HiCR::backend::cloudr::InstanceManager &instances)
{
  // A deployment with one replica on the local machine, and that machine full.
  StateMap map;
  map.addNode(Node{"local", "127.0.0.1", {0}});

  Replica first;
  first.name    = "p0a";
  first.node    = "local";
  first.devices = {0};
  first.port    = 8000;
  first.phase   = Phase::Active;
  map.addReplica(first);
  report("before", map);

  CloudrCapacity cloud(instances, /*devicesPerNode=*/1, "cloud");
  cloud.setLimit(2);

  ScalingConfig config;
  config.maxReplicas       = 4;
  config.devicesPerReplica = 1;

  for (int step = 1; step <= 3; ++step)
  {
    const auto decisions = decideScaling(map, saturated(map), config);
    if (decisions.empty())
    {
      std::printf("\nstep %d: the policy is satisfied\n", step);
      break;
    }
    std::printf("\nstep %d: %s\n", step, decisions.front().reason.c_str());

    const auto outcomes = applyScaling(map, cloud, decisions, config);
    if (!outcomes.front().applied)
    {
      // The ordinary answer once the emulator's pool is spent.
      std::printf("  declined: %s\n", outcomes.front().declined.c_str());
      break;
    }
    const Replica *added = map.replica(outcomes.front().replica);
    std::printf("  took delivery of node '%s' for replica '%s' (HiCR instance %llu)\n", added->node.c_str(), added->name.c_str(), static_cast<unsigned long long>(added->instance));
    // A new replica is pending until something starts it; there is no launcher
    // for an emulated node, so promote it by hand to keep the policy moving.
    map.setPhase(added->name, Phase::Active);
  }
  report("after", map);

  // Giving the machines back: an emulated node still costs a real process.
  for (const Replica &replica : map.replicas())
  {
    if (replica.node != "local") { cloud.release(replica.name); }
  }
  std::printf("\nreleased; instances still held: %d\n", cloud.outstanding());
}

} // namespace

int main(int argc, char *argv[])
{
  auto instanceManager      = HiCR::backend::mpi::InstanceManager::createDefault(&argc, &argv);
  auto communicationManager = HiCR::backend::mpi::CommunicationManager(MPI_COMM_WORLD);
  auto memoryManager        = HiCR::backend::mpi::MemoryManager();
  auto computeManager       = HiCR::backend::pthreads::ComputeManager();

  hwloc_topology_t hwlocTopology;
  hwloc_topology_init(&hwlocTopology);
  auto hwlocTopologyManager = HiCR::backend::hwloc::TopologyManager(&hwlocTopology);

  const auto &topology           = hwlocTopologyManager.queryTopology();
  const auto &firstDevice        = *topology.getDevices().begin();
  const auto &RPCMemorySpace     = *firstDevice->getMemorySpaceList().begin();
  const auto &RPCComputeResource = *firstDevice->getComputeResourceList().begin();

  HiCR::frontend::RPCEngine rpcEngine(communicationManager, *instanceManager, memoryManager, computeManager, RPCMemorySpace, RPCComputeResource);
  rpcEngine.initialize();

  // Only the root scales; the others are the pool it is handed out of.
  HiCR::backend::cloudr::InstanceManager cloudr(&rpcEngine, &computeManager, topology, [&]() {
    if (cloudr.getCurrentInstance()->isRootInstance()) { scaleOntoTheCloud(cloudr); }
  });

  cloudr.initialize();
  cloudr.finalize();
  instanceManager->finalize();
  hwloc_topology_destroy(hwlocTopology);
  return 0;
}
