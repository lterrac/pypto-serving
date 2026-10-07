/**
 * One document, on the node that owns it.
 *
 * The deployer -- the instance that started the application, known to everyone
 * -- holds the deployment document and answers an RPC with a copy. The other
 * instances keep no map of their own and replicate nothing; they ask.
 *
 *     mpirun --allow-run-as-root --oversubscribe -np 3 ./document-rpc
 */

#include <cstdio>

#include <hicr/backends/cloudr/instanceManager.hpp>
#include <hicr/backends/hwloc/topologyManager.hpp>
#include <hicr/backends/mpi/communicationManager.hpp>
#include <hicr/backends/mpi/instanceManager.hpp>
#include <hicr/backends/mpi/memoryManager.hpp>
#include <hicr/backends/pthreads/computeManager.hpp>
#include <hicr/frontends/RPCEngine/RPCEngine.hpp>

#include <serving/infra/document_rpc.hpp>

namespace
{

using namespace serving::infra;

/// What the deployer knows: a three-partition pipeline across two machines.
StateMap deployment()
{
  StateMap map;
  map.addNode(Node{"hg-atlas-01", "192.168.150.11", {0, 1, 2, 3}});
  map.addNode(Node{"hg-atlas-02", "192.168.150.12", {0, 1}});

  const char *nodes[] = {"hg-atlas-01", "hg-atlas-01", "hg-atlas-02"};
  for (int partition = 0; partition < 3; ++partition)
  {
    Replica replica;
    replica.name      = "p" + std::to_string(partition);
    replica.partition = partition;
    replica.node      = nodes[partition];
    replica.devices   = {partition % 2};
    replica.port      = 8000 + partition;
    replica.phase     = Phase::Active;
    map.addReplica(replica);
  }
  return map;
}

void describe(const char *who, const StateMap &map)
{
  std::printf("[%s] version %llu, %zu node(s), %zu replica(s)\n", who, static_cast<unsigned long long>(map.version()), map.nodes().size(), map.replicas().size());
  for (const Replica &replica : map.replicas())
  {
    std::printf(
      "[%s]   partition %d -> %s on %s card %d\n", who, replica.partition, replica.name.c_str(), replica.node.c_str(), replica.devices.empty() ? -1 : replica.devices.front());
  }
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

  const bool isDeployer = instanceManager->getCurrentInstance()->getId() == instanceManager->getRootInstanceId();

  // Built on every instance, because only the deployer's is ever called and
  // the others cost nothing: who the deployer is, is a deployment decision.
  StateMap       owned = isDeployer ? deployment() : StateMap{};
  DocumentServer server(rpcEngine, computeManager, owned);

  if (isDeployer)
  {
    describe("deployer", owned);
    std::printf("[deployer] serving %s; waiting for the others\n", DOCUMENT_RPC);
    // One listen per asker.
    for (size_t i = 1; i < instanceManager->getInstances().size(); ++i) { rpcEngine.listen(); }
    std::printf("[deployer] handed out %d copies\n", server.served());
  }
  else
  {
    const auto id = instanceManager->getCurrentInstance()->getId();
    std::printf("[coordinator %lu] my own map is empty; asking the deployer\n", id);
    try
    {
      const StateMap fetched = fetchDocument(rpcEngine, instanceManager->getRootInstanceId());
      describe("coordinator", fetched);
      std::printf("[coordinator %lu] partitions it can now plan over: %zu\n", id, fetched.partitions().size());
    }
    catch (const DocumentError &e)
    {
      std::printf("[coordinator %lu] could not read the deployment: %s\n", id, e.what());
    }
  }

  instanceManager->finalize();
  hwloc_topology_destroy(hwlocTopology);
  return 0;
}
