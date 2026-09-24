/**
 * Drives the two routing strategies from a configuration file, printing the
 * path per turn and what `rules` would publish.
 *
 *   routing-modes --config FILE
 */

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <serving/coordinator/config_file.hpp>
#include <serving/router/strategy.hpp>

namespace
{

using namespace serving::router;

/// Stands in for the platform's channels: reports rather than sends.
class PrintingTransport : public RuleTransport
{
  public:

  void publish(const ReplicaSpec &holder, const std::string &sessionId, const ReplicaSpec &nextHop, uint64_t generation) override
  {
    std::printf("      [rule gen=%lu] %s: session %s -> forward to %s\n", static_cast<unsigned long>(generation), holder.name.c_str(), sessionId.c_str(), nextHop.name.c_str());
  }

  void revoke(const std::string &sessionId, uint64_t generation) override
  {
    std::printf("      [revoke gen=%lu] session %s\n", static_cast<unsigned long>(generation), sessionId.c_str());
  }
};

void printPath(const char *label, const RoutingPath &path)
{
  std::printf("      %s: ", label);
  for (size_t i = 0; i < path.hops.size(); ++i) { std::printf("%s%s", i ? " -> " : "", path.hops[i].name.c_str()); }
  std::printf("   (generation %lu)\n", static_cast<unsigned long>(path.generation));
}

} // namespace

int main(int argc, char **argv)
{
  std::string configPath;
  for (int i = 1; i < argc; ++i)
  {
    const std::string arg = argv[i];
    if (arg == "--config" && i + 1 < argc) { configPath = argv[++i]; }
    else if (arg == "--help" || arg == "-h")
    {
      std::printf("usage: %s --config FILE\n", argv[0]);
      return 0;
    }
  }
  if (configPath.empty())
  {
    std::fprintf(stderr, "usage: %s --config FILE\n", argv[0]);
    return 2;
  }

  try
  {
    const serving::coordinator::CoordinatorConfig config = serving::coordinator::loadCoordinatorConfig(configPath);
    const RoutingMode                             mode   = config.mode;
    std::printf("routing mode: %s (%s), from %s\n\n",
                routingModeName(mode),
                mode == RoutingMode::CoordinatorIngress ? "Option A -- coordinator is the ingress point" : "Option B -- coordinator publishes routing rules",
                configPath.c_str());

    // The file lists every replica in the pipeline; the strategy layer wants
    // only their routing identity.
    std::vector<ReplicaSpec> replicas;
    for (const auto &binding : config.replicas) { replicas.push_back(binding.spec); }

    SessionDirectory  sessions(config.routing.sessionTtlSeconds);
    RoutingPlanner    planner(config.routing, sessions, replicas);
    PrintingTransport transport;
    auto              strategy = makeRoutingStrategy(mode, planner, &transport);

    std::printf("  three turns of one conversation:\n");
    for (int turn = 1; turn <= 3; ++turn)
    {
      std::printf("    turn %d\n", turn);
      printPath("path", strategy->onRequest("chat-1"));
    }

    const RoutingStats afterTurns = strategy->stats();
    std::printf("\n    plans computed: %d    rules published: %d\n", afterTurns.plansComputed, afterTurns.rulesPublished);
    std::printf("    %s\n\n",
                mode == RoutingMode::CoordinatorIngress ? "-> the coordinator decided on every turn: it is on the critical path"
                                                        : "-> the coordinator decided once: the steady-state path never consults it");

    std::printf("  a replica on the path dies:\n");
    const RoutingPath current = strategy->onRequest("chat-1");
    const std::string lost    = current.hops[1].name;
    std::printf("    losing %s\n", lost.c_str());
    strategy->onReplicaLost(lost);
    printPath("replanned", strategy->onRequest("chat-1"));

    const RoutingStats final = strategy->stats();
    std::printf("\n    plans computed: %d    rules published: %d    paths invalidated: %d\n", final.plansComputed, final.rulesPublished, final.pathsInvalidated);
    return 0;
  }
  catch (const std::exception &e)
  {
    std::fprintf(stderr, "fatal: %s\n", e.what());
    return 1;
  }
}
