/**
 * Drives the two routing strategies, printing the path per turn and what
 * `rules` would publish.
 *
 *   routing-modes [--routing-mode ingress|rules]
 */

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

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

ReplicaSpec replica(const std::string &name, int partition, int port)
{
  ReplicaSpec s;
  s.name      = name;
  s.partition = partition;
  s.host      = "127.0.0.1";
  s.port      = port;
  return s;
}

void printPath(const char *label, const RoutingPath &path)
{
  std::printf("      %s: ", label);
  for (size_t i = 0; i < path.hops.size(); ++i) { std::printf("%s%s", i ? " -> " : "", path.hops[i].name.c_str()); }
  std::printf("   (generation %lu)\n", static_cast<unsigned long>(path.generation));
}

} // namespace

int main(int argc, char **argv)
{
  std::string modeText = "ingress";
  for (int i = 1; i < argc; ++i)
  {
    const std::string arg = argv[i];
    if (arg == "--routing-mode" && i + 1 < argc) { modeText = argv[++i]; }
    else if (arg == "--help" || arg == "-h")
    {
      std::printf("usage: %s [--routing-mode ingress|rules]\n", argv[0]);
      return 0;
    }
  }

  try
  {
    const RoutingMode mode = parseRoutingMode(modeText);
    std::printf("routing mode: %s (%s)\n\n",
                routingModeName(mode),
                mode == RoutingMode::CoordinatorIngress ? "Option A -- coordinator is the ingress point" : "Option B -- coordinator publishes routing rules");

    // Three partitions of two replicas: a split model, each stage replicated.
    const std::vector<ReplicaSpec> replicas{
      replica("p0a", 0, 8000), replica("p0b", 0, 8001), replica("p1a", 1, 8002), replica("p1b", 1, 8003), replica("p2a", 2, 8004), replica("p2b", 2, 8005)};

    SessionDirectory  sessions(600.0);
    RoutingPlanner    planner(RoutingConfig{}, sessions, replicas);
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
