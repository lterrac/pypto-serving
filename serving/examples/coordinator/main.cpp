/**
 * Runs CoordinatorModule through initialize, routing, a replica loss, a
 * service tick and finalize. No HiCR instances are stood up.
 *
 *   coordinator-module [--routing-mode ingress|rules]
 */

#include <cstdio>
#include <string>
#include <vector>

#include <serving/coordinator/platform_module.hpp>

namespace
{

using namespace serving;

class PrintingTransport : public router::RuleTransport
{
  public:

  void publish(const router::ReplicaSpec &holder, const std::string &sessionId, const router::ReplicaSpec &nextHop, uint64_t generation) override
  {
    std::printf("    [rule gen=%lu] %s -> %s for session %s\n", static_cast<unsigned long>(generation), holder.name.c_str(), nextHop.name.c_str(), sessionId.c_str());
  }

  void revoke(const std::string &sessionId, uint64_t generation) override
  {
    std::printf("    [revoke gen=%lu] session %s\n", static_cast<unsigned long>(generation), sessionId.c_str());
  }
};

coordinator::ReplicaBinding binding(const std::string &name, int partition, uint64_t instanceId)
{
  coordinator::ReplicaBinding b;
  b.spec.name      = name;
  b.spec.partition = partition;
  b.instanceId     = instanceId;
  return b;
}

void printPath(const router::RoutingPath &path)
{
  std::printf("    path: ");
  for (size_t i = 0; i < path.hops.size(); ++i) { std::printf("%s%s", i ? " -> " : "", path.hops[i].name.c_str()); }
  std::printf("   (generation %lu)\n", static_cast<unsigned long>(path.generation));
}

} // namespace

int main(int argc, char **argv)
{
  std::string modeText = "rules";
  for (int i = 1; i < argc; ++i)
  {
    const std::string arg = argv[i];
    if (arg == "--routing-mode" && i + 1 < argc) { modeText = argv[++i]; }
  }

  coordinator::CoordinatorConfig config;
  config.partition = 1;
  config.mode      = router::parseRoutingMode(modeText);
  config.replicas  = {binding("p0a", 0, 10), binding("p0b", 0, 11), binding("p1a", 1, 20), binding("p1b", 1, 21), binding("p2a", 2, 30), binding("p2b", 2, 31)};

  PrintingTransport transport;
  // A real deployment registers this with serving::system::Engine::addModule,
  // which then drives the lifecycle below.
  coordinator::CoordinatorModule module(config, &transport);

  std::printf("coordinator for partition %d, mode %s\n\n", config.partition, modeText.c_str());

  std::printf("  initialize()\n");
  module.initialize();
  std::printf("    has a service: %s\n\n", module.hasService() ? "yes" : "no");

  std::printf("  two turns of a conversation\n");
  printPath(module.coordinator().routeFor("chat-1"));
  const auto second = module.coordinator().routeFor("chat-1");
  printPath(second);

  std::printf("\n  a heartbeat reports %s gone\n", second.hops[1].name.c_str());
  module.coordinator().reportReplicaLost(second.hops[1].name);
  std::printf("    before the tick, losses applied: %d\n", module.coordinator().lossesApplied());

  // The platform engine drives this on its timer; stepped by hand here.
  std::printf("  service() tick\n");
  module.coordinator().service();
  std::printf("    after the tick, losses applied: %d\n", module.coordinator().lossesApplied());

  std::printf("\n  the next turn re-routes around it\n");
  printPath(module.coordinator().routeFor("chat-1"));

  const auto stats = module.coordinator().stats();
  std::printf("\n  plans: %d   rules published: %d   paths invalidated: %d\n", stats.plansComputed, stats.rulesPublished, stats.pathsInvalidated);

  std::printf("\n  finalize()\n");
  module.finalize();
  std::printf("    initialized: %s\n", module.coordinator().initialized() ? "yes" : "no");
  return 0;
}
