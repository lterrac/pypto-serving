/**
 * pypto-serving-deployer-cpp: the control plane for one deployment.
 *
 * Reads the document, launches and supervises the replicas it asks for,
 * watches their health, runs the coordinator that plans where a session goes,
 * and serves both the ingress requests arrive on and the operator API.
 *
 *   pypto-serving-deployer-cpp --config deployment-config.json
 *
 * --dry-run prints what it would start and exits, because the one thing worse
 * than a deployment that will not start is one that starts the wrong thing.
 */

#include <cstdio>
#include <exception>
#include <string>

#include <serving/deployer/deployment.hpp>
#include <serving/infra/document.hpp>
#include <serving/infra/remote_starter.hpp>
#include <serving/util/signals.hpp>

namespace
{

[[noreturn]] void usage(const char *argv0, int code)
{
  std::fprintf(code == 0 ? stdout : stderr,
               "usage: %s --config FILE [--dry-run]\n"
               "\n"
               "  --config FILE   the deployment configuration (JSON)\n"
               "  --dry-run       report what would be started, then exit\n",
               argv0);
  std::exit(code);
}

} // namespace

int main(int argc, char **argv)
{
  std::string configPath;
  bool        dryRun = false;

  for (int i = 1; i < argc; ++i)
  {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") { usage(argv[0], 0); }
    else if (arg == "--config" && i + 1 < argc) { configPath = argv[++i]; }
    else if (arg == "--dry-run") { dryRun = true; }
    else { usage(argv[0], 2); }
  }
  if (configPath.empty()) { usage(argv[0], 2); }

  try
  {
    const auto config = serving::deployer::loadDeployerConfig(configPath);

    if (dryRun)
    {
      const auto map = serving::infra::loadDocument(config.document);
      std::printf("deployment from %s (document %s)\n", configPath.c_str(), config.document.c_str());
      std::printf("  routing: %s, requests enter partition %d\n", serving::router::routingModeName(config.mode), config.partition);
      std::printf("  ingress %s:%d   operator API %s:%d\n", config.ingressHost.c_str(), config.ingressPort, config.operatorHost.c_str(), config.operatorPort);

      const auto problems = serving::infra::validate(map);
      if (!problems.empty())
      {
        std::fprintf(stderr, "  the deployment is not coherent:\n");
        for (const auto &problem : problems) { std::fprintf(stderr, "    %s\n", problem.c_str()); }
        return 1;
      }

      // Through the same starter the run would use, so what is printed is what
      // would happen -- including the reach command for a replica on another
      // node, which is the part worth seeing before it runs.
      serving::infra::ProcessStarter local(config.launch);
      serving::infra::RemoteStarter  remote(map, config.node, config.launch, config.remote);
      for (const auto &replica : map.replicas())
      {
        const bool here = config.node.empty() || replica.node == config.node;
        std::printf("  %-10s partition %d on %-14s devices", replica.name.c_str(), replica.partition, replica.node.c_str());
        for (const int device : replica.devices) { std::printf(" %d", device); }
        std::printf("  port %d%s\n", replica.port, here ? "" : "   (reached)");
        if (!config.launch.empty()) { std::printf("      %s\n", (config.node.empty() ? local.commandFor(replica) : remote.commandFor(replica)).c_str()); }
      }
      if (config.launch.empty()) { std::printf("  no launch command: the replicas are expected to be running already\n"); }
      return 0;
    }

    // Before anything starts a thread. `awaitShutdownSignal` uses sigwait,
    // which needs the signals blocked in every thread; without this SIGTERM
    // takes its default action and kills the process outright -- no shutdown,
    // no replicas stopped, and on another node nothing left to stop them.
    serving::util::blockShutdownSignals();

    serving::deployer::Deployment deployment(config);
    deployment.start();

    std::printf(
      "[deployer] ingress on %s:%d, operator API on %s:%d\n", config.ingressHost.c_str(), deployment.ingressPort(), config.operatorHost.c_str(), deployment.operatorPort());
    std::fflush(stdout);

    deployment.run();
    return 0;
  }
  catch (const std::exception &e)
  {
    std::fprintf(stderr, "fatal: %s\n", e.what());
    return 1;
  }
}
