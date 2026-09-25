/**
 * pypto-serving-router-cpp: fronts N replicas with session affinity and
 * health polling. Routes to replicas that already exist; it does not launch
 * them.
 */

#include <atomic>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

#include <serving/router/router_server.hpp>
#include <serving/util/signals.hpp>

namespace
{

[[noreturn]] void usage(const char *argv0, int code)
{
  std::fprintf(code == 0 ? stdout : stderr,
               "usage: %s --replica NAME=HOST:PORT [--replica ...] [options]\n"
               "\n"
               "  --replica NAME=HOST:PORT   a serving replica (repeatable, at least one)\n"
               "  --host HOST                bind address (default: 0.0.0.0)\n"
               "  --port N                   bind port (default: 8080)\n"
               "  --session-ttl N            seconds a conversation stays pinned (default: 1800)\n"
               "  --affinity-slack N         extra outstanding requests affinity tolerates (default: 8)\n"
               "  --health-interval N        seconds between health polls (default: 5)\n"
               "  --connect-timeout N        seconds to reach a replica (default: 2)\n"
               "  --request-timeout N        seconds for one generation (default: 300)\n",
               argv0);
  std::exit(code);
}

/// Parse NAME=HOST:PORT.
serving::router::ReplicaSpec parseReplica(const std::string &text)
{
  const auto equals = text.find('=');
  const auto colon  = text.rfind(':');
  if (equals == std::string::npos || colon == std::string::npos || colon < equals) { throw std::invalid_argument("expected NAME=HOST:PORT, got '" + text + "'"); }
  serving::router::ReplicaSpec spec;
  spec.name = text.substr(0, equals);
  spec.host = text.substr(equals + 1, colon - equals - 1);
  spec.port = std::stoi(text.substr(colon + 1));
  if (spec.name.empty() || spec.host.empty()) { throw std::invalid_argument("empty name or host in '" + text + "'"); }
  return spec;
}

} // namespace

int main(int argc, char **argv)
{
  // Before any thread exists: threads inherit the mask, and a shutdown signal
  // delivered to one that is not waiting would terminate the process.
  serving::util::blockShutdownSignals();

  serving::router::RouterServerConfig       config;
  std::vector<serving::router::ReplicaSpec> replicas;

  try
  {
    for (int i = 1; i < argc; ++i)
    {
      const std::string arg  = argv[i];
      const auto        next = [&]() -> std::string {
        if (i + 1 >= argc) { usage(argv[0], 2); }
        return argv[++i];
      };
      if (arg == "--help" || arg == "-h") { usage(argv[0], 0); }
      else if (arg == "--replica") { replicas.push_back(parseReplica(next())); }
      else if (arg == "--host") { config.host = next(); }
      else if (arg == "--port") { config.port = std::stoi(next()); }
      else if (arg == "--session-ttl") { config.sessionTtlSeconds = std::stod(next()); }
      else if (arg == "--affinity-slack") { config.affinitySlack = std::stoi(next()); }
      else if (arg == "--health-interval") { config.healthIntervalSeconds = std::stod(next()); }
      else if (arg == "--connect-timeout") { config.connectTimeoutSeconds = std::stod(next()); }
      else if (arg == "--request-timeout") { config.requestTimeoutSeconds = std::stod(next()); }
      else
      {
        std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
        usage(argv[0], 2);
      }
    }
    if (replicas.empty())
    {
      std::fprintf(stderr, "at least one --replica is required\n");
      usage(argv[0], 2);
    }

    serving::router::RouterServer router(config, replicas);
    router.start();
    std::printf("[router] listening on %s:%d, fronting %zu replica(s):\n", config.host.c_str(), router.boundPort(), replicas.size());
    for (const auto &replica : replicas) { std::printf("[router]   %s -> %s\n", replica.name.c_str(), replica.baseUrl().c_str()); }
    std::fflush(stdout);

    serving::util::awaitShutdownSignal();
    std::printf("[router] shutting down\n");
    router.stop();
    return 0;
  }
  catch (const std::exception &e)
  {
    std::fprintf(stderr, "[router] fatal: %s\n", e.what());
    return 1;
  }
}
