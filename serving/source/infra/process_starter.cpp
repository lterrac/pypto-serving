#include <serving/infra/process_starter.hpp>

#include <signal.h>
#include <csignal>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <httplib.h>

extern char **environ;

namespace serving::infra
{

namespace
{

/// How long a replica is given to exit on SIGTERM before it is killed.
constexpr int GRACE_MILLISECONDS = 5000;
constexpr int POLL_MILLISECONDS  = 50;
/// How long to keep reaping after SIGKILL. The signal is delivered promptly,
/// but not before the next instruction, which is why a single WNOHANG here
/// never collected anything.
constexpr int KILL_REAP_MILLISECONDS = 500;

void replaceAll(std::string &text, const std::string &token, const std::string &value)
{
  for (size_t at = text.find(token); at != std::string::npos; at = text.find(token, at + value.size())) { text.replace(at, token.size(), value); }
}

std::string joinDevices(const std::vector<int> &devices)
{
  std::string joined;
  for (const int device : devices) { joined += (joined.empty() ? "" : ",") + std::to_string(device); }
  return joined;
}

} // namespace

ProcessStarter::ProcessStarter(std::string commandTemplate)
  : _template(std::move(commandTemplate))
{}

ProcessStarter::~ProcessStarter()
{
  // A supervisor going away must not leave replicas holding cards.
  for (const auto &[name, launch] : _pids)
  {
    ::kill(-launch.leader, SIGTERM);
    if (launch.reaped) { continue; }
    int status = 0;
    ::waitpid(launch.leader, &status, 0);
  }
}

std::string ProcessStarter::commandFor(const Replica &replica) const
{
  std::string command = _template;
  replaceAll(command, "{name}", replica.name);
  replaceAll(command, "{node}", replica.node);
  replaceAll(command, "{port}", std::to_string(replica.port));
  replaceAll(command, "{devices}", joinDevices(replica.devices));
  // The common case is one card, so {device} is the first one rather than a list.
  replaceAll(command, "{device}", replica.devices.empty() ? "" : std::to_string(replica.devices.front()));
  return command;
}

bool ProcessStarter::start(const Replica &replica)
{
  // A second start for a replica already running would lose the first pid and
  // leave it holding its card forever.
  if (running(replica.name)) { return false; }

  const std::string command = commandFor(replica);
  char             *argv[]  = {const_cast<char *>("/bin/sh"), const_cast<char *>("-c"), const_cast<char *>(command.c_str()), nullptr};

  // Its own process group. The command is a shell, and a shell that spawns
  // rather than execs leaves the replica one indirection away: signalling the
  // shell alone can leave the replica running and holding its card, which is
  // precisely the failure this class exists to recover from. Signalling the
  // group reaches whatever the template actually started.
  posix_spawnattr_t attributes;
  if (::posix_spawnattr_init(&attributes) != 0) { return false; }
  // The mask is reset as well as the group. This process blocks SIGINT and
  // SIGTERM before it starts any thread, and a spawned child inherits that
  // mask: the replica's own signal handler would never run, so every stop
  // would wait out the grace period and end in SIGKILL with no graceful engine
  // shutdown and no chance to release its device cleanly.
  sigset_t empty;
  ::sigemptyset(&empty);
  ::posix_spawnattr_setsigmask(&attributes, &empty);
  ::posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_SETSIGMASK);
  ::posix_spawnattr_setpgroup(&attributes, 0);

  pid_t     pid = 0;
  const int rc  = ::posix_spawn(&pid, "/bin/sh", nullptr, &attributes, argv, environ);
  ::posix_spawnattr_destroy(&attributes);
  if (rc != 0) { return false; }

  _pids[replica.name] = Launch{pid, false};
  return true;
}

void ProcessStarter::stop(const Replica &replica)
{
  const auto it = _pids.find(replica.name);
  if (it == _pids.end()) { return; }

  Launch     &launch = it->second;
  const pid_t leader = launch.leader;
  // Negative pid: the whole group, so a wrapper that spawned the replica
  // rather than exec'ing it does not leave it behind holding a card.
  ::kill(-leader, SIGTERM);

  // A model mid-load does not always die promptly, and a replica still alive
  // after this would keep its device out of the pool for good.
  for (int waited = 0; waited < GRACE_MILLISECONDS; waited += POLL_MILLISECONDS)
  {
    if (reap(launch)) { break; }
    ::usleep(POLL_MILLISECONDS * 1000);
  }
  ::kill(-leader, SIGKILL);
  // Reaped with a wait that can actually succeed. WNOHANG issued the instant
  // after SIGKILL always returns 0 -- the signal has not been delivered yet --
  // so the old code reaped nothing and then dropped the pid, leaving a zombie
  // per forced stop until the process table filled.
  for (int waited = 0; waited < KILL_REAP_MILLISECONDS; waited += POLL_MILLISECONDS)
  {
    if (reap(launch)) { break; }
    ::usleep(POLL_MILLISECONDS * 1000);
  }

  _pids.erase(it);
}

bool ProcessStarter::reap(Launch &launch)
{
  if (launch.reaped) { return true; }
  int         status = 0;
  const pid_t waited = ::waitpid(launch.leader, &status, WNOHANG);
  if (waited == 0) { return false; }
  // Reaped, or already gone. Either way there is nothing left to wait for, and
  // waiting again on a reaped pid would block or fail.
  launch.reaped = true;
  return true;
}

bool ProcessStarter::running(const std::string &replica)
{
  const auto it = _pids.find(replica);
  if (it == _pids.end()) { return false; }

  if (!reap(it->second)) { return true; }

  // The leader is gone, but its group may not be: the documented launch is a
  // wrapper, and the replica's per-chip workers were forked into that group.
  // They still hold the card's memory, so the replica is still running and the
  // entry has to stay -- dropping it here made every later stop() return early
  // and left those workers unreachable for the life of the process.
  if (::kill(-it->second.leader, 0) == 0) { return true; }

  _pids.erase(it);
  return false;
}

std::string HttpHealthProbe::addressOf(const Replica &replica) const
{
  if (_map != nullptr)
  {
    if (const Node *node = _map->node(replica.node); node != nullptr && !node->host.empty()) { return node->host; }
  }
  return replica.node;
}

Readiness HttpHealthProbe::check(const Replica &replica)
{
  // Nowhere to ask. A replica with no port cannot be probed, so it can only be
  // judged by whether its process is alive -- see LiveAndReadyProbe.
  if (replica.port == 0) { return Readiness::Loading; }

  httplib::Client client(addressOf(replica), replica.port);
  client.set_connection_timeout(static_cast<time_t>(_timeout), 0);
  client.set_read_timeout(static_cast<time_t>(_timeout), 0);

  const auto response = client.Get("/health");
  // 503 while the engine loop is not live yet, which is the whole point of the
  // endpoint reporting readiness rather than liveness. A refused connection is
  // the same answer: still coming up, as far as HTTP can tell.
  if (response == nullptr) { return Readiness::Loading; }
  return response->status == 200 ? Readiness::Ready : Readiness::Loading;
}

} // namespace serving::infra
