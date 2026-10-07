#include <serving/infra/process_starter.hpp>

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
  for (const auto &[name, pid] : _pids)
  {
    ::kill(-pid, SIGTERM);
    int status = 0;
    ::waitpid(pid, &status, 0);
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
  ::posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP);
  ::posix_spawnattr_setpgroup(&attributes, 0);

  pid_t     pid = 0;
  const int rc  = ::posix_spawn(&pid, "/bin/sh", nullptr, &attributes, argv, environ);
  ::posix_spawnattr_destroy(&attributes);
  if (rc != 0) { return false; }

  _pids[replica.name] = pid;
  return true;
}

void ProcessStarter::stop(const Replica &replica)
{
  const auto it = _pids.find(replica.name);
  if (it == _pids.end()) { return; }

  const pid_t leader = it->second;
  // Negative pid: the whole group, so a wrapper that spawned the replica
  // rather than exec'ing it does not leave it behind holding a card.
  ::kill(-leader, SIGTERM);

  // A model mid-load does not always die promptly, and a replica still alive
  // after this would keep its device out of the pool for good.
  for (int waited = 0; waited < GRACE_MILLISECONDS; waited += POLL_MILLISECONDS)
  {
    int         status = 0;
    const pid_t reaped = ::waitpid(leader, &status, WNOHANG);
    if (reaped != 0) { break; }
    ::usleep(POLL_MILLISECONDS * 1000);
  }
  ::kill(-leader, SIGKILL);
  int status = 0;
  ::waitpid(leader, &status, WNOHANG);

  _pids.erase(it);
}

bool ProcessStarter::running(const std::string &replica)
{
  const auto it = _pids.find(replica);
  if (it == _pids.end()) { return false; }

  int         status = 0;
  const pid_t waited = ::waitpid(it->second, &status, WNOHANG);
  if (waited == 0) { return true; }

  // Exited or vanished: reap it so the pid is not reused under us.
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
