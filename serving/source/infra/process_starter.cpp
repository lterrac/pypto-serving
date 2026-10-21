#include <serving/infra/process_starter.hpp>

#include <dirent.h>
#include <signal.h>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <spawn.h>
#include <sstream>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

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

/**
 * Every live process in session `sid`, excluding the session leader itself.
 *
 * The group is not enough. simpler's per-chip children call `os.setpgid(0, 0)`
 * as they fork (`simpler/worker.py:8127`), making each its own group leader, so
 * `kill(-pgid)` reaches the wrapper and the main python and misses precisely
 * the processes holding the card's memory. `setpgid` moves a process between
 * groups *within* its session, so the session is what still contains them --
 * which is why simpler reaps its own remote runner the same way
 * (`remote_l3_worker.py:146`).
 *
 * A PPID walk would not do: an orphaned worker is reparented to init, and the
 * orphan is the case that strands a card.
 */
std::vector<pid_t> sessionMembers(pid_t sid)
{
  std::vector<pid_t> members;
  if (sid <= 0) { return members; }

  DIR *proc = ::opendir("/proc");
  if (proc == nullptr) { return members; }

  while (const dirent *entry = ::readdir(proc))
  {
    const pid_t pid = static_cast<pid_t>(std::atoi(entry->d_name));
    if (pid <= 0 || pid == sid) { continue; }

    std::ifstream stat("/proc/" + std::string(entry->d_name) + "/stat");
    std::string   line;
    if (!stat || !std::getline(stat, line)) { continue; }

    // `comm` is parenthesised and may itself contain spaces and parentheses,
    // so the fields are read from after its last ')': state, ppid, pgrp, sid.
    const size_t close = line.rfind(')');
    if (close == std::string::npos) { continue; }

    std::istringstream rest(line.substr(close + 1));
    std::string        state;
    pid_t              ppid = 0;
    pid_t              pgrp = 0;
    pid_t              session = 0;
    if (!(rest >> state >> ppid >> pgrp >> session)) { continue; }
    if (session == sid) { members.push_back(pid); }
  }
  ::closedir(proc);
  return members;
}

/// Signal every process still in the replica's session. Returns how many were
/// signalled, so a caller can tell "nothing left" from "still dying".
int signalSession(pid_t sid, int signal)
{
  int reached = 0;
  for (const pid_t pid : sessionMembers(sid))
  {
    if (::kill(pid, signal) == 0) { reached += 1; }
  }
  return reached;
}

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

  // Its own session, not merely its own group. The command is a shell, and a
  // shell that spawns rather than execs leaves the replica one indirection
  // away; worse, simpler's per-chip children regroup themselves as they fork,
  // so a group is not a container for them. A session is: setpgid moves a
  // process between groups within its session and cannot leave it.
  //
  // A session leader is also its own group leader, so `kill(-pid)` still works
  // for everything that did stay in the group, and `waitpid` still works
  // because the leader remains this process's child -- the objection recorded
  // against setsid in remote_starter.cpp applies to detaching the ssh *reach*,
  // not to a local child.
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
  bool ownSession = false;
#ifdef POSIX_SPAWN_SETSID
  // SETSID already makes the child a session *and* group leader, so SETPGROUP
  // would be redundant and is not combined with it.
  ::posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETSID | POSIX_SPAWN_SETSIGMASK);
  ownSession = true;
#else
  ::posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_SETSIGMASK);
  ::posix_spawnattr_setpgroup(&attributes, 0);
#endif

  pid_t     pid = 0;
  const int rc  = ::posix_spawn(&pid, "/bin/sh", nullptr, &attributes, argv, environ);
  ::posix_spawnattr_destroy(&attributes);
  if (rc != 0) { return false; }

  // Without a session of its own the session id is this process's, and sweeping
  // by it would signal the deployer and everything else it started. Recorded
  // per launch so the sweep is only ever used when it is safe.
  _pids[replica.name] = Launch{pid, false, ownSession};
  return true;
}

void ProcessStarter::stop(const Replica &replica)
{
  const auto it = _pids.find(replica.name);
  if (it == _pids.end()) { return; }

  Launch     &launch = it->second;
  const pid_t leader = launch.leader;
  const pid_t session = launch.ownSession ? leader : 0;

  // The group first, for the wrapper and anything that stayed in it, then every
  // other process in the session -- which is where simpler's per-chip workers
  // went when they regrouped themselves.
  ::kill(-leader, SIGTERM);
  signalSession(session, SIGTERM);

  // A model mid-load does not always die promptly, and a replica still alive
  // after this would keep its device out of the pool for good. The leader
  // exiting is not enough on its own: the workers holding the card can outlive
  // it, which is the leak this waits for.
  for (int waited = 0; waited < GRACE_MILLISECONDS; waited += POLL_MILLISECONDS)
  {
    const bool leaderGone  = reap(launch);
    const bool sessionGone = sessionMembers(session).empty();
    if (leaderGone && sessionGone) { break; }
    ::usleep(POLL_MILLISECONDS * 1000);
  }
  ::kill(-leader, SIGKILL);
  signalSession(session, SIGKILL);
  // Reaped with a wait that can actually succeed. WNOHANG issued the instant
  // after SIGKILL always returns 0 -- the signal has not been delivered yet --
  // so the old code reaped nothing and then dropped the pid, leaving a zombie
  // per forced stop until the process table filled.
  for (int waited = 0; waited < KILL_REAP_MILLISECONDS; waited += POLL_MILLISECONDS)
  {
    if (reap(launch) && sessionMembers(session).empty()) { break; }
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

  // The leader is gone, but the replica may not be: the documented launch is a
  // wrapper, and the per-chip workers outlive the python that forked them.
  // They still hold the card's memory, so the replica is still running and the
  // entry has to stay -- dropping it here made every later stop() return early
  // and left those workers unreachable for the life of the process.
  //
  // Checked by session as well as by group, because a worker that called
  // setpgid(0, 0) is in neither this group nor any other we know the name of.
  if (::kill(-it->second.leader, 0) == 0) { return true; }
  if (it->second.ownSession && !sessionMembers(it->second.leader).empty()) { return true; }

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
