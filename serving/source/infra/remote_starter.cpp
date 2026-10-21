#include <serving/infra/remote_starter.hpp>

#include <cstdio>

#include <unistd.h>

#include <utility>

namespace serving::infra
{

namespace
{

void replaceAll(std::string &text, const std::string &token, const std::string &value)
{
  for (size_t at = text.find(token); at != std::string::npos; at = text.find(token, at + value.size())) { text.replace(at, token.size(), value); }
}

/// Wrap a command so a shell on the far side runs it as one argument.
///
/// Single quotes, with embedded ones broken out: a model directory or a
/// generate-config with a space in it must not become two arguments on the
/// other side of ssh.
std::string quoted(const std::string &command)
{
  std::string out = "'";
  for (const char c : command)
  {
    if (c == '\'') { out += "'\\''"; }
    else { out += c; }
  }
  out += "'";
  return out;
}

} // namespace

RemoteStarter::RemoteStarter(const StateMap &map, std::string localNode, std::string commandTemplate, RemoteLaunch launch)
  : _map(map),
    _localNode(std::move(localNode)),
    _template(std::move(commandTemplate)),
    _launch(std::move(launch))
{}

bool RemoteStarter::isLocal(const Replica &replica) const { return !_localNode.empty() && replica.node == _localNode; }

std::string RemoteStarter::hostOf(const Replica &replica) const
{
  const Node *node = _map.node(replica.node);
  return node == nullptr || node->host.empty() ? replica.node : node->host;
}

std::string RemoteStarter::pidFileFor(const Replica &replica) const { return _launch.stateDirectory + "/serving-" + replica.name + ".pgid"; }

std::string RemoteStarter::remoteCommand(const Replica &replica) const
{
  // Deliberately not detached. `setsid` here looked right -- the replica would
  // lead its own group on the far side -- but it returns as soon as it has
  // forked, so the reach exits immediately and its liveness stops meaning
  // anything. The supervisor then reads a launch that is still loading as one
  // that failed, retries, and starts a second replica on the same card while
  // the first is still coming up. Both were then leaked, because only the
  // newer one was still referenced.
  //
  // Staying attached makes the reach last exactly as long as the replica, so
  // `running()` answers the same question it answers locally. ssh holds the
  // connection open for the replica's lifetime, which is what it is for.
  //
  // `$$` is recorded before `exec` so the file names the replica rather than a
  // wrapper, and `exec` leaves no wrapper to signal separately.
  // The path is quoted because it embeds the replica's name, which arrives from
  // a deployment document this process did not write. A name carrying a space
  // would otherwise split the redirect and leave the pid file unwritten, so the
  // replica could never be stopped; one carrying `;` or a backtick would run on
  // the far side as the ssh user.
  const std::string file = quoted(pidFileFor(replica));
  return "sh -c " + quoted("echo $$ > " + file + "; exec " + ProcessStarter(_template).commandFor(replica));
}

std::string RemoteStarter::commandFor(const Replica &replica) const
{
  if (isLocal(replica)) { return ProcessStarter(_template).commandFor(replica); }

  std::string reach = _launch.reach;
  replaceAll(reach, "{host}", hostOf(replica));
  replaceAll(reach, "{command}", quoted(remoteCommand(replica)));
  return reach;
}

bool RemoteStarter::start(const Replica &replica)
{
  if (running(replica.name)) { return false; }

  // The composed command is already final, so the per-replica starter is given
  // it verbatim rather than a template with anything left to substitute.
  auto       starter = std::make_unique<ProcessStarter>(commandFor(replica));
  const bool started = starter->start(replica);
  if (started) { _starters[replica.name] = std::move(starter); }
  return started;
}

bool RemoteStarter::running(const std::string &replica)
{
  const auto it = _starters.find(replica);
  return it != _starters.end() && it->second->running(replica);
}

void RemoteStarter::stop(const Replica &replica)
{
  const auto it = _starters.find(replica.name);
  if (it == _starters.end()) { return; }

  if (!isLocal(replica))
  {
    // Killing the reach command would leave the replica running on the far
    // side holding its card, so the far side is told first and the reach is
    // torn down after. Failure here is not reported: the node may already be
    // gone, which is frequently why it is being stopped.
    // Quoted for the same reason the launch side quotes it: the name inside the
    // path is not this process's to trust.
    const std::string file = quoted(pidFileFor(replica));
    // The group first, because the replica forks its own workers, then the pid
    // itself: without `setsid` the replica may not lead a group of its own, and
    // a kill that silently reaches nothing is how one gets left holding a card.
    const std::string grace = std::to_string(_launch.graceSeconds);
    // The file is read and removed before the killing starts. The stop waits a
    // bounded time and then tears the reach down, so anything after the grace
    // sleep may never run -- and a pidfile left behind outlives the pid, which
    // the next reader has no way to tell from a live one.
    //
    // The group first, because the replica forks its own workers, then the pid
    // itself: without a session of its own the replica may not lead a group,
    // and a kill that silently reaches nothing is how one gets left holding a
    // card.
    const std::string kill = "p=$(cat " + file + " 2>/dev/null); rm -f " + file + "; [ -n \"$p\" ] || exit 0; kill -TERM -$p 2>/dev/null; kill -TERM $p 2>/dev/null; sleep " +
                             grace + "; kill -KILL -$p 2>/dev/null; kill -KILL $p 2>/dev/null";

    std::string reach = _launch.reach;
    replaceAll(reach, "{host}", hostOf(replica));
    replaceAll(reach, "{command}", quoted("sh -c " + quoted(kill)));

    ProcessStarter cleanup(reach);
    Replica        asItself = replica;
    if (!cleanup.start(asItself))
    {
      // Nothing else will reach it: tearing the reach down kills the local end
      // only, and over ssh that leaves the replica serving on the far side
      // with its card held and nobody tracking it.
      std::fprintf(stderr, "[remote] could not reach %s to stop %s; it may still be running there\n", hostOf(replica).c_str(), replica.name.c_str());
    }
    else
    {
      // Wait for it rather than racing the reach teardown below: a cleanup
      // killed halfway through can leave the replica alive and unreferenced.
      bool finished = false;
      for (int waited = 0; waited < (_launch.graceSeconds + 5) * 1000 && !finished; waited += 100)
      {
        finished = !cleanup.running(asItself.name);
        if (!finished) { ::usleep(100 * 1000); }
      }
      if (!finished)
      {
        std::fprintf(
          stderr, "[remote] stopping %s on %s did not finish within %ds; it may still be running there\n", replica.name.c_str(), hostOf(replica).c_str(), _launch.graceSeconds + 5);
      }
      cleanup.stop(asItself);
    }
  }

  it->second->stop(replica);
  _starters.erase(it);
}

} // namespace serving::infra
