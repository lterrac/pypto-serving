#pragma once

/**
 * Binding an httplib server and serving it on a thread, in one place.
 *
 * Three servers -- the replica's API, the router's ingress and the deployer's
 * operator surface -- each carried their own copy of this, down to the comment
 * explaining why `bind_to_any_port` is used for an ephemeral port. The copies
 * had begun to differ in ways that were accidental rather than meant: only one
 * guarded `stop()` against being entered twice, and that one is not among the
 * two that are also called from a destructor.
 *
 * What genuinely differs between them stays with them: the router runs a health
 * poller it has to wake and join, and each names itself in its own error.
 */

#include <atomic>
#include <stdexcept>
#include <string>
#include <thread>

#include <httplib.h>

namespace serving::util
{

/**
 * Bind, then accept on `thread`. Returns the port actually bound.
 *
 * `port` 0 asks the kernel for one: `bind_to_port(host, 0)` reports success
 * without saying which it chose, so `bind_to_any_port` is used and its answer
 * returned. `what` names the server in the error, because "could not bind" on
 * a machine running three of them says nothing on its own.
 *
 * Returns only once the server is accepting, rather than letting the first
 * request race the accept loop.
 */
inline int bindAndServe(httplib::Server &server, const std::string &host, int port, std::thread &thread, const char *what)
{
  int bound = port;
  if (port == 0)
  {
    bound = server.bind_to_any_port(host.c_str());
    if (bound <= 0) { throw std::runtime_error(std::string(what) + " could not bind " + host + " to an ephemeral port"); }
  }
  else if (!server.bind_to_port(host.c_str(), port))
  {
    throw std::runtime_error(std::string(what) + " could not bind " + host + ":" + std::to_string(port));
  }

  thread = std::thread([&server] { server.listen_after_bind(); });
  server.wait_until_ready();
  return bound;
}

/**
 * Stop accepting and join the thread, once.
 *
 * `serving` is exchanged to false so a second call -- or a concurrent one from
 * a destructor while a caller is already stopping -- returns rather than
 * joining a thread another thread is already joining, which is undefined.
 */
inline void stopServing(httplib::Server *server, std::thread &thread, std::atomic<bool> &serving)
{
  if (!serving.exchange(false)) { return; }
  if (server != nullptr) { server->stop(); }
  if (thread.joinable()) { thread.join(); }
}

} // namespace serving::util
