#pragma once

/**
 * Streaming reverse proxy to one replica.
 *
 * Request bytes are forwarded verbatim; only the session id is read. The
 * upstream runs on its own thread and feeds the response's chunked provider
 * through a bounded queue. Closing the client closes the upstream, which is
 * serving's cancellation path.
 */

#include <atomic>
#include <mutex>
#include <string>

#include <serving/router/routing.hpp>

namespace httplib
{
struct Request;
struct Response;
} // namespace httplib

namespace serving::router
{

inline constexpr const char *SESSION_HEADER = "x-session-id";

/**
 * Whether a client-supplied session id may be used as-is.
 *
 * It is echoed in a response header and used as a map key, so it must be
 * header-safe: no CR/LF and no non-latin-1 byte. Matches the Python's
 * `\A[A-Za-z0-9._:-]{1,128}\Z`, anchored so a trailing newline cannot slip
 * through.
 */
[[nodiscard]] bool isUsableSessionId(const std::string &candidate);

/**
 * Header, then a top-level `session_id` in the JSON body, then a new id.
 *
 * An unusable id is replaced rather than refused: the id only decides routing,
 * and a bad one should not fail the request. An absent or unparsable body is
 * likewise not an error -- the router does not validate the payload, the
 * replica does.
 */
[[nodiscard]] std::string resolveSessionId(const httplib::Request &request, const std::string &body);

/**
 * Routes one request to a replica and relays the response back.
 *
 * A replica that dies before answering is retried on another replica of the
 * same partition -- the deck's "re-send its in-flight requests". The retry is
 * only ever attempted while **nothing has reached the client**: once the
 * upstream status is known the relay has begun, and a second attempt would put
 * two answers on one connection. That line is not a heuristic, it is the whole
 * of the correctness argument.
 *
 * A replica that dies *during* the answer cannot be retried. The stream is
 * aborted rather than ended cleanly, because a truncated answer that looks
 * like a complete one is worse than a visible failure -- which is the deck's
 * complaint about the Python stack, in the router rather than the engine.
 */
class ReplicaProxy
{
  public:

  /// `registryMutex` guards `registry` against the health-monitor thread.
  /// `maxAttempts` counts the first try, so 1 disables retrying.
  ReplicaProxy(ReplicaRegistry &registry, std::mutex &registryMutex, double connectTimeoutSeconds, double requestTimeoutSeconds, int maxAttempts = 3);

  void forward(const httplib::Request &request, httplib::Response &response, const std::string &path);

  /**
   * The same relay, but the first attempt goes to a replica somebody else
   * chose -- the coordinator, which planned this session's path and wants the
   * request to enter it at that hop rather than wherever affinity would land.
   *
   * Only the first attempt is pinned. A pinned replica that cannot be reached
   * falls back to the registry, because a request that nothing has answered
   * yet is still better served by a sibling than failed: the coordinator hears
   * about the loss separately and replans.
   *
   * An empty or unroutable name is the unpinned relay.
   */
  void forwardPinned(const httplib::Request &request, httplib::Response &response, const std::string &path, const std::string &preferredReplica);

  /// Requests that reached a second replica because the first was unreachable.
  [[nodiscard]] int resends() const { return _resends.load(); }
  /// Answers cut short by a replica dying mid-stream. These cannot be retried.
  [[nodiscard]] int truncated() const { return _truncated.load(); }

  private:

  struct Attempt;

  /// One attempt against one replica, returning once its status is known or
  /// the attempt has provably failed without one.
  [[nodiscard]] Attempt send(const ReplicaSpec &replica, const httplib::Request &request, const std::string &body, const std::string &path);

  ReplicaRegistry &_registry;
  std::mutex      &_registryMutex;
  double           _connectTimeoutSeconds;
  double           _requestTimeoutSeconds;
  int              _maxAttempts;

  std::atomic<int> _resends{0};
  std::atomic<int> _truncated{0};
};

} // namespace serving::router
