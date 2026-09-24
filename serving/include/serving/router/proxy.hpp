#pragma once

/**
 * Streaming reverse proxy to one replica.
 *
 * Request bytes are forwarded verbatim; only the session id is read. The
 * upstream runs on its own thread and feeds the response's chunked provider
 * through a bounded queue. Closing the client closes the upstream, which is
 * serving's cancellation path.
 */

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

/// Routes one request to a replica and relays the response back.
class ReplicaProxy
{
  public:

  /// `registryMutex` guards `registry` against the health-monitor thread.
  ReplicaProxy(ReplicaRegistry &registry, std::mutex &registryMutex, double connectTimeoutSeconds, double requestTimeoutSeconds);

  void forward(const httplib::Request &request, httplib::Response &response, const std::string &path);

  private:

  ReplicaRegistry &_registry;
  std::mutex      &_registryMutex;
  double           _connectTimeoutSeconds;
  double           _requestTimeoutSeconds;
};

} // namespace serving::router
