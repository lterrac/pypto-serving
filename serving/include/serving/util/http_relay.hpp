#pragma once

/**
 * Sending one HTTP request and relaying its answer as it arrives.
 *
 * Two places need this: the router proxying to a replica, and a replica
 * forwarding to the next hop of its pipeline. Both buffered at some point in
 * their life, and buffering is the bug that matters here -- an SSE generation
 * held until it is complete turns time-to-first-token into time-to-last-token,
 * and a read timeout measured across the whole body fails a long generation
 * after discarding everything it produced.
 *
 * What the two share is the mechanism: fire the request on a thread, push each
 * chunk into a bounded queue as the upstream produces it, and let the caller
 * drain that queue into its own response. What they do not share is policy --
 * the router retries across siblings and holds a slot it must release, the
 * replica has one fixed next hop and a forwarding-depth header -- so this
 * returns the parts and wires nothing.
 */

#include <atomic>
#include <cstddef>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <httplib.h>

#include <serving/util/blocking_queue.hpp>

namespace serving::util
{

using ChunkQueue = BlockingQueue<std::string>;

/// One request in flight. The status and headers are settled by the time
/// `startRelay` returns; the body keeps arriving on `queue` after it.
struct Relay
{
  std::shared_ptr<ChunkQueue>        queue;
  std::shared_ptr<std::atomic<int>>  status;
  std::shared_ptr<httplib::Headers>  headers;
  /// The connection failed rather than answering. Distinct from a non-200:
  /// an upstream that refuses is reachable, one that never spoke is not.
  std::shared_ptr<std::atomic<bool>> transportFailed;

  /// Nothing answered at all -- no status ever arrived. A caller may still try
  /// somewhere else, because nothing has reached its own client yet.
  [[nodiscard]] bool unreachable() const { return transportFailed->load() && status->load() == 0; }
};

/**
 * POST `body` to `host`:`port``path` and begin relaying the answer.
 *
 * Returns once the upstream status is known, or once the attempt has provably
 * failed without one -- so a caller can decide what to do before it has
 * committed anything to its own client.
 *
 * `requestTimeoutSeconds` is the gap between chunks, not a budget for the whole
 * answer: a generation longer than it is ordinary, a silence that long is not.
 * Closing `queue` stops the upstream read and so drops the connection, which is
 * the only cancellation this path has.
 */
inline Relay startRelay(const std::string      &host,
                        int                     port,
                        const std::string      &path,
                        const std::string      &body,
                        const httplib::Headers &headers,
                        double                  connectTimeoutSeconds,
                        double                  requestTimeoutSeconds,
                        size_t                  maxChunksInFlight)
{
  Relay relay;
  relay.queue           = std::make_shared<ChunkQueue>(maxChunksInFlight);
  relay.status          = std::make_shared<std::atomic<int>>(0);
  relay.headers         = std::make_shared<httplib::Headers>();
  relay.transportFailed = std::make_shared<std::atomic<bool>>(false);

  // The status and headers must be known before the body is relayed, so the
  // thread signals once the response handler has fired -- or once it has given
  // up, so a caller is never left waiting on a handler that will not run.
  auto ready     = std::make_shared<std::promise<void>>();
  auto future    = ready->get_future();
  auto signalled = std::make_shared<std::once_flag>();

  auto queue           = relay.queue;
  auto upstreamStatus  = relay.status;
  auto upstreamHeaders = relay.headers;
  auto transportFailed = relay.transportFailed;

  std::thread upstream([=] {
    httplib::Client client(host, port);
    client.set_connection_timeout(static_cast<time_t>(connectTimeoutSeconds), 0);
    client.set_read_timeout(static_cast<time_t>(requestTimeoutSeconds), 0);
    client.set_write_timeout(static_cast<time_t>(requestTimeoutSeconds), 0);

    httplib::Request request;
    request.method  = "POST";
    request.path    = path;
    request.body    = body;
    request.headers = headers;
    request.set_header("Content-Type", "application/json");

    request.response_handler = [=](const httplib::Response &response) {
      upstreamStatus->store(response.status);
      *upstreamHeaders = response.headers;
      std::call_once(*signalled, [&] { ready->set_value(); });
      return true;
    };
    request.content_receiver = [=](const char *data, size_t length, uint64_t, uint64_t) {
      if (queue->closed()) { return false; } // the client hung up; stop pulling
      queue->push(std::string(data, length));
      return true;
    };

    httplib::Response response;
    httplib::Error    error = httplib::Error::Success;
    const bool        ok    = client.send(request, response, error);
    if (!ok || error != httplib::Error::Success) { transportFailed->store(true); }

    // Unblock the caller even when the handler never fired.
    std::call_once(*signalled, [&] { ready->set_value(); });
    queue->close();
  });
  upstream.detach();

  future.wait();
  return relay;
}

} // namespace serving::util
