#include <serving/router/proxy.hpp>

#include <serving/util/http_relay.hpp>

#include <serving/util/blocking_queue.hpp>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cctype>
#include <deque>
#include <optional>
#include <future>
#include <memory>
#include <set>
#include <thread>

#include <cstdio>

#include <httplib.h>
#include <nlohmann/json.hpp>

namespace serving::router
{

using ChunkQueue = util::ChunkQueue;

namespace
{

/// Chunks in flight from the replica to the client. Bounded, so a slow client
/// stops the upstream rather than growing without limit.
constexpr size_t MAX_CHUNKS_IN_FLIGHT = 256;

using Json = nlohmann::json;

/// Headers that describe one hop and must not be relayed to the next.
const std::set<std::string> kHopByHop{"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade"};

std::string lower(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
  return value;
}

bool dropFromRequest(const std::string &key)
{
  const auto k = lower(key);
  // The router re-frames the body, so upstream framing headers no longer
  // describe what the client receives.
  return kHopByHop.count(k) != 0 || k == "host" || k == "content-length";
}

bool dropFromResponse(const std::string &key)
{
  const auto k = lower(key);
  return kHopByHop.count(k) != 0 || k == "content-length";
}

void sendError(httplib::Response &response, int status, const std::string &message, const std::string &sessionId)
{
  response.status = status;
  // The session id is returned on every response, this one included: clients
  // rely on it for affinity and may retry with the same id.
  response.set_header(SESSION_HEADER, sessionId);
  response.set_content(Json{{"object", "error"}, {"message", message}}.dump(), "application/json");
}

} // namespace

bool isUsableSessionId(const std::string &candidate)
{
  if (candidate.empty() || candidate.size() > 128) { return false; }
  return std::all_of(candidate.begin(), candidate.end(), [](unsigned char c) { return std::isalnum(c) != 0 || c == '.' || c == '_' || c == ':' || c == '-'; });
}

std::string resolveSessionId(const httplib::Request &request, const std::string &body)
{
  std::string supplied;
  if (request.has_header(SESSION_HEADER)) { supplied = request.get_header_value(SESSION_HEADER); }

  if (supplied.empty() && !body.empty())
  {
    try
    {
      const auto payload = Json::parse(body);
      if (payload.is_object() && payload.contains("session_id") && payload.at("session_id").is_string()) { supplied = payload.at("session_id").get<std::string>(); }
    }
    catch (const std::exception &)
    {
      // Not an error: the router does not validate the payload, the replica does.
    }
  }

  if (isUsableSessionId(supplied)) { return supplied; }
  return newSessionId();
}

ReplicaProxy::ReplicaProxy(ReplicaRegistry &registry, std::mutex &registryMutex, double connectTimeoutSeconds, double requestTimeoutSeconds, int maxAttempts)
  : _registry(registry),
    _registryMutex(registryMutex),
    _connectTimeoutSeconds(connectTimeoutSeconds),
    _requestTimeoutSeconds(requestTimeoutSeconds),
    _maxAttempts(std::max(1, maxAttempts))
{}

/// One attempt against one replica, up to the point where the answer starts.
/// The relay itself is shared with the replica-side forward; what is proxy's
/// own is the retry across siblings and the slot it has to release.
struct ReplicaProxy::Attempt : util::Relay
{
};

/// Fires the request at one replica and returns once its status is known --
/// or once the attempt has provably failed without one. The body keeps
/// arriving on `queue` after this returns.
ReplicaProxy::Attempt
ReplicaProxy::send(const ReplicaSpec &replica, const httplib::Request &request, const std::string &body, const std::string &path, const std::string &sessionId)
{
  httplib::Headers requestHeaders;
  for (const auto &[key, value] : request.headers)
  {
    // Content-Type is set by the relay, and the session header is replaced
    // rather than added to: httplib's headers are a multimap and set_header
    // appends, so copying the client's through would leave two field lines.
    // Two Content-Types make the message malformed for any standards-respecting
    // intermediary, and two session ids let the client echo back the wrong one.
    if (dropFromRequest(key)) { continue; }
    if (lower(key) == "content-type" || lower(key) == lower(std::string(SESSION_HEADER))) { continue; }
    requestHeaders.emplace(key, value);
  }
  // The id this hop resolved, so the next one does not resolve its own and miss
  // the rule that was published for this session.
  requestHeaders.emplace(std::string(SESSION_HEADER), sessionId);

  Attempt attempt;
  static_cast<util::Relay &>(attempt) =
    util::startRelay(replica.host, replica.port, path, body, requestHeaders, _connectTimeoutSeconds, _requestTimeoutSeconds, MAX_CHUNKS_IN_FLIGHT);
  return attempt;
}

void ReplicaProxy::forward(const httplib::Request &request, httplib::Response &response, const std::string &path)
{
  forwardPinned(request, response, path, {}, resolveSessionId(request, request.body));
}

void ReplicaProxy::forwardPinned(const httplib::Request &request,
                                 httplib::Response      &response,
                                 const std::string      &path,
                                 const std::string      &preferredReplica,
                                 const std::string      &sessionId)
{
  const std::string body = request.body;

  ReplicaSpec           replica;
  Attempt               attempt;
  std::function<void()> releaseOnce;

  for (int tries = 0; tries < _maxAttempts; ++tries)
  {
    {
      const std::lock_guard<std::mutex> lock(_registryMutex);
      // The pin applies to the first attempt only; a retry means the chosen hop
      // did not answer, and the registry knows which sibling can.
      const ReplicaState *pinned = tries == 0 && !preferredReplica.empty() ? _registry.state(preferredReplica) : nullptr;
      try
      {
        if (pinned != nullptr && pinned->routable()) { replica = pinned->spec; }
        else { replica = _registry.select(sessionId).replica; }
      }
      catch (const NoReplicaAvailable &)
      {
        // 503 only when there was never anywhere to go. Once we have actually
        // tried a replica the failure is an upstream one, and 502 says so --
        // the two are different enough to be worth telling apart in a log.
        if (tries == 0) { sendError(response, 503, "no serving replica is currently routable", sessionId); }
        else { sendError(response, 502, "no replica of this partition could be reached", sessionId); }
        return;
      }
      _registry.acquire(replica.name);
    }

    // Released exactly once, however the relay ends.
    auto              released = std::make_shared<std::once_flag>();
    const std::string held     = replica.name;
    releaseOnce                = [this, held, released] {
      std::call_once(*released, [this, &held] {
        const std::lock_guard<std::mutex> lock(_registryMutex);
        _registry.release(held);
      });
    };

    attempt = send(replica, request, body, path, sessionId);

    if (!attempt.unreachable()) { break; }

    // Nothing has reached the client, so another replica may still answer this
    // request. Take this one out of rotation first: `select` then skips it, and
    // the session's pin moves with it.
    releaseOnce();
    {
      const std::lock_guard<std::mutex> lock(_registryMutex);
      _registry.setReady(replica.name, false);
    }
    attempt.queue->close();

    if (tries + 1 < _maxAttempts) { _resends.fetch_add(1); }
  }

  auto queue           = attempt.queue;
  auto upstreamStatus  = attempt.status;
  auto upstreamHeaders = attempt.headers;
  auto transportFailed = attempt.transportFailed;

  if (attempt.unreachable())
  {
    // Every attempt was spent and none of them answered.
    sendError(response, 502, "replica " + replica.name + " is unreachable", sessionId);
    return;
  }

  response.status         = upstreamStatus->load();
  std::string contentType = "application/json";
  for (const auto &[key, value] : *upstreamHeaders)
  {
    if (lower(key) == "content-type") { contentType = value; }
    // The session header is set once, below. An upstream that set its own
    // would otherwise reach the client as a second, different value.
    if (!dropFromResponse(key) && lower(key) != "content-type" && lower(key) != lower(std::string(SESSION_HEADER)))
    {
      response.set_header(key.c_str(), value.c_str());
    }
  }
  // Headers precede the body, so this is the only mechanism that returns the
  // session id on a streaming response without rewriting SSE.
  response.set_header(SESSION_HEADER, sessionId);

  // httplib stops invoking the provider once the peer is gone, so a disconnect
  // between chunks would otherwise never reach the two lines below: the upstream
  // thread would block on a full queue for ever and the replica would keep its
  // slot in `outstanding`. Destroying the closure releases either way.
  std::shared_ptr<void> closeOnDestroy(nullptr, [queue, releaseOnce](void *) {
    queue->close();
    releaseOnce();
  });

  auto              truncated   = &_truncated;
  const std::string replicaName = replica.name;
  response.set_chunked_content_provider(contentType, [queue, releaseOnce, closeOnDestroy, transportFailed, truncated, replicaName](size_t, httplib::DataSink &sink) {
    auto chunk = queue->pop();
    if (!chunk.has_value())
    {
      releaseOnce();
      if (transportFailed->load())
      {
        // The replica died part-way through its answer. Terminating the
        // chunked body here would hand the client a short answer that is
        // framed as a complete one, which is worse than a visible failure --
        // returning false drops the connection instead, so every client sees
        // a truncated transfer whether or not it parses the payload.
        truncated->fetch_add(1);
        // Said out loud: a client may or may not notice a broken transfer --
        // curl reports it, a urllib line-iterator does not -- and an operator
        // who is never told is exactly the fault the deck names in the Python
        // stack.
        std::fprintf(stderr, "[router] answer cut short for %s: the replica stopped mid-stream\n", replicaName.c_str());
        return false;
      }
      sink.done();
      return true;
    }
    if (!sink.write(chunk->data(), chunk->size()))
    {
      // The client went away. Closing the queue makes the upstream receiver
      // return false, which closes the connection to the replica -- serving's
      // only cancellation path.
      queue->close();
      releaseOnce();
      return false;
    }
    return true;
  });
}

} // namespace serving::router
