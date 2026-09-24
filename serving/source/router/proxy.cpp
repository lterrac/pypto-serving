#include <serving/router/proxy.hpp>

#include <algorithm>
#include <condition_variable>
#include <cctype>
#include <deque>
#include <optional>
#include <future>
#include <memory>
#include <set>
#include <thread>

#include <httplib.h>
#include <nlohmann/json.hpp>

namespace serving::router
{

namespace
{

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

/// Push-to-pull bridge: the upstream thread writes, the response provider reads.
class ChunkQueue
{
  public:

  void push(const char *data, size_t length)
  {
    std::unique_lock<std::mutex> lock(_mutex);
    _cv.wait(lock, [this] { return _chunks.size() < kMaxChunks || _closed; });
    if (_closed) { return; }
    _chunks.emplace_back(data, length);
    lock.unlock();
    _cv.notify_all();
  }

  /// Next chunk, or nullopt once the upstream is finished and drained.
  std::optional<std::string> pop()
  {
    std::unique_lock<std::mutex> lock(_mutex);
    _cv.wait(lock, [this] { return !_chunks.empty() || _closed; });
    if (_chunks.empty()) { return std::nullopt; }
    std::string chunk = std::move(_chunks.front());
    _chunks.pop_front();
    lock.unlock();
    _cv.notify_all();
    return chunk;
  }

  void close()
  {
    {
      const std::lock_guard<std::mutex> lock(_mutex);
      _closed = true;
    }
    _cv.notify_all();
  }

  [[nodiscard]] bool closed() const
  {
    const std::lock_guard<std::mutex> lock(_mutex);
    return _closed;
  }

  private:

  static constexpr size_t kMaxChunks = 256;

  mutable std::mutex      _mutex;
  std::condition_variable _cv;
  std::deque<std::string> _chunks;
  bool                    _closed = false;
};

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

ReplicaProxy::ReplicaProxy(ReplicaRegistry &registry, std::mutex &registryMutex, double connectTimeoutSeconds, double requestTimeoutSeconds)
  : _registry(registry),
    _registryMutex(registryMutex),
    _connectTimeoutSeconds(connectTimeoutSeconds),
    _requestTimeoutSeconds(requestTimeoutSeconds)
{}

void ReplicaProxy::forward(const httplib::Request &request, httplib::Response &response, const std::string &path)
{
  const std::string body      = request.body;
  const std::string sessionId = resolveSessionId(request, body);

  ReplicaSpec replica;
  {
    const std::lock_guard<std::mutex> lock(_registryMutex);
    try
    {
      replica = _registry.select(sessionId).replica;
    }
    catch (const NoReplicaAvailable &)
    {
      sendError(response, 503, "no serving replica is currently routable", sessionId);
      return;
    }
    _registry.acquire(replica.name);
  }

  // Released exactly once, however the relay ends.
  auto released    = std::make_shared<std::once_flag>();
  auto releaseOnce = [this, replica, released] {
    std::call_once(*released, [this, &replica] {
      const std::lock_guard<std::mutex> lock(_registryMutex);
      _registry.release(replica.name);
    });
  };

  auto queue = std::make_shared<ChunkQueue>();

  // The upstream status and headers must be known before the body is relayed,
  // so the thread signals once the response handler has fired.
  auto headersReady    = std::make_shared<std::promise<void>>();
  auto headersFuture   = headersReady->get_future();
  auto upstreamStatus  = std::make_shared<int>(0);
  auto upstreamHeaders = std::make_shared<httplib::Headers>();
  auto transportFailed = std::make_shared<bool>(false);
  auto signalled       = std::make_shared<std::once_flag>();

  const std::string host           = replica.host;
  const int         port           = replica.port;
  const double      connectTimeout = _connectTimeoutSeconds;
  const double      requestTimeout = _requestTimeoutSeconds;

  httplib::Headers upstreamRequestHeaders;
  for (const auto &[key, value] : request.headers)
  {
    if (!dropFromRequest(key)) { upstreamRequestHeaders.emplace(key, value); }
  }

  std::thread upstream([=] {
    httplib::Client client(host, port);
    client.set_connection_timeout(static_cast<time_t>(connectTimeout), 0);
    client.set_read_timeout(static_cast<time_t>(requestTimeout), 0);
    client.set_write_timeout(static_cast<time_t>(requestTimeout), 0);

    httplib::Request upstreamRequest;
    upstreamRequest.method  = "POST";
    upstreamRequest.path    = path;
    upstreamRequest.body    = body;
    upstreamRequest.headers = upstreamRequestHeaders;
    upstreamRequest.set_header("Content-Type", "application/json");

    upstreamRequest.response_handler = [=](const httplib::Response &upstreamResponse) {
      *upstreamStatus  = upstreamResponse.status;
      *upstreamHeaders = upstreamResponse.headers;
      std::call_once(*signalled, [&] { headersReady->set_value(); });
      return true;
    };
    upstreamRequest.content_receiver = [=](const char *data, size_t length, uint64_t, uint64_t) {
      if (queue->closed()) { return false; } // the client hung up; stop pulling
      queue->push(data, length);
      return true;
    };

    httplib::Response upstreamResponse;
    httplib::Error    error = httplib::Error::Success;
    const bool        ok    = client.send(upstreamRequest, upstreamResponse, error);
    if (!ok || error != httplib::Error::Success) { *transportFailed = true; }

    // Unblock the caller even when the handler never fired.
    std::call_once(*signalled, [&] { headersReady->set_value(); });
    queue->close();
  });
  upstream.detach();

  headersFuture.wait();

  if (*transportFailed && *upstreamStatus == 0)
  {
    // Any transport failure means a dead replica: take it out of rotation so the
    // next request does not repeat the wait.
    releaseOnce();
    {
      const std::lock_guard<std::mutex> lock(_registryMutex);
      _registry.setReady(replica.name, false);
    }
    queue->close();
    sendError(response, 502, "replica " + replica.name + " is unreachable", sessionId);
    return;
  }

  response.status         = *upstreamStatus;
  std::string contentType = "application/json";
  for (const auto &[key, value] : *upstreamHeaders)
  {
    if (lower(key) == "content-type") { contentType = value; }
    if (!dropFromResponse(key) && lower(key) != "content-type") { response.set_header(key.c_str(), value.c_str()); }
  }
  // Headers precede the body, so this is the only mechanism that returns the
  // session id on a streaming response without rewriting SSE.
  response.set_header(SESSION_HEADER, sessionId);

  response.set_chunked_content_provider(contentType, [queue, releaseOnce](size_t, httplib::DataSink &sink) {
    auto chunk = queue->pop();
    if (!chunk.has_value())
    {
      releaseOnce();
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
