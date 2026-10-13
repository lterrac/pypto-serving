#pragma once

/**
 * The router's HTTP surface and health poller. Links neither Python nor a
 * tokenizer.
 *
 * Given a coordinator it becomes the front of a partitioned deployment rather
 * than a flat one: the coordinator plans where a session goes -- publishing
 * forwarding rules if the deployment runs Option B -- and the request enters
 * the pipeline at the first hop of that path. Without one it routes by session
 * affinity across a single partition, which is what it has always done.
 */

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <serving/coordinator/module.hpp>
#include <serving/router/proxy.hpp>
#include <serving/router/routing.hpp>

namespace httplib
{
class Server;
struct Request;
} // namespace httplib

namespace serving::router
{

struct RouterServerConfig
{
  std::string host = "0.0.0.0";
  /// 0 binds an ephemeral port; read it back with `boundPort()`.
  int    port                  = 8080;
  double healthIntervalSeconds = 5.0;
  double connectTimeoutSeconds = 2.0;
  double requestTimeoutSeconds = 300.0;
  /// Replicas a request may be tried on when one is unreachable, counting the
  /// first. 1 disables re-sending. Only ever applied before any of the answer
  /// has reached the client.
  int    maxAttempts       = 3;
  double sessionTtlSeconds = 1800.0;
  int    affinitySlack     = 8;

  /// Start the background health poller. A test that drives `probeOnce()`
  /// itself turns this off: two probers racing means an in-flight cycle can
  /// land after a manual one and reset a replica's failure count. Production
  /// has exactly one poller, so its cycles never overlap.
  bool startHealthPoller = true;
};

class RouterServer
{
  public:

  /// `coordinator` is optional and borrowed; it must outlive the server. The
  /// server never drives its lifecycle -- the deployment that owns it does.
  RouterServer(RouterServerConfig config, const std::vector<ReplicaSpec> &replicas, coordinator::Coordinator *coordinator = nullptr);
  ~RouterServer();

  RouterServer(const RouterServer &)            = delete;
  RouterServer &operator=(const RouterServer &) = delete;

  /// Bind, start the health poller, then serve. Returns once listening.
  void start();
  void stop();

  [[nodiscard]] int boundPort() const { return _boundPort.load(); }

  /// One probe cycle, exposed so a test need not wait for the timer.
  void probeOnce();

  [[nodiscard]] ReplicaRegistry &registry() { return _registry; }
  [[nodiscard]] std::mutex      &registryMutex() { return _registryMutex; }

  /// Requests re-sent to another replica because the first was unreachable.
  [[nodiscard]] int proxyResends() const { return _proxy.resends(); }
  /// Answers cut short by a replica dying mid-stream; these cannot be retried.
  [[nodiscard]] int proxyTruncated() const { return _proxy.truncated(); }

  private:

  void registerRoutes();
  void healthLoop();

  /// The replica the coordinator wants this request to enter at, or empty when
  /// there is no coordinator or it could not plan a path.
  [[nodiscard]] std::string plannedHop(const httplib::Request &request, const std::string &sessionId);

  RouterServerConfig        _config;
  coordinator::Coordinator *_coordinator = nullptr;
  SessionDirectory          _sessions;
  ReplicaRegistry           _registry;
  std::mutex                _registryMutex;
  ReplicaProxy              _proxy;

  std::unique_ptr<httplib::Server> _server;
  std::thread                      _serverThread;
  std::thread                      _healthThread;

  std::mutex              _wakeMutex;
  std::condition_variable _wakeCv;
  std::atomic<bool>       _running{false};
  std::atomic<int>        _boundPort{0};
};

} // namespace serving::router
