#pragma once

/**
 * The router's HTTP surface and health poller. Links neither Python nor a
 * tokenizer.
 */

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <serving/router/proxy.hpp>
#include <serving/router/routing.hpp>

namespace httplib
{
class Server;
}

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
  double sessionTtlSeconds     = 1800.0;
  int    affinitySlack         = 8;

  /// Start the background health poller. A test that drives `probeOnce()`
  /// itself turns this off: two probers racing means an in-flight cycle can
  /// land after a manual one and reset a replica's failure count. Production
  /// has exactly one poller, so its cycles never overlap.
  bool startHealthPoller = true;
};

class RouterServer
{
  public:

  RouterServer(RouterServerConfig config, const std::vector<ReplicaSpec> &replicas);
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

  private:

  void registerRoutes();
  void healthLoop();

  RouterServerConfig _config;
  SessionDirectory   _sessions;
  ReplicaRegistry    _registry;
  std::mutex         _registryMutex;
  ReplicaProxy       _proxy;

  std::unique_ptr<httplib::Server> _server;
  std::thread                      _serverThread;
  std::thread                      _healthThread;

  std::mutex              _wakeMutex;
  std::condition_variable _wakeCv;
  std::atomic<bool>       _running{false};
  std::atomic<int>        _boundPort{0};
};

} // namespace serving::router
