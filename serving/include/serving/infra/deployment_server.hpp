#pragma once

/**
 * The operator's view of the deployment, and the lever.
 *
 * Scaling is decided by a person, not by a policy running on its own. This
 * exposes what they need to decide -- what runs where and how loaded it is --
 * and the two actions that follow: add a replica to a partition, or drain one.
 *
 * It belongs on the deployer, the node that owns the document, because that is
 * the only place with the whole picture.
 *
 * HTTP, deliberately. This is a person or an operator tool talking to the
 * deployment; inter-node control traffic goes over channels instead, where
 * there is no human and no browser.
 *
 * It decides nothing. `infra/scaling.hpp` can still compute an opinion, and
 * `/deployment/advice` reports it, but nothing acts on that opinion until
 * someone asks.
 */

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <serving/infra/capacity.hpp>
#include <serving/infra/launcher.hpp>
#include <serving/infra/scaling.hpp>
#include <serving/infra/state.hpp>

namespace httplib
{
class Server;
}

namespace serving::infra
{

struct DeploymentServerConfig
{
  std::string host = "0.0.0.0";
  /// 0 binds an ephemeral port; read it back with `boundPort()`.
  int port = 9000;
  /// How long to wait on a replica that is not answering its load endpoint.
  double probeTimeoutSeconds = 2.0;
};

/**
 * Serves the deployment and takes scaling instructions.
 *
 *   GET  /deployment          what runs where, with each replica's load
 *   GET  /deployment/document the document itself, as it would be written
 *   GET  /deployment/advice   what the policy would do, if asked
 *   POST /deployment/scale    {"partition": N, "direction": "up"|"down"}
 *
 * The map, the provider and the supervisor are borrowed, not owned: the
 * deployer already has them, and two copies of a deployment is the problem
 * this whole layer exists to avoid.
 */
class DeploymentServer
{
  public:

  DeploymentServer(DeploymentServerConfig config, StateMap &map, CapacityProvider &provider, ScalingConfig scaling);
  ~DeploymentServer();

  DeploymentServer(const DeploymentServer &)            = delete;
  DeploymentServer &operator=(const DeploymentServer &) = delete;

  /// Bind, then serve on a background thread. Returns once the port is listening.
  void start();
  void stop();

  [[nodiscard]] int boundPort() const { return _boundPort.load(); }

  /// Scale-ups and drains carried out on request.
  [[nodiscard]] int scaled() const { return _scaled.load(); }

  /// Ask every active replica what it is carrying. Exposed because collecting
  /// load is the slow part, and a caller may want it on its own schedule.
  [[nodiscard]] LoadSnapshot collectLoad() const;

  private:

  void registerRoutes();

  DeploymentServerConfig _config;
  StateMap              &_map;
  CapacityProvider      &_provider;
  ScalingConfig          _scaling;

  /// Guards the map against the request threads that read and edit it.
  mutable std::mutex _mutex;

  std::unique_ptr<httplib::Server> _server;
  std::thread                      _thread;
  std::atomic<int>                 _boundPort{0};
  std::atomic<int>                 _scaled{0};
};

} // namespace serving::infra
