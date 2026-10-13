#pragma once

/**
 * One node running a deployment.
 *
 * Every piece of the control plane existed and nothing put them together: the
 * state map, the launcher, the health monitor, the coordinator and the
 * operator API were each built and tested on their own, and a deployment was
 * assembled by hand in a shell script every time one was needed. This is the
 * assembly.
 *
 * It is the deployer of `infra/document_rpc.hpp` -- the node that started the
 * application and therefore owns the one copy of the document. It:
 *
 *   - reads the document, and writes it back as the deployment changes
 *   - launches and supervises the replicas the document asks for
 *   - watches their health and marks the dead ones
 *   - runs a coordinator, which plans a session's path and, under Option B,
 *     publishes the forwarding rules that let replicas skip it afterwards
 *   - serves the ingress requests enter by, and the operator API
 *
 * It decides nothing the deck does not let it decide: scaling is still a
 * person's call through the operator API, and nothing here runs a scaling
 * policy on its own.
 */

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <serving/coordinator/module.hpp>
#include <serving/infra/capacity.hpp>
#include <serving/infra/deployment_server.hpp>
#include <serving/infra/launcher.hpp>
#include <serving/infra/monitor.hpp>
#include <serving/infra/process_starter.hpp>
#include <serving/infra/scaling.hpp>
#include <serving/infra/state.hpp>
#include <serving/router/http_rule_transport.hpp>
#include <serving/router/router_server.hpp>

namespace serving::deployer
{

/// Anything wrong with a deployer configuration. Mirrors `ConfigError` and
/// `DocumentError`: strict parsing, and the message names the offending key.
class ConfigError : public std::runtime_error
{
  public:

  using std::runtime_error::runtime_error;
};

struct DeployerConfig
{
  /// The document describing what should run. Required: a deployment without
  /// one is a deployment nobody wrote down.
  std::filesystem::path document;

  /// Written back here as the map moves. Empty means the document path itself;
  /// `"none"` disables persisting, which is what a test wants.
  std::filesystem::path documentOut;
  bool                  persist = true;

  /// How a replica is started. `{name}`, `{device}`, `{devices}`, `{port}` and
  /// `{node}` are substituted. Empty means the replicas are already running and
  /// this process only coordinates them.
  std::string launch;

  router::RoutingMode   mode = router::RoutingMode::CoordinatorIngress;
  router::RoutingConfig routing;

  /// The partition the coordinator owns. Requests enter the pipeline here.
  router::PartitionId partition = 0;

  std::string ingressHost = "0.0.0.0";
  int         ingressPort = 8080;

  std::string operatorHost = "0.0.0.0";
  int         operatorPort = 9000;

  /// How long a replica may take to answer its health probe before the attempt
  /// counts as failed. A model load is minutes, so this is not a request timeout.
  double readyTimeoutSeconds = 600.0;

  /// Seconds between control ticks: supervise, monitor, sync routing, persist.
  double serviceIntervalSeconds = 2.0;

  infra::Backoff       backoff{5.0, 60.0, 3};
  infra::MonitorConfig monitor;
  infra::ScalingConfig scaling;

  double connectTimeoutSeconds = 2.0;
  double requestTimeoutSeconds = 300.0;
  int    maxAttempts           = 3;
};

[[nodiscard]] DeployerConfig parseDeployerConfig(const std::string &json, const std::string &source = "<config>");
[[nodiscard]] DeployerConfig loadDeployerConfig(const std::filesystem::path &path);

/// What one control tick did, so the log says something and a test can assert on it.
struct TickReport
{
  std::vector<std::string> started;
  std::vector<std::string> promoted;
  /// The attempt failed and will be retried after the backoff.
  std::vector<std::string> retrying;
  /// Tried enough times, or probed as gone: out of routing either way.
  std::vector<std::string> lost;
  /// Routing changes the tick brought the coordinator to.
  int rephased = 0;
  /// True when the document was rewritten because the map moved.
  bool persisted = false;

  [[nodiscard]] bool empty() const { return started.empty() && promoted.empty() && retrying.empty() && lost.empty() && rephased == 0; }
};

class Deployment
{
  public:

  explicit Deployment(DeployerConfig config);
  ~Deployment();

  Deployment(const Deployment &)            = delete;
  Deployment &operator=(const Deployment &) = delete;

  /// Read the document, build everything, and start serving. Replicas are not
  /// waited for: `service()` brings them up, so the ingress answers 503 until
  /// one is ready rather than blocking a launch for the length of a model load.
  void start();

  /// One control tick. Called on a timer by `run()`, and directly by a test.
  TickReport service();

  /// Tick until stopped by SIGINT/SIGTERM.
  void run();

  void stop();

  /// True once every replica the document wants is serving.
  [[nodiscard]] bool settled() const;

  [[nodiscard]] infra::StateMap           &map() { return _coordinator->stateMap(); }
  [[nodiscard]] coordinator::Coordinator  &coordinator() { return *_coordinator; }
  [[nodiscard]] router::HttpRuleTransport &transport() { return *_transport; }
  [[nodiscard]] int                        ingressPort() const;
  [[nodiscard]] int                        operatorPort() const;

  private:

  /// Bring the routing registry and the rule transport to whatever the map now
  /// says. Called on every tick, cheap when nothing moved.
  void refreshRouting();

  DeployerConfig _config;

  std::unique_ptr<router::HttpRuleTransport> _transport;
  std::unique_ptr<coordinator::Coordinator>  _coordinator;

  std::unique_ptr<infra::ProcessStarter>   _starter;
  std::unique_ptr<infra::HttpHealthProbe>  _httpProbe;
  std::unique_ptr<infra::HealthProbe>      _probe;
  std::unique_ptr<infra::Supervisor>       _supervisor;
  std::unique_ptr<infra::HealthMonitor>    _monitor;
  std::unique_ptr<infra::CapacityProvider> _capacity;

  std::unique_ptr<router::RouterServer>    _ingress;
  std::unique_ptr<infra::DeploymentServer> _operator;

  uint64_t _persistedVersion = 0;
  bool     _running          = false;
};

} // namespace serving::deployer
