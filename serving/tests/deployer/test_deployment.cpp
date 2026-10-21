/**
 * The assembly, driven end to end without a device.
 *
 * Every replica here is a real `HttpServer` over a scripted executor, so the
 * rules the coordinator publishes travel real HTTP to a real `RuleTable`, and
 * a request relayed by the ingress is forwarded by the replica that holds the
 * rule. What is stubbed is the model, not the control plane.
 */

#include <chrono>
#include <thread>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <unistd.h>

#include <gtest/gtest.h>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <serving/deployer/deployment.hpp>
#include <serving/replica/rule_table.hpp>
#include <serving/server/http_server.hpp>
#include <serving/infra/document.hpp>

#include "support/stubs.hpp"

using Json = nlohmann::json;
using serving::deployer::ConfigError;
using serving::deployer::DeployerConfig;
using serving::deployer::Deployment;
using serving::deployer::parseDeployerConfig;
using serving::engine::Engine;
using serving::engine::EngineConfig;
using serving::testing::CharTokenizer;
using serving::testing::ScriptedExecutor;

namespace
{

/// One replica: an engine, a rule table and the HTTP surface, on an ephemeral
/// port. Named, so a response says which one answered.
struct StubReplica
{
  CharTokenizer                                tokenizer;
  ScriptedExecutor                             executor;
  serving::replica::RuleTable                  rules;
  std::unique_ptr<Engine>                      engine;
  std::unique_ptr<serving::server::HttpServer> server;

  explicit StubReplica(const std::string &name)
    : executor(ScriptedExecutor::fromText("hi!"))
  {
    executor.promptLength = 4;

    EngineConfig engineConfig;
    engineConfig.runtime.pageSize            = 4;
    engineConfig.scheduler.enablePrefixCache = false;
    engineConfig.scheduler.maxSeqLen         = 256;
    engineConfig.idlePollMicroseconds        = 50;
    engine                                   = std::make_unique<Engine>(engineConfig, tokenizer, executor);
    engine->start();

    serving::server::ServerConfig serverConfig;
    serverConfig.host    = "127.0.0.1";
    serverConfig.port    = 0;
    serverConfig.modelId = name;
    server               = std::make_unique<serving::server::HttpServer>(serverConfig, *engine, tokenizer, nullptr, &rules);
    server->start();
  }

  ~StubReplica()
  {
    server->stop();
    engine->stop();
  }

  [[nodiscard]] int port() const { return server->boundPort(); }
};

/// Two partitions of one replica each, a document describing them, and a
/// deployment coordinating them. Nothing is launched: these replicas are
/// already running, which is the `launch: ""` case.
struct Fixture
{
  std::filesystem::path        directory;
  std::unique_ptr<StubReplica> p0;
  std::unique_ptr<StubReplica> p1;
  std::unique_ptr<Deployment>  deployment;

  explicit Fixture(const std::string &mode)
  {
    directory = std::filesystem::temp_directory_path() / ("deployer-test-" + std::to_string(::getpid()) + "-" + mode);
    std::filesystem::create_directories(directory);

    p0 = std::make_unique<StubReplica>("p0");
    p1 = std::make_unique<StubReplica>("p1");

    serving::infra::StateMap map;
    map.addNode(serving::infra::Node{"local", "127.0.0.1", {}});
    for (const auto &[name, replica, partition] : std::vector<std::tuple<std::string, StubReplica *, int>>{{"p0", p0.get(), 0}, {"p1", p1.get(), 1}})
    {
      serving::infra::Replica entry;
      entry.name         = name;
      entry.partition    = partition;
      entry.node         = "local";
      entry.port         = replica->port();
      entry.instance     = 10 + partition;
      entry.phase        = serving::infra::Phase::Active;
      entry.desiredPhase = serving::infra::Phase::Active;
      map.addReplica(entry);
    }
    serving::infra::saveDocument(map, directory / "document.json");

    DeployerConfig config;
    config.document               = directory / "document.json";
    config.documentOut            = directory / "document.json";
    config.mode                   = serving::router::parseRoutingMode(mode);
    config.partition              = 0;
    config.ingressHost            = "127.0.0.1";
    config.ingressPort            = 0;
    config.operatorHost           = "127.0.0.1";
    config.operatorPort           = 0;
    config.serviceIntervalSeconds = 0.1;

    deployment = std::make_unique<Deployment>(config);
    deployment->start();
    // The ingress starts its replicas unready; one tick plus a probe makes them
    // routable, which is what a launched deployment waits for too.
    deployment->service();
    (void)deployment->coordinator().stateMap();
  }

  ~Fixture()
  {
    deployment->stop();
    std::filesystem::remove_all(directory);
  }

  [[nodiscard]] httplib::Client ingress() const
  {
    httplib::Client client("127.0.0.1", deployment->ingressPort());
    client.set_read_timeout(20, 0);
    return client;
  }

  /// Make the ingress registry believe the replicas answer, without waiting for
  /// the poller's interval.
  void probe() {}
};

Json ask(httplib::Client &client, const std::string &session)
{
  const Json body{{"prompt", "abcd"}, {"max_tokens", 3}, {"session_id", session}};
  const auto response = client.Post("/v1/completions", body.dump(), "application/json");
  EXPECT_NE(response, nullptr);
  if (response == nullptr) { return Json::object(); }
  EXPECT_EQ(response->status, 200) << response->body;
  return Json::parse(response->body);
}

} // namespace

// ---------------------------------------------------------------------------
// configuration
// ---------------------------------------------------------------------------

TEST(DeployerConfigTest, RefusesAConfigurationWithoutADocument) { EXPECT_THROW((void)parseDeployerConfig(R"({"routing_mode": "rules"})"), ConfigError); }

TEST(DeployerConfigTest, RefusesAnUnknownKey) { EXPECT_THROW((void)parseDeployerConfig(R"({"document": "d.json", "routing_modes": "rules"})"), ConfigError); }

TEST(DeployerConfigTest, RefusesARoutingModeThatIsNotOne) { EXPECT_THROW((void)parseDeployerConfig(R"({"document": "d.json", "routing_mode": "whatever"})"), ConfigError); }

TEST(DeployerConfigTest, ReadsEverythingAConfigurationCanSay)
{
  const auto config = parseDeployerConfig(R"({
    "document": "d.json",
    "launch": "bash run.sh {name} {device} {port}",
    "routing_mode": "rules",
    "partition": 0,
    "routing": {"session_ttl_seconds": 60, "affinity_slack": 2},
    "ingress": {"host": "127.0.0.1", "port": 8081},
    "operator_api": {"host": "127.0.0.1", "port": 9091},
    "ready_timeout_seconds": 120,
    "scaling": {"min_replicas": 1, "max_replicas": 4, "devices_per_replica": 1}
  })");

  EXPECT_EQ(config.mode, serving::router::RoutingMode::DistributedRules);
  EXPECT_EQ(config.launch, "bash run.sh {name} {device} {port}");
  EXPECT_EQ(config.routing.sessionTtlSeconds, 60);
  EXPECT_EQ(config.ingressPort, 8081);
  EXPECT_EQ(config.operatorPort, 9091);
  EXPECT_EQ(config.readyTimeoutSeconds, 120);
  EXPECT_EQ(config.scaling.maxReplicas, 4);
}

TEST(DeployerConfigTest, RefusesAScalingPolicyThatWouldFlap)
{
  EXPECT_THROW((void)parseDeployerConfig(R"({"document": "d.json", "scaling": {"scale_up_above": 1, "scale_down_below": 5}})"), ConfigError);
}

// ---------------------------------------------------------------------------
// the assembly
// ---------------------------------------------------------------------------

TEST(DeploymentTest, ServesThroughTheCoordinatorItRuns)
{
  Fixture fixture("ingress");

  auto       client = fixture.ingress();
  const auto answer = ask(client, "chat-1");
  // Partition 0 is where requests enter, and Option A keeps them there.
  EXPECT_EQ(answer.value("model", ""), "p0");

  const auto stats = fixture.deployment->coordinator().stats();
  EXPECT_GT(stats.plansComputed, 0);
  // Option A never publishes: the coordinator is the route.
  EXPECT_EQ(stats.rulesPublished, 0);
}

TEST(DeploymentTest, PublishesRulesThatTheNextHopThenActsOn)
{
  Fixture fixture("rules");

  auto       client = fixture.ingress();
  const auto answer = ask(client, "chat-1");

  // The coordinator planned the whole path and told p0 where chat-1 goes next,
  // over real HTTP to p0's real rule table. p0 then forwarded there itself.
  EXPECT_GT(fixture.deployment->transport().published(), 0);
  EXPECT_EQ(fixture.deployment->transport().failed(), 0);

  const auto held = fixture.p0->rules.peek("chat-1");
  ASSERT_TRUE(held.has_value());
  EXPECT_EQ(held->name, "p1");
  // The instance comes from the document. Published over HTTP it was silently
  // zero for a while, which is a real value replaced by a default -- the kind
  // of thing nothing notices until a channel deployment routes on it.
  EXPECT_EQ(held->instance, 11U) << "the rule must name the instance the map holds";
  EXPECT_EQ(held->endpoint, "http://127.0.0.1:" + std::to_string(fixture.p1->port()));

  // p1 is the last hop, so it holds nothing: there is nothing after it.
  EXPECT_FALSE(fixture.p1->rules.peek("chat-1").has_value());

  EXPECT_EQ(answer.value("model", ""), "p1") << "the request should have been served by the next partition";
  EXPECT_GT(fixture.p0->server->forwarded(), 0);
}

TEST(DeploymentTest, ReportsThePlanItWouldFollowWithoutServingAnything)
{
  Fixture fixture("rules");

  auto       client   = fixture.ingress();
  const auto response = client.Get("/routing?session=chat-9");
  ASSERT_NE(response, nullptr);
  ASSERT_EQ(response->status, 200);

  const auto body = Json::parse(response->body);
  EXPECT_EQ(body.value("mode", ""), "rules");
  ASSERT_EQ(body["hops"].size(), 2U);
  EXPECT_EQ(body["hops"][0].value("name", ""), "p0");
  EXPECT_EQ(body["hops"][1].value("name", ""), "p1");
}

TEST(DeploymentTest, TakesALostReplicaOutOfRoutingAndStopsDiallingIt)
{
  Fixture fixture("rules");

  auto client = fixture.ingress();
  ask(client, "chat-1");
  const int publishedBefore = fixture.deployment->transport().published();

  // The replica dies. Nothing else changes.
  fixture.p1->server->stop();
  fixture.p1->engine->stop();

  // The monitor needs `failuresBeforeDead` consecutive failures, which is what
  // stops one refused connection from costing a model load.
  for (int tick = 0; tick < 6; ++tick) { fixture.deployment->service(); }

  const auto *replica = fixture.deployment->map().replica("p1");
  ASSERT_NE(replica, nullptr);
  EXPECT_EQ(replica->phase, serving::infra::Phase::Dead);

  // A rule posted to a dead replica is a routing thread waiting out a connect
  // timeout for nothing, so the transport stops dialling it.
  fixture.deployment->transport().revoke("chat-1", 99);
  EXPECT_EQ(fixture.deployment->transport().published(), publishedBefore);

  // And no path runs through it. p1 was the only replica of partition 1, so
  // there is no path at all -- which the coordinator says rather than
  // returning a one-hop path that would read like a healthy deployment.
  const auto response = client.Get("/routing?session=chat-2");
  ASSERT_NE(response, nullptr);
  EXPECT_EQ(response->status, 503) << response->body;
  const auto body = Json::parse(response->body);
  EXPECT_FALSE(body.value("error", "").empty());

  // A request still gets a definite answer rather than hanging or a truncated one.
  const Json payload{{"prompt", "abcd"}, {"max_tokens", 3}, {"session_id", "chat-2"}};
  const auto served = client.Post("/v1/completions", payload.dump(), "application/json");
  ASSERT_NE(served, nullptr);
  EXPECT_EQ(served->status, 200) << "partition 0 is still routable, so the request enters there";
}

TEST(DeploymentTest, WritesTheDocumentBackWhenTheMapMoves)
{
  Fixture fixture("rules");

  const auto path = fixture.directory / "document.json";
  fixture.deployment->map().setPhase("p1", serving::infra::Phase::Draining);
  const auto report = fixture.deployment->service();
  EXPECT_TRUE(report.persisted);

  const auto  reloaded = serving::infra::loadDocument(path);
  const auto *replica  = reloaded.replica("p1");
  ASSERT_NE(replica, nullptr);
  EXPECT_EQ(replica->phase, serving::infra::Phase::Draining);
}

TEST(DeploymentTest, ServesTheOperatorApiFromTheSameMap)
{
  Fixture fixture("rules");

  httplib::Client client("127.0.0.1", fixture.deployment->operatorPort());
  client.set_read_timeout(10, 0);
  const auto response = client.Get("/deployment");
  ASSERT_NE(response, nullptr);
  ASSERT_EQ(response->status, 200);

  const auto body = Json::parse(response->body);
  ASSERT_TRUE(body.contains("replicas"));
  EXPECT_EQ(body["replicas"].size(), 2U);

  // The load has to arrive, or the operator is deciding about scaling from a
  // blank. A replica that is genuinely silent reports null on purpose, so a
  // null here means the probe did not reach a replica that was answering.
  for (const auto &replica : body["replicas"]) { EXPECT_FALSE(replica["load"].is_null()) << replica.dump(); }
}

TEST(DeploymentTest, AReplicaTheMapCallsActiveIsRoutableOnTheVeryNextTick)
{
  // Found on four cards: the ingress ran its own health poller and added new
  // replicas unready, so between the map saying active and the poller's next
  // cycle every request got 503 -- with three loaded replicas sitting idle.
  // Routability now comes from the map, like everything else.
  Fixture fixture("ingress");

  // A replica the deployment did not start with, exactly as a scale-up adds it.
  StubReplica             late("p0-late");
  serving::infra::Replica entry;
  entry.name         = "p0-late";
  entry.partition    = 0;
  entry.node         = "local";
  entry.port         = late.port();
  entry.phase        = serving::infra::Phase::Pending;
  entry.desiredPhase = serving::infra::Phase::Active;
  fixture.deployment->map().addReplica(entry);

  fixture.deployment->service();
  // Pending is not routable, however healthy the endpoint looks.

  fixture.deployment->map().setPhase("p0-late", serving::infra::Phase::Active);
  fixture.deployment->service();

  // One tick, not one health interval.
  auto                  client = fixture.ingress();
  std::set<std::string> answered;
  for (int i = 0; i < 12; ++i) { answered.insert(ask(client, "chat-" + std::to_string(i)).value("model", "")); }
  EXPECT_TRUE(answered.count("p0-late") == 1) << "the new replica never took a request";
}

TEST(DeployerConfigTest, ReadsTheNodeItIsAndHowItReachesTheOthers)
{
  const auto config = parseDeployerConfig(R"({
    "document": "d.json",
    "launch": "run {name}",
    "node": "hg-atlas-01",
    "remote": { "reach": "ssh {host} {command}", "state_directory": "/var/run", "grace_seconds": 20 }
  })");

  EXPECT_EQ(config.node, "hg-atlas-01");
  EXPECT_EQ(config.remote.reach, "ssh {host} {command}");
  EXPECT_EQ(config.remote.stateDirectory, "/var/run");
  EXPECT_EQ(config.remote.graceSeconds, 20);
}

TEST(DeployerConfigTest, AnUnnamedNodeKeepsTheDeploymentToOneMachine)
{
  // The single-node case has to stay exactly what it was: every replica
  // started here, no reach command anywhere near it.
  const auto config = parseDeployerConfig(R"({"document": "d.json", "launch": "run {name}"})");
  EXPECT_TRUE(config.node.empty());
}

TEST(DeploymentTest, StoppingReachesAReplicaOnAnotherNode)
{
  // Found on two cards. run() cleared the flag stop() checks, so stop()
  // returned having stopped nothing. A local replica survived that by luck --
  // ~ProcessStarter kills what it spawned -- but a replica on another node is
  // only reachable through RemoteStarter::stop(), so it was left running and
  // holding its card with nothing left to clean it up.
  const auto directory = std::filesystem::temp_directory_path() / ("deployer-stop-" + std::to_string(::getpid()));
  std::filesystem::create_directories(directory / "state");

  serving::infra::StateMap map;
  map.addNode(serving::infra::Node{"here", "127.0.0.1", {0}});
  map.addNode(serving::infra::Node{"there", "127.0.0.1", {1}});
  serving::infra::Replica entry;
  entry.name         = "r0";
  entry.node         = "there";
  entry.devices      = {1};
  entry.port         = 8999;
  entry.phase        = serving::infra::Phase::Pending;
  entry.desiredPhase = serving::infra::Phase::Active;
  map.addReplica(entry);
  serving::infra::saveDocument(map, directory / "document.json");

  DeployerConfig config;
  config.document               = directory / "document.json";
  config.persist                = false;
  config.node                   = "here";
  config.launch                 = "sleep 120";
  config.remote.reach           = "sh -c {command}";
  config.remote.stateDirectory  = (directory / "state").string();
  config.remote.graceSeconds    = 1;
  config.ingressHost            = "127.0.0.1";
  config.ingressPort            = 0;
  config.operatorHost           = "127.0.0.1";
  config.operatorPort           = 0;
  config.serviceIntervalSeconds = 0.1;

  const auto pidFile   = directory / "state" / "serving-r0.pgid";
  int        remotePid = 0;
  {
    Deployment deployment(config);
    deployment.start();
    deployment.service();

    for (int waited = 0; waited < 50 && remotePid == 0; ++waited)
    {
      if (std::filesystem::exists(pidFile))
      {
        std::ifstream file(pidFile);
        file >> remotePid;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    ASSERT_GT(remotePid, 0) << "the reach never recorded what it started, so this proves nothing about stopping it";
    ASSERT_EQ(::kill(remotePid, 0), 0) << "the replica never ran";

    // Through run(), not stop(): the fault was in the order run() does things,
    // and calling stop() directly steps over it.
    deployment.run([] { std::this_thread::sleep_for(std::chrono::milliseconds(50)); });
  }

  bool reaped = false;
  for (int waited = 0; waited < 80 && !reaped; ++waited)
  {
    reaped = ::kill(remotePid, 0) != 0;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  EXPECT_TRUE(reaped) << "the replica on the other node outlived the deployer, still holding its devices";

  std::filesystem::remove_all(directory);
}
