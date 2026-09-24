#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <unistd.h>

#include <gtest/gtest.h>

#include <serving/coordinator/config_file.hpp>

using serving::coordinator::ConfigError;
using serving::coordinator::Coordinator;
using serving::coordinator::CoordinatorConfig;
using serving::coordinator::loadCoordinatorConfig;
using serving::coordinator::parseCoordinatorConfig;
using serving::router::ReplicaSpec;
using serving::router::RoutingMode;
using serving::router::RuleTransport;

namespace
{

/// A three-partition pipeline, two replicas each, as the file would spell it.
const char *kPipeline = R"json({
  "routing_mode": "rules",
  "partition": 1,
  "routing": { "session_ttl_seconds": 120, "affinity_slack": 2 },
  "replicas": [
    { "name": "p0a", "host": "10.0.0.1", "port": 8000, "partition": 0, "instance": 10 },
    { "name": "p0b", "host": "10.0.0.1", "port": 8001, "partition": 0, "instance": 11 },
    { "name": "p1a", "host": "10.0.0.2", "port": 8000, "partition": 1, "instance": 20 },
    { "name": "p1b", "host": "10.0.0.2", "port": 8001, "partition": 1, "instance": 21, "scheme": "https" },
    { "name": "p2a", "host": "10.0.0.3", "port": 8000, "partition": 2, "instance": 30 },
    { "name": "p2b", "host": "10.0.0.3", "port": 8001, "partition": 2, "instance": 31 }
  ]
})json";

/// The error message, so a test can assert on what the operator will read.
std::string errorOf(const std::string &json)
{
  try
  {
    (void)parseCoordinatorConfig(json, "test.json");
  }
  catch (const ConfigError &e)
  {
    return e.what();
  }
  return "";
}

class RecordingTransport : public RuleTransport
{
  public:

  void publish(const ReplicaSpec &, const std::string &, const ReplicaSpec &, uint64_t) override { published += 1; }
  void revoke(const std::string &, uint64_t) override {}
  int  published = 0;
};

class TempFile
{
  public:

  explicit TempFile(const std::string &contents)
    : _path(std::filesystem::temp_directory_path() / ("serving-config-test-" + std::to_string(::getpid()) + "-" + std::to_string(++_counter) + ".json"))
  {
    std::ofstream(_path) << contents;
  }
  ~TempFile() { std::filesystem::remove(_path); }

  [[nodiscard]] const std::filesystem::path &path() const { return _path; }

  private:

  static inline int     _counter = 0;
  std::filesystem::path _path;
};

} // namespace

// ---------------------------------------------------------------------------
// What a valid file yields
// ---------------------------------------------------------------------------

TEST(ConfigFileTest, ParsesAFullDeployment)
{
  const CoordinatorConfig config = parseCoordinatorConfig(kPipeline);

  EXPECT_EQ(config.mode, RoutingMode::DistributedRules);
  EXPECT_EQ(config.partition, 1);
  EXPECT_DOUBLE_EQ(config.routing.sessionTtlSeconds, 120.0);
  EXPECT_EQ(config.routing.affinitySlack, 2);

  ASSERT_EQ(config.replicas.size(), 6u);
  const auto &p1b = config.replicas[3];
  EXPECT_EQ(p1b.spec.name, "p1b");
  EXPECT_EQ(p1b.spec.host, "10.0.0.2");
  EXPECT_EQ(p1b.spec.port, 8001);
  EXPECT_EQ(p1b.spec.partition, 1);
  EXPECT_EQ(p1b.spec.scheme, "https");
  EXPECT_EQ(p1b.instanceId, 21u);
  EXPECT_EQ(p1b.spec.baseUrl(), "https://10.0.0.2:8001");
}

TEST(ConfigFileTest, AppliesTheDefaults)
{
  const auto config = parseCoordinatorConfig(R"({"routing_mode": "ingress", "replicas": [{"name": "a", "host": "h", "port": 1}]})");

  EXPECT_EQ(config.mode, RoutingMode::CoordinatorIngress);
  EXPECT_EQ(config.partition, 0) << "a single-partition deployment need not say so";
  EXPECT_DOUBLE_EQ(config.routing.sessionTtlSeconds, serving::router::RoutingConfig{}.sessionTtlSeconds);
  EXPECT_EQ(config.routing.affinitySlack, serving::router::RoutingConfig{}.affinitySlack);
  ASSERT_EQ(config.replicas.size(), 1u);
  EXPECT_EQ(config.replicas[0].spec.scheme, "http");
  EXPECT_EQ(config.replicas[0].spec.partition, 0);
  EXPECT_EQ(config.replicas[0].instanceId, 0u);
}

TEST(ConfigFileTest, TheFileSelectsTheStrategy)
{
  // The same deployment with the other mode builds a coordinator that behaves
  // as that mode.
  const auto run = [](const char *mode) {
    std::string text = kPipeline;
    text.replace(text.find("\"rules\""), 7, std::string("\"") + mode + "\"");

    RecordingTransport transport;
    Coordinator        coordinator(parseCoordinatorConfig(text), &transport);
    coordinator.initialize();
    for (int turn = 0; turn < 3; ++turn) { (void)coordinator.routeFor("chat-1"); }
    return std::pair{coordinator.stats().plansComputed, transport.published};
  };

  const auto [ingressPlans, ingressRules] = run("ingress");
  const auto [rulesPlans, rulesRules]     = run("rules");
  EXPECT_EQ(ingressPlans, 3);
  EXPECT_EQ(ingressRules, 0);
  EXPECT_EQ(rulesPlans, 1);
  EXPECT_EQ(rulesRules, 2);
}

// ---------------------------------------------------------------------------
// What it refuses, and how it says so
// ---------------------------------------------------------------------------

TEST(ConfigFileTest, RoutingModeIsRequiredAndValidated)
{
  EXPECT_NE(errorOf(R"({"replicas": [{"name": "a", "host": "h", "port": 1}]})").find("'routing_mode' is required"), std::string::npos);
  const std::string bad = errorOf(R"({"routing_mode": "fast", "replicas": [{"name": "a", "host": "h", "port": 1}]})");
  EXPECT_NE(bad.find("'routing_mode'"), std::string::npos);
  EXPECT_NE(bad.find("expected 'ingress' or 'rules'"), std::string::npos) << bad;
}

TEST(ConfigFileTest, RejectsUnknownKeysAtEveryLevel)
{
  EXPECT_NE(errorOf(R"({"routing_mode": "ingress", "colour": 1, "replicas": [{"name": "a", "host": "h", "port": 1}]})").find("unknown key 'colour'"), std::string::npos);

  const std::string inReplica = errorOf(R"({"routing_mode": "ingress", "replicas": [{"name": "a", "host": "h", "port": 1, "prot": 1}]})");
  EXPECT_NE(inReplica.find("replica 0"), std::string::npos) << inReplica;
  EXPECT_NE(inReplica.find("unknown key 'prot'"), std::string::npos);

  const std::string inRouting = errorOf(R"({"routing_mode": "ingress", "routing": {"ttl": 1}, "replicas": [{"name": "a", "host": "h", "port": 1}]})");
  EXPECT_NE(inRouting.find("routing"), std::string::npos);
  EXPECT_NE(inRouting.find("unknown key 'ttl'"), std::string::npos);
}

TEST(ConfigFileTest, RefusesTheLaunchersHostsKeyByName)
{
  // Not an anonymous unknown: the operator took it from the router's fleet file
  // and should be told what it means here.
  const std::string msg = errorOf(R"({"routing_mode": "ingress", "hosts": [], "replicas": [{"name": "a", "host": "h", "port": 1}]})");
  EXPECT_NE(msg.find("'hosts'"), std::string::npos);
  EXPECT_NE(msg.find("launcher"), std::string::npos) << msg;
}

TEST(ConfigFileTest, RequiresAtLeastOneReplica)
{
  EXPECT_NE(errorOf(R"({"routing_mode": "ingress"})").find("'replicas' is required"), std::string::npos);
  EXPECT_NE(errorOf(R"({"routing_mode": "ingress", "replicas": []})").find("at least one replica"), std::string::npos);
  EXPECT_NE(errorOf(R"({"routing_mode": "ingress", "replicas": {}})").find("must be a list"), std::string::npos);
}

TEST(ConfigFileTest, ReplicaErrorsNameTheEntry)
{
  const std::string msg = errorOf(R"({"routing_mode": "ingress", "replicas": [
    {"name": "a", "host": "h", "port": 1},
    {"name": "b", "host": "h"}
  ]})");
  EXPECT_NE(msg.find("test.json: replica 1: 'port' is required"), std::string::npos) << msg;
}

TEST(ConfigFileTest, RejectsABadPort)
{
  for (const char *port : {"0", "65536", "\"8000\"", "8000.5", "true", "-1"})
  {
    const std::string msg = errorOf(std::string(R"({"routing_mode": "ingress", "replicas": [{"name": "a", "host": "h", "port": )") + port + "}]}");
    EXPECT_NE(msg.find("'port'"), std::string::npos) << "port " << port << " accepted: " << msg;
    EXPECT_NE(msg.find("1..65535"), std::string::npos) << msg;
  }
}

TEST(ConfigFileTest, RejectsDuplicateNamesAndEmptyStrings)
{
  EXPECT_NE(errorOf(R"({"routing_mode": "ingress", "replicas": [{"name": "a", "host": "h", "port": 1}, {"name": "a", "host": "h", "port": 2}]})").find("duplicate name 'a'"),
            std::string::npos);
  EXPECT_NE(errorOf(R"({"routing_mode": "ingress", "replicas": [{"name": "", "host": "h", "port": 1}]})").find("'name' must not be empty"), std::string::npos);
  EXPECT_NE(errorOf(R"({"routing_mode": "ingress", "replicas": [{"name": "a", "host": "", "port": 1}]})").find("'host' must not be empty"), std::string::npos);
}

TEST(ConfigFileTest, RejectsABadScheme)
{
  EXPECT_NE(errorOf(R"({"routing_mode": "ingress", "replicas": [{"name": "a", "host": "h", "port": 1, "scheme": "ftp"}]})").find("'scheme' must be 'http' or 'https'"),
            std::string::npos);
}

TEST(ConfigFileTest, RejectsNegativeCounts)
{
  EXPECT_NE(errorOf(R"({"routing_mode": "ingress", "partition": -1, "replicas": [{"name": "a", "host": "h", "port": 1}]})").find("'partition'"), std::string::npos);
  EXPECT_NE(errorOf(R"({"routing_mode": "ingress", "replicas": [{"name": "a", "host": "h", "port": 1, "partition": -2}]})").find("'partition'"), std::string::npos);
  EXPECT_NE(errorOf(R"({"routing_mode": "ingress", "replicas": [{"name": "a", "host": "h", "port": 1, "instance": -3}]})").find("'instance'"), std::string::npos);
}

TEST(ConfigFileTest, TheCoordinatorsPartitionMustHaveReplicas)
{
  const std::string msg = errorOf(R"({"routing_mode": "ingress", "partition": 7, "replicas": [{"name": "a", "host": "h", "port": 1, "partition": 0}]})");
  EXPECT_NE(msg.find("'partition' is 7 but no replica belongs to that partition"), std::string::npos) << msg;
}

TEST(ConfigFileTest, ValidatesTheRoutingKnobs)
{
  EXPECT_NE(errorOf(R"({"routing_mode": "ingress", "routing": {"session_ttl_seconds": 0}, "replicas": [{"name": "a", "host": "h", "port": 1}]})")
              .find("'session_ttl_seconds' must be positive"),
            std::string::npos);
  EXPECT_NE(errorOf(R"({"routing_mode": "ingress", "routing": {"session_ttl_seconds": "long"}, "replicas": [{"name": "a", "host": "h", "port": 1}]})")
              .find("'session_ttl_seconds' must be a number"),
            std::string::npos);
  EXPECT_NE(errorOf(R"({"routing_mode": "ingress", "routing": {"affinity_slack": -1}, "replicas": [{"name": "a", "host": "h", "port": 1}]})").find("'affinity_slack'"),
            std::string::npos);
  EXPECT_NE(errorOf(R"({"routing_mode": "ingress", "routing": [], "replicas": [{"name": "a", "host": "h", "port": 1}]})").find("routing: must be an object"), std::string::npos);
}

TEST(ConfigFileTest, RejectsWhatIsNotAnObject)
{
  EXPECT_NE(errorOf("[]").find("must be a JSON object"), std::string::npos);
  EXPECT_NE(errorOf("\"rules\"").find("must be a JSON object"), std::string::npos);
  EXPECT_NE(errorOf("{not json").find("not valid JSON"), std::string::npos);
}

TEST(ConfigFileTest, ErrorsNameTheSource)
{
  EXPECT_EQ(errorOf("[]").rfind("test.json: ", 0), 0u) << "every message starts with the source";
  try
  {
    (void)parseCoordinatorConfig("[]", "/etc/pypto/coordinator.json");
    FAIL() << "expected ConfigError";
  }
  catch (const ConfigError &e)
  {
    EXPECT_EQ(std::string(e.what()).rfind("/etc/pypto/coordinator.json: ", 0), 0u);
  }
}

// ---------------------------------------------------------------------------
// Reading from disk
// ---------------------------------------------------------------------------

TEST(ConfigFileTest, LoadsFromAFile)
{
  const TempFile file(kPipeline);
  const auto     config = loadCoordinatorConfig(file.path());
  EXPECT_EQ(config.mode, RoutingMode::DistributedRules);
  EXPECT_EQ(config.replicas.size(), 6u);
}

TEST(ConfigFileTest, LoadErrorsNameThePath)
{
  const std::filesystem::path missing = std::filesystem::temp_directory_path() / "serving-config-test-does-not-exist.json";
  try
  {
    (void)loadCoordinatorConfig(missing);
    FAIL() << "expected ConfigError";
  }
  catch (const ConfigError &e)
  {
    EXPECT_NE(std::string(e.what()).find(missing.string()), std::string::npos);
    EXPECT_NE(std::string(e.what()).find("cannot read"), std::string::npos);
  }

  const TempFile broken("{\"routing_mode\": ");
  try
  {
    (void)loadCoordinatorConfig(broken.path());
    FAIL() << "expected ConfigError";
  }
  catch (const ConfigError &e)
  {
    EXPECT_NE(std::string(e.what()).find(broken.path().string()), std::string::npos);
    EXPECT_NE(std::string(e.what()).find("not valid JSON"), std::string::npos);
  }
}

TEST(ConfigFileTest, RulesModeRequiresAnInstancePerReplica)
{
  // Left out, instanceId defaults to 0 and a published rule addresses whichever
  // replica happens to be instance 0 -- a deployment that looks like it works.
  const std::string msg = errorOf(R"({"routing_mode": "rules", "replicas": [{"name": "a", "host": "h", "port": 1}]})");
  EXPECT_NE(msg.find("'instance' is required when routing_mode is 'rules'"), std::string::npos) << msg;

  // ingress publishes nothing, so it does not need one.
  EXPECT_EQ(errorOf(R"({"routing_mode": "ingress", "replicas": [{"name": "a", "host": "h", "port": 1}]})"), "");

  const std::string dup = errorOf(R"({"routing_mode": "rules", "replicas": [
    {"name": "a", "host": "h", "port": 1, "instance": 7},
    {"name": "b", "host": "h", "port": 2, "instance": 7}]})");
  EXPECT_NE(dup.find("duplicate instance 7"), std::string::npos) << dup;
}

TEST(ConfigFileTest, PartitionsMustFormAChainWithNoGaps)
{
  // {0, 2} plans a two-hop path that skips stage 1 -- routing around a
  // partition rather than through it.
  const std::string msg = errorOf(R"({"routing_mode": "ingress", "partition": 0, "replicas": [
    {"name": "a", "host": "h", "port": 1, "partition": 0},
    {"name": "c", "host": "h", "port": 2, "partition": 2}]})");
  EXPECT_NE(msg.find("partitions must be 0..1 with no gaps"), std::string::npos) << msg;

  EXPECT_EQ(errorOf(R"({"routing_mode": "ingress", "partition": 0, "replicas": [
    {"name": "a", "host": "h", "port": 1, "partition": 0},
    {"name": "b", "host": "h", "port": 2, "partition": 1}]})"),
            "");
}

TEST(ConfigFileTest, RejectsTwoReplicasOnTheSameEndpoint)
{
  const std::string msg = errorOf(R"({"routing_mode": "ingress", "replicas": [
    {"name": "a", "host": "h", "port": 1},
    {"name": "b", "host": "h", "port": 1}]})");
  EXPECT_NE(msg.find("duplicate endpoint http://h:1"), std::string::npos) << msg;
}
