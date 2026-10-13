#include <string>

#include <gtest/gtest.h>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <serving/infra/deployment_server.hpp>
#include <serving/infra/document.hpp>

using Json = nlohmann::json;
using serving::infra::CapacityRequest;
using serving::infra::DeploymentServer;
using serving::infra::DeploymentServerConfig;
using serving::infra::LocalCapacity;
using serving::infra::Node;
using serving::infra::Phase;
using serving::infra::Replica;
using serving::infra::ScalingConfig;
using serving::infra::StateMap;

namespace
{

StateMap deployment()
{
  StateMap map;
  map.addNode(Node{"hg-atlas-01", "127.0.0.1", {0, 1, 2, 3}});
  Replica first;
  first.name    = "p0a";
  first.node    = "hg-atlas-01";
  first.devices = {0};
  first.phase   = Phase::Active;
  map.addReplica(first);
  return map;
}

ScalingConfig scaling()
{
  ScalingConfig c;
  c.maxReplicas = 4;
  c.minReplicas = 1;
  return c;
}

struct Fixture
{
  StateMap                          map = deployment();
  LocalCapacity                     provider{map};
  std::unique_ptr<DeploymentServer> server;

  Fixture()
  {
    DeploymentServerConfig config;
    config.host                = "127.0.0.1";
    config.port                = 0;
    config.probeTimeoutSeconds = 0.3; // nothing is listening; do not wait on it
    server                     = std::make_unique<DeploymentServer>(config, map, provider, scaling());
    server->start();
  }

  ~Fixture() { server->stop(); }

  [[nodiscard]] httplib::Client client() const
  {
    httplib::Client c("127.0.0.1", server->boundPort());
    c.set_read_timeout(10, 0);
    return c;
  }
};

} // namespace

TEST(DeploymentServerTest, ShowsWhatRunsWhere)
{
  Fixture fixture;
  auto    client = fixture.client();

  const auto response = client.Get("/deployment");
  ASSERT_NE(response, nullptr);
  ASSERT_EQ(response->status, 200);

  const auto body = Json::parse(response->body);
  EXPECT_EQ(body.at("replicas").size(), 1U);
  EXPECT_EQ(body.at("replicas")[0].at("name"), "p0a");
  EXPECT_EQ(body.at("nodes")[0].at("in_use"), Json::array({0}));
  EXPECT_EQ(body.at("nodes")[0].at("free"), Json::array({1, 2, 3}));
  EXPECT_TRUE(body.at("problems").empty());
}

TEST(DeploymentServerTest, ASilentReplicaIsNotReportedAsIdle)
{
  Fixture fixture;
  auto    client = fixture.client();

  // Nothing is listening on the replica's port, so its load is unknown. Zero
  // would read as idle and invite an operator to drain a replica that may be
  // perfectly busy.
  const auto body = Json::parse(client.Get("/deployment")->body);
  EXPECT_TRUE(body.at("replicas")[0].at("load").is_null());
}

TEST(DeploymentServerTest, ServesTheDocumentItself)
{
  Fixture fixture;
  auto    client = fixture.client();

  const auto response = client.Get("/deployment/document");
  ASSERT_NE(response, nullptr);
  const auto reloaded = serving::infra::readDocument(response->body);
  EXPECT_EQ(reloaded.replicas().size(), 1U);
  EXPECT_EQ(reloaded.version(), fixture.map.version());
}

TEST(DeploymentServerTest, AdviceIsReportedNeverActedOn)
{
  Fixture fixture;
  auto    client = fixture.client();

  const auto before = fixture.map.replicas().size();
  const auto body   = Json::parse(client.Get("/deployment/advice")->body);
  ASSERT_TRUE(body.contains("advice"));

  // Looking at the policy's opinion must not enact it.
  EXPECT_EQ(fixture.map.replicas().size(), before);
  EXPECT_EQ(fixture.server->scaled(), 0);
}

TEST(DeploymentServerTest, ScalingUpOnRequestAddsAPendingReplica)
{
  Fixture fixture;
  auto    client = fixture.client();

  const auto response = client.Post("/deployment/scale", Json{{"partition", 0}, {"direction", "up"}}.dump(), "application/json");
  ASSERT_NE(response, nullptr);
  ASSERT_EQ(response->status, 200) << response->body;

  const auto body = Json::parse(response->body);
  EXPECT_TRUE(body.at("applied").get<bool>());
  EXPECT_EQ(fixture.map.replicas().size(), 2U);

  // Pending, not active: it cannot serve until something starts it and its
  // model loads.
  const auto *added = fixture.map.replica(body.at("replica").get<std::string>());
  ASSERT_NE(added, nullptr);
  EXPECT_EQ(added->phase, Phase::Pending);
  EXPECT_EQ(fixture.server->scaled(), 1);
}

TEST(DeploymentServerTest, ScalingDownDrainsRatherThanKills)
{
  Fixture fixture;
  auto    client = fixture.client();

  const auto response = client.Post("/deployment/scale", Json{{"partition", 0}, {"direction", "down"}}.dump(), "application/json");
  ASSERT_NE(response, nullptr);
  ASSERT_EQ(response->status, 200) << response->body;

  // Draining keeps serving what it already took, and keeps its card until it
  // is gone.
  EXPECT_EQ(fixture.map.replica("p0a")->phase, Phase::Draining);
  EXPECT_EQ(fixture.map.devicesInUse("hg-atlas-01"), std::vector<int>{0});
}

TEST(DeploymentServerTest, RefusesInstructionsItCannotCarryOut)
{
  Fixture fixture;
  auto    client = fixture.client();

  EXPECT_EQ(client.Post("/deployment/scale", "not json", "application/json")->status, 400);
  EXPECT_EQ(client.Post("/deployment/scale", Json{{"partition", 0}, {"direction", "sideways"}}.dump(), "application/json")->status, 400);
  EXPECT_EQ(client.Post("/deployment/scale", Json{{"direction", "up"}}.dump(), "application/json")->status, 400);
  // A partition that is not in this deployment.
  EXPECT_EQ(client.Post("/deployment/scale", Json{{"partition", 7}, {"direction", "up"}}.dump(), "application/json")->status, 404);
}

TEST(DeploymentServerTest, AnExhaustedNodeDeclinesRatherThanFails)
{
  Fixture fixture;
  auto    client = fixture.client();

  // Four cards, one taken: three more fit, the fifth does not.
  for (int i = 0; i < 3; ++i) { ASSERT_EQ(client.Post("/deployment/scale", Json{{"partition", 0}, {"direction", "up"}}.dump(), "application/json")->status, 200) << "at " << i; }
  const auto full = client.Post("/deployment/scale", Json{{"partition", 0}, {"direction", "up"}}.dump(), "application/json");
  ASSERT_NE(full, nullptr);
  EXPECT_EQ(full->status, 409);
  EXPECT_FALSE(Json::parse(full->body).at("applied").get<bool>());
  EXPECT_NE(Json::parse(full->body).at("declined").get<std::string>().find("no capacity"), std::string::npos);
}

TEST(DeploymentServerTest, DrainingTheLastServingReplicaIsRefused)
{
  Fixture fixture;
  auto    client = fixture.client();

  ASSERT_EQ(client.Post("/deployment/scale", Json{{"partition", 0}, {"direction", "down"}}.dump(), "application/json")->status, 200);
  // Now nothing is serving; there is nothing left to drain, and saying so is
  // better than draining a replica that is already leaving.
  const auto again = client.Post("/deployment/scale", Json{{"partition", 0}, {"direction", "down"}}.dump(), "application/json");
  ASSERT_NE(again, nullptr);
  EXPECT_EQ(again->status, 409);
}
