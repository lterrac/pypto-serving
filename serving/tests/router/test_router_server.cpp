#include <atomic>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <serving/router/router_server.hpp>

using Json = nlohmann::json;
using serving::router::isUsableSessionId;
using serving::router::ReplicaSpec;
using serving::router::RouterServer;
using serving::router::RouterServerConfig;
using serving::router::SESSION_HEADER;

namespace
{

/**
 * A stand-in for one serving replica: answers /health, /v1/models and
 * /v1/completions, streaming when asked. The Python suite uses an equivalent
 * stub (tests/unit/router/stub_replica.py) for the same reason -- the router's
 * behaviour is what is under test, not a model's.
 */
class StubReplica
{
  public:

  explicit StubReplica(std::string name)
    : _name(std::move(name)),
      _server(std::make_unique<httplib::Server>())
  {
    _server->Get("/health", [this](const httplib::Request &, httplib::Response &response) {
      response.status = healthy.load() ? 200 : 503;
      response.set_content(Json{{"status", healthy.load() ? "ok" : "not_ready"}}.dump(), "application/json");
    });

    _server->Get("/v1/models", [this](const httplib::Request &, httplib::Response &response) {
      response.set_content(Json{{"object", "list"}, {"data", Json::array({Json{{"id", _name}}})}}.dump(), "application/json");
    });

    _server->Post("/v1/completions", [this](const httplib::Request &request, httplib::Response &response) {
      received += 1;
      lastBody = request.body;
      sawHost  = request.has_header("Host");

      bool stream = false;
      try
      {
        const auto body = Json::parse(request.body);
        stream          = body.value("stream", false);
      }
      catch (const std::exception &)
      {}

      if (!stream)
      {
        response.set_content(Json{{"replica", _name}, {"object", "text_completion"}}.dump(), "application/json");
        return;
      }
      auto sent = std::make_shared<int>(0);
      response.set_chunked_content_provider("text/event-stream", [this, sent](size_t, httplib::DataSink &sink) {
        if (*sent >= 3)
        {
          const std::string done = "data: [DONE]\n\n";
          sink.write(done.data(), done.size());
          sink.done();
          return true;
        }
        const std::string chunk = "data: {\"replica\":\"" + _name + "\",\"n\":" + std::to_string(*sent) + "}\n\n";
        sink.write(chunk.data(), chunk.size());
        *sent += 1;
        return true;
      });
    });
  }

  ~StubReplica() { stop(); }

  void start()
  {
    _port   = _server->bind_to_any_port("127.0.0.1");
    _thread = std::thread([this] { _server->listen_after_bind(); });
    _server->wait_until_ready();
  }

  void stop()
  {
    if (_server) { _server->stop(); }
    if (_thread.joinable()) { _thread.join(); }
  }

  [[nodiscard]] ReplicaSpec spec() const
  {
    ReplicaSpec s;
    s.name = _name;
    s.host = "127.0.0.1";
    s.port = _port;
    return s;
  }

  std::atomic<bool> healthy{true};
  std::atomic<int>  received{0};
  std::string       lastBody;
  std::atomic<bool> sawHost{false};

  private:

  std::string                      _name;
  std::unique_ptr<httplib::Server> _server;
  std::thread                      _thread;
  int                              _port = 0;
};

/// Router in front of two stub replicas.
struct Fixture
{
  StubReplica                   a{"replica-a"};
  StubReplica                   b{"replica-b"};
  std::unique_ptr<RouterServer> router;

  Fixture()
  {
    a.start();
    b.start();

    RouterServerConfig config;
    config.host = "127.0.0.1";
    config.port = 0;
    // The tests drive probing themselves, so the background poller stays off:
    // an in-flight background cycle can land after a manual probe and reset the
    // failure count, which made this suite flaky under tsan.
    config.startHealthPoller     = false;
    config.healthIntervalSeconds = 3600.0;
    config.connectTimeoutSeconds = 2.0;
    router                       = std::make_unique<RouterServer>(config, std::vector<ReplicaSpec>{a.spec(), b.spec()});
    router->start();
  }

  ~Fixture()
  {
    router->stop();
    a.stop();
    b.stop();
  }

  [[nodiscard]] httplib::Client client() const
  {
    httplib::Client c("127.0.0.1", router->boundPort());
    c.set_read_timeout(10, 0);
    return c;
  }
};

} // namespace

TEST(SessionIdValidationTest, AcceptsOnlyHeaderSafeIds)
{
  EXPECT_TRUE(isUsableSessionId("abc123"));
  EXPECT_TRUE(isUsableSessionId("a.b_c:d-e"));
  EXPECT_TRUE(isUsableSessionId(std::string(128, 'a')));

  EXPECT_FALSE(isUsableSessionId(""));
  EXPECT_FALSE(isUsableSessionId(std::string(129, 'a')));
  // CR/LF would be a header-injection vector and is what the anchored regex
  // in the Python exists to stop.
  EXPECT_FALSE(isUsableSessionId("abc\r\ndef"));
  EXPECT_FALSE(isUsableSessionId("abc\n"));
  EXPECT_FALSE(isUsableSessionId("with space"));
  EXPECT_FALSE(isUsableSessionId("caf\xC3\xA9"));
}

TEST(RouterServerTest, ReportsItsReplicaTable)
{
  Fixture    fixture;
  auto       client   = fixture.client();
  const auto response = client.Get("/health");
  ASSERT_NE(response, nullptr);
  EXPECT_EQ(response->status, 200);

  const auto body = Json::parse(response->body);
  EXPECT_EQ(body.at("status"), "ok");
  EXPECT_EQ(body.at("routable"), 2);
  EXPECT_EQ(body.at("replicas").size(), 2u);
}

TEST(RouterServerTest, AnswersModelsFromARoutableReplica)
{
  Fixture    fixture;
  auto       client   = fixture.client();
  const auto response = client.Get("/v1/models");
  ASSERT_NE(response, nullptr);
  EXPECT_EQ(response->status, 200);
  EXPECT_NE(response->body.find("replica-"), std::string::npos);
}

TEST(RouterServerTest, ForwardsACompletionAndMintsASessionId)
{
  Fixture fixture;
  auto    client = fixture.client();

  const auto response = client.Post("/v1/completions", Json{{"prompt", "hi"}}.dump(), "application/json");
  ASSERT_NE(response, nullptr);
  ASSERT_EQ(response->status, 200) << response->body;

  // Every response carries a session id, so a client can pin its next turn.
  ASSERT_TRUE(response->has_header(SESSION_HEADER));
  EXPECT_TRUE(isUsableSessionId(response->get_header_value(SESSION_HEADER)));

  EXPECT_EQ(fixture.a.received.load() + fixture.b.received.load(), 1);
}

TEST(RouterServerTest, ForwardsTheBodyVerbatim)
{
  Fixture fixture;
  auto    client = fixture.client();

  // The router never parses or rebuilds the payload; an unknown field must
  // survive the hop untouched.
  const std::string body     = Json{{"prompt", "hi"}, {"something_unknown", 42}}.dump();
  const auto        response = client.Post("/v1/completions", body, "application/json");
  ASSERT_NE(response, nullptr);
  ASSERT_EQ(response->status, 200);

  const std::string relayed = fixture.a.received.load() > 0 ? fixture.a.lastBody : fixture.b.lastBody;
  EXPECT_EQ(relayed, body);
}

TEST(RouterServerTest, KeepsAConversationOnOneReplica)
{
  Fixture fixture;
  auto    client = fixture.client();

  const auto first = client.Post("/v1/completions", Json{{"prompt", "turn 1"}}.dump(), "application/json");
  ASSERT_NE(first, nullptr);
  const std::string sessionId = first->get_header_value(SESSION_HEADER);

  const std::string firstReplica = Json::parse(first->body).at("replica").get<std::string>();

  httplib::Headers headers{{SESSION_HEADER, sessionId}};
  for (int turn = 0; turn < 4; ++turn)
  {
    const auto next = client.Post("/v1/completions", headers, Json{{"prompt", "turn n"}}.dump(), "application/json");
    ASSERT_NE(next, nullptr);
    ASSERT_EQ(next->status, 200);
    // The prefix cache is per replica: this is the whole reason affinity exists.
    EXPECT_EQ(Json::parse(next->body).at("replica").get<std::string>(), firstReplica);
    EXPECT_EQ(next->get_header_value(SESSION_HEADER), sessionId) << "the session id must be echoed back";
  }
}

TEST(RouterServerTest, TakesTheSessionIdFromTheBodyWhenTheHeaderIsAbsent)
{
  Fixture fixture;
  auto    client = fixture.client();

  const auto first = client.Post("/v1/completions", Json{{"prompt", "x"}, {"session_id", "from-body"}}.dump(), "application/json");
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(first->get_header_value(SESSION_HEADER), "from-body");

  const std::string firstReplica = Json::parse(first->body).at("replica").get<std::string>();
  const auto        second       = client.Post("/v1/completions", Json{{"prompt", "y"}, {"session_id", "from-body"}}.dump(), "application/json");
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(Json::parse(second->body).at("replica").get<std::string>(), firstReplica);
}

TEST(RouterServerTest, ReplacesAnUnusableSessionIdRatherThanFailing)
{
  Fixture fixture;
  auto    client = fixture.client();

  // The id only decides routing, so a bad one must not fail the request.
  httplib::Headers headers{{SESSION_HEADER, "not a valid id"}};
  const auto       response = client.Post("/v1/completions", headers, Json{{"prompt", "x"}}.dump(), "application/json");
  ASSERT_NE(response, nullptr);
  EXPECT_EQ(response->status, 200);
  const std::string minted = response->get_header_value(SESSION_HEADER);
  EXPECT_NE(minted, "not a valid id");
  EXPECT_TRUE(isUsableSessionId(minted));
}

TEST(RouterServerTest, SpreadsDistinctConversationsAcrossReplicas)
{
  Fixture fixture;
  auto    client = fixture.client();

  std::set<std::string> reached;
  for (int i = 0; i < 6; ++i)
  {
    const auto response = client.Post("/v1/completions", Json{{"prompt", "x"}}.dump(), "application/json");
    ASSERT_NE(response, nullptr);
    reached.insert(Json::parse(response->body).at("replica").get<std::string>());
  }
  EXPECT_EQ(reached.size(), 2u) << "distinct sessions should not all land on one replica";
}

TEST(RouterServerTest, RelaysAStreamingResponse)
{
  Fixture fixture;
  auto    client = fixture.client();

  const auto response = client.Post("/v1/completions", Json{{"prompt", "x"}, {"stream", true}}.dump(), "application/json");
  ASSERT_NE(response, nullptr);
  ASSERT_EQ(response->status, 200);

  // Relayed byte for byte, including the SSE framing.
  EXPECT_NE(response->body.find("data: {"), std::string::npos);
  EXPECT_NE(response->body.find("data: [DONE]"), std::string::npos);
  EXPECT_EQ(std::count(response->body.begin(), response->body.end(), '\n'), 8);
  // And the session id still arrives, because headers precede the body.
  EXPECT_TRUE(isUsableSessionId(response->get_header_value(SESSION_HEADER)));
}

TEST(RouterServerTest, TakesAnUnhealthyReplicaOutOfRotation)
{
  Fixture fixture;
  auto    client = fixture.client();

  fixture.a.healthy = false;
  // One blip should not depin every session, so it takes UNHEALTHY_THRESHOLD
  // consecutive failures.
  fixture.router->probeOnce();
  {
    const std::lock_guard<std::mutex> lock(fixture.router->registryMutex());
    EXPECT_TRUE(fixture.router->registry().state("replica-a")->ready) << "one failed probe must not evict";
  }
  fixture.router->probeOnce();

  for (int i = 0; i < 4; ++i)
  {
    const auto response = client.Post("/v1/completions", Json{{"prompt", "x"}}.dump(), "application/json");
    ASSERT_NE(response, nullptr);
    ASSERT_EQ(response->status, 200);
    EXPECT_EQ(Json::parse(response->body).at("replica").get<std::string>(), "replica-b");
  }

  // Recovery is immediate on the first success.
  fixture.a.healthy = true;
  fixture.router->probeOnce();
  {
    const std::lock_guard<std::mutex> lock(fixture.router->registryMutex());
    EXPECT_TRUE(fixture.router->registry().state("replica-a")->ready);
  }
}

TEST(RouterServerTest, RejectsWhenEveryReplicaIsDown)
{
  Fixture fixture;
  auto    client = fixture.client();

  fixture.a.healthy = false;
  fixture.b.healthy = false;
  fixture.router->probeOnce();
  fixture.router->probeOnce();

  const auto response = client.Post("/v1/completions", Json{{"prompt", "x"}}.dump(), "application/json");
  ASSERT_NE(response, nullptr);
  EXPECT_EQ(response->status, 503);
  // Even a rejection carries the session id: clients may retry with it.
  EXPECT_TRUE(isUsableSessionId(response->get_header_value(SESSION_HEADER)));

  const auto health = client.Get("/health");
  ASSERT_NE(health, nullptr);
  EXPECT_EQ(health->status, 503);
}

TEST(RouterServerTest, ReturnsBadGatewayWhenAReplicaIsUnreachable)
{
  Fixture fixture;
  auto    client = fixture.client();

  // Up as far as the health table is concerned, but gone.
  fixture.b.stop();
  fixture.a.stop();

  const auto response = client.Post("/v1/completions", Json{{"prompt", "x"}}.dump(), "application/json");
  ASSERT_NE(response, nullptr);
  EXPECT_EQ(response->status, 502);

  // A transport failure means a dead replica: it leaves rotation immediately
  // rather than after the next poll.
  const std::lock_guard<std::mutex> lock(fixture.router->registryMutex());
  const int                         ready = fixture.router->registry().readyCount();
  EXPECT_LT(ready, 2);
}
