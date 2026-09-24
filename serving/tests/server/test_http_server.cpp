#include <map>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <serving/server/http_server.hpp>

using Json = nlohmann::json;
using serving::config::GenerateConfig;
using serving::engine::Engine;
using serving::engine::EngineConfig;
using serving::engine::ModelExecutor;
using serving::engine::StepCommand;
using serving::engine::StepResult;
using serving::model::TokenizerAdapter;
using serving::server::HttpServer;
using serving::server::mapFinishReason;
using serving::server::ServerConfig;

namespace
{

/// One character per token, so a prompt's text is its token ids.
class CharTokenizer : public TokenizerAdapter
{
  public:

  [[nodiscard]] std::vector<int> encode(const std::string &text) const override
  {
    std::vector<int> ids;
    for (const char c : text) { ids.push_back(static_cast<int>(c)); }
    return ids;
  }

  [[nodiscard]] std::string decode(const std::vector<int> &ids, bool = true) const override
  {
    std::string out;
    for (const int id : ids) { out += static_cast<char>(id); }
    return out;
  }

  [[nodiscard]] std::optional<int> eosTokenId() const override { return 0; }
};

/// Emits a fixed script, one token per step per request.
class ScriptedExecutor : public ModelExecutor
{
  public:

  explicit ScriptedExecutor(std::string script)
    : _script(std::move(script))
  {}

  int registerModel() override { return 64; }

  StepResult executeStep(const StepCommand &command) override
  {
    StepResult result;
    if (!failWith.empty())
    {
      result.error = failWith;
      return result;
    }
    for (const auto &item : command.prefill)
    {
      const int covered = item.numComputedTokens + static_cast<int>(item.chunkTokens.size());
      if (covered >= promptLength) { result.newTokens[item.requestId] = {next(item.requestId)}; }
    }
    for (const auto &item : command.decode) { result.newTokens[item.requestId] = {next(item.requestId)}; }
    return result;
  }

  int         promptLength = 0;
  std::string failWith;

  private:

  int next(const std::string &requestId)
  {
    const size_t index = _emitted[requestId]++;
    return static_cast<int>(_script[index % _script.size()]);
  }

  std::string                   _script;
  std::map<std::string, size_t> _emitted;
};

/// Engine + server on an ephemeral port, torn down in the right order.
struct Fixture
{
  CharTokenizer               tokenizer;
  ScriptedExecutor            executor;
  std::unique_ptr<Engine>     engine;
  std::unique_ptr<HttpServer> server;

  explicit Fixture(std::string script = "hi!", int promptLength = 4)
    : executor(std::move(script))
  {
    executor.promptLength = promptLength;

    EngineConfig engineConfig;
    engineConfig.runtime.pageSize            = 4;
    engineConfig.scheduler.enablePrefixCache = false;
    engineConfig.scheduler.maxSeqLen         = 256;
    engineConfig.idlePollMicroseconds        = 50;
    engine                                   = std::make_unique<Engine>(engineConfig, tokenizer, executor);
    engine->start();

    ServerConfig serverConfig;
    serverConfig.host    = "127.0.0.1";
    serverConfig.port    = 0; // ephemeral
    serverConfig.modelId = "test-model";
    server               = std::make_unique<HttpServer>(serverConfig, *engine, tokenizer);
    server->start();
  }

  ~Fixture()
  {
    server->stop();
    engine->stop();
  }

  [[nodiscard]] httplib::Client client() const
  {
    httplib::Client c("127.0.0.1", server->boundPort());
    c.set_read_timeout(10, 0);
    return c;
  }
};

} // namespace

TEST(FinishReasonTest, MapsSchedulerStatusesToTheOpenAiVocabulary)
{
  EXPECT_EQ(mapFinishReason("FINISHED_EOS"), "eos");
  EXPECT_EQ(mapFinishReason("FINISHED_LENGTH"), "length");
  EXPECT_EQ(mapFinishReason("FINISHED_STOP"), "stop");
  EXPECT_EQ(mapFinishReason("FINISHED_ABORTED"), "aborted");
  EXPECT_EQ(mapFinishReason("something else"), "stop");
}

TEST(HttpServerTest, HealthReportsReadiness)
{
  Fixture fixture;
  auto    client = fixture.client();

  const auto response = client.Get("/health");
  ASSERT_NE(response, nullptr);
  EXPECT_EQ(response->status, 200);
  EXPECT_EQ(Json::parse(response->body).at("status"), "ok");
}

TEST(HttpServerTest, HealthReports503WhenTheEngineIsDown)
{
  Fixture fixture;
  // /health is 503 until the engine is ready.
  fixture.engine->stop();

  auto       client   = fixture.client();
  const auto response = client.Get("/health");
  ASSERT_NE(response, nullptr);
  EXPECT_EQ(response->status, 503);
}

TEST(HttpServerTest, ListsTheServedModel)
{
  Fixture    fixture;
  auto       client   = fixture.client();
  const auto response = client.Get("/v1/models");
  ASSERT_NE(response, nullptr);
  EXPECT_EQ(response->status, 200);

  const auto body = Json::parse(response->body);
  EXPECT_EQ(body.at("object"), "list");
  ASSERT_EQ(body.at("data").size(), 1u);
  EXPECT_EQ(body.at("data")[0].at("id"), "test-model");
}

TEST(HttpServerTest, CompletesNonStreaming)
{
  Fixture fixture;
  auto    client = fixture.client();

  const Json request{{"prompt", "Say:"}, {"max_tokens", 3}, {"ignore_eos", true}};
  const auto response = client.Post("/v1/completions", request.dump(), "application/json");
  ASSERT_NE(response, nullptr);
  ASSERT_EQ(response->status, 200) << response->body;

  const auto body = Json::parse(response->body);
  EXPECT_EQ(body.at("object"), "text_completion");
  EXPECT_EQ(body.at("model"), "test-model");
  ASSERT_EQ(body.at("choices").size(), 1u);
  EXPECT_EQ(body.at("choices")[0].at("text"), "hi!");
  EXPECT_EQ(body.at("choices")[0].at("finish_reason"), "length");
  EXPECT_EQ(body.at("usage").at("prompt_tokens"), 4);
  EXPECT_EQ(body.at("usage").at("completion_tokens"), 3);
  EXPECT_EQ(body.at("usage").at("total_tokens"), 7);
}

TEST(HttpServerTest, StreamsCompletionChunks)
{
  Fixture fixture;
  auto    client = fixture.client();

  const Json request{{"prompt", "Say:"}, {"max_tokens", 3}, {"ignore_eos", true}, {"stream", true}};
  const auto response = client.Post("/v1/completions", request.dump(), "application/json");
  ASSERT_NE(response, nullptr);
  ASSERT_EQ(response->status, 200);
  // cpp-httplib reassembles the chunked body, which is all this needs: the
  // framing is in the bytes either way.
  const std::string &received = response->body;

  // SSE framing: one `data:` line per chunk, terminated by [DONE].
  EXPECT_NE(received.find("data: "), std::string::npos);
  EXPECT_NE(received.find("data: [DONE]"), std::string::npos);

  // The deltas must reassemble into the full text.
  std::string assembled;
  bool        sawUsage = false;
  size_t      pos      = 0;
  while ((pos = received.find("data: ", pos)) != std::string::npos)
  {
    const size_t start = pos + 6;
    const size_t end   = received.find("\n\n", start);
    if (end == std::string::npos) { break; }
    const std::string payload = received.substr(start, end - start);
    pos                       = end;
    if (payload == "[DONE]") { continue; }

    const auto chunk = Json::parse(payload);
    if (chunk.contains("usage"))
    {
      sawUsage = true;
      EXPECT_TRUE(chunk.at("choices").empty()) << "the usage chunk must carry no choices";
      EXPECT_EQ(chunk.at("usage").at("completion_tokens"), 3);
      continue;
    }
    assembled += chunk.at("choices")[0].at("text").get<std::string>();
  }
  EXPECT_EQ(assembled, "hi!");
  EXPECT_TRUE(sawUsage) << "no terminal usage chunk";
}

TEST(HttpServerTest, RejectsMalformedJson)
{
  Fixture    fixture;
  auto       client   = fixture.client();
  const auto response = client.Post("/v1/completions", "{not json", "application/json");
  ASSERT_NE(response, nullptr);
  EXPECT_EQ(response->status, 400);
}

TEST(HttpServerTest, RejectsAMissingPrompt)
{
  Fixture    fixture;
  auto       client   = fixture.client();
  const auto response = client.Post("/v1/completions", json{{"max_tokens", 2}}.dump(), "application/json");
  ASSERT_NE(response, nullptr);
  EXPECT_EQ(response->status, 400);
}

TEST(HttpServerTest, AFailedStepIsAnErrorNotACompletion)
{
  Fixture fixture;
  fixture.executor.failWith = "device exploded";
  auto client               = fixture.client();

  const auto response = client.Post("/v1/completions", json{{"prompt", "hi"}, {"max_tokens", 4}}.dump(), "application/json");
  ASSERT_NE(response, nullptr);
  // Previously 200 with finish_reason "stop": a crashed step was indistinguishable
  // from a short but successful answer.
  EXPECT_EQ(response->status, 500);
  EXPECT_NE(response->body.find("device exploded"), std::string::npos) << response->body;
}

TEST(HttpServerTest, RejectsAPromptTheSchedulerCannotAdmit)
{
  Fixture fixture;
  auto    client = fixture.client();

  std::string huge(500, 'x'); // beyond maxSeqLen 256
  const auto  response = client.Post("/v1/completions", json{{"prompt", huge}}.dump(), "application/json");
  ASSERT_NE(response, nullptr);
  EXPECT_EQ(response->status, 400);
  EXPECT_NE(response->body.find("leaves no room for generation within max_seq_len"), std::string::npos);
}

TEST(HttpServerTest, ChatCompletionsNeedATemplate)
{
  Fixture fixture;
  auto    client = fixture.client();

  const Json request{{"messages", Json::array({json{{"role", "user"}, {"content", "hi"}}})}};
  const auto response = client.Post("/v1/chat/completions", request.dump(), "application/json");
  ASSERT_NE(response, nullptr);
  // No template was supplied to this server, so the request is refused rather
  // than served with an un-templated prompt.
  EXPECT_EQ(response->status, 400);
}
