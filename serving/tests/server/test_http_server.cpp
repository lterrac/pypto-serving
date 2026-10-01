#include <map>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <serving/server/http_server.hpp>

#include "support/stubs.hpp"

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

using serving::testing::CharTokenizer;
using serving::testing::drain;
using serving::testing::ScriptedExecutor;

/// One character per token, so a prompt's text is its token ids.

/// Emits a fixed script, one token per step per request.

/// Engine + server on an ephemeral port, torn down in the right order.
struct Fixture
{
  CharTokenizer                                 tokenizer;
  ScriptedExecutor                              executor;
  std::unique_ptr<serving::model::ChatTemplate> chatTemplate;
  std::unique_ptr<Engine>                       engine;
  std::unique_ptr<HttpServer>                   server;

  explicit Fixture(std::string script = "hi!", int promptLength = 4, const std::string &chatTemplateSource = "")
    : executor(ScriptedExecutor::fromText(script))
  {
    executor.promptLength = promptLength;
    if (!chatTemplateSource.empty()) { chatTemplate = std::make_unique<serving::model::ChatTemplate>(chatTemplateSource, "", "<|im_end|>"); }

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
    server               = std::make_unique<HttpServer>(serverConfig, *engine, tokenizer, chatTemplate.get());
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

TEST(HttpServerTest, RefusesSamplingItDoesNotImplement)
{
  Fixture fixture;
  auto    client = fixture.client();

  // Accepting these and sampling greedily anyway returns a different
  // distribution than the caller asked for, silently.
  for (const auto &body : {json{{"prompt", "hi"}, {"temperature", 0.7}}, json{{"prompt", "hi"}, {"top_k", 40}}})
  {
    const auto response = client.Post("/v1/completions", body.dump(), "application/json");
    ASSERT_NE(response, nullptr);
    EXPECT_EQ(response->status, 400) << body.dump();
    EXPECT_NE(response->body.find("greedy"), std::string::npos) << response->body;
  }

  // Greedy defaults still work.
  const auto ok = client.Post("/v1/completions", json{{"prompt", "Say:"}, {"temperature", 0.0}, {"max_tokens", 2}, {"ignore_eos", true}}.dump(), "application/json");
  ASSERT_NE(ok, nullptr);
  EXPECT_EQ(ok->status, 200);
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

/// Renders the role and the content, then shows whatever template
/// keywords reached it. With a one-character-per-token tokenizer the rendered
/// length is the reported `prompt_tokens`, so a test can see what was passed.
constexpr const char *KWARG_TEMPLATE = "{%- for m in messages %}{{ m.role }}:{{ m.content }}{%- endfor %}"
                                       "{%- if enable_thinking is defined and not enable_thinking %}|nothink{%- endif %}"
                                       "{%- if reasoning_effort is defined %}|e={{ reasoning_effort }}{%- endif %}";

/// The prompt the template rendered, read back off the usage block.
static int renderedLength(httplib::Client &client, const Json &body)
{
  const auto response = client.Post("/v1/chat/completions", body.dump(), "application/json");
  EXPECT_NE(response, nullptr);
  if (response == nullptr) { return -1; }
  EXPECT_EQ(response->status, 200) << response->body;
  if (response->status != 200) { return -1; }
  return Json::parse(response->body).at("usage").at("prompt_tokens").get<int>();
}

TEST(HttpServerTest, ForwardsChatTemplateKwargs)
{
  Fixture fixture("hi!", 1, KWARG_TEMPLATE);
  auto    client = fixture.client();

  const Json messages = Json::array({json{{"role", "user"}, {"content", "hi"}}});
  const Json base{{"messages", messages}, {"max_tokens", 1}};

  // "user:hi" -- a request carrying neither field renders as if the template took
  // no keywords at all, which is what the Python server passes.
  EXPECT_EQ(renderedLength(client, base), 7);

  // "user:hi|nothink"
  Json withKwargs                    = base;
  withKwargs["chat_template_kwargs"] = json{{"enable_thinking", false}};
  EXPECT_EQ(renderedLength(client, withKwargs), 15);

  // "user:hi|e=high" -- reasoning_effort supplies enable_thinking=true.
  Json effort                = base;
  effort["reasoning_effort"] = "high";
  EXPECT_EQ(renderedLength(client, effort), 14);

  // "user:hi|nothink|e=none" -- and enable_thinking=false for "none".
  Json none                = base;
  none["reasoning_effort"] = "none";
  EXPECT_EQ(renderedLength(client, none), 22);

  // "user:hi|e=none" -- an explicit enable_thinking wins over the one
  // reasoning_effort would have supplied, matching the Python's setdefault.
  Json both                    = base;
  both["chat_template_kwargs"] = json{{"enable_thinking", true}};
  both["reasoning_effort"]     = "none";
  EXPECT_EQ(renderedLength(client, both), 14);
}

TEST(HttpServerTest, RejectsMalformedChatTemplateKwargs)
{
  Fixture fixture("hi!", 1, KWARG_TEMPLATE);
  auto    client = fixture.client();

  const Json messages = Json::array({json{{"role", "user"}, {"content", "hi"}}});
  for (const auto &body : {json{{"messages", messages}, {"chat_template_kwargs", "enable_thinking"}}, json{{"messages", messages}, {"reasoning_effort", 3}}})
  {
    const auto response = client.Post("/v1/chat/completions", body.dump(), "application/json");
    ASSERT_NE(response, nullptr);
    EXPECT_EQ(response->status, 400) << body.dump();
  }

  // An explicit null is the same as omitting the field.
  const Json nulls{{"messages", messages}, {"max_tokens", 1}, {"chat_template_kwargs", nullptr}, {"reasoning_effort", nullptr}};
  EXPECT_EQ(renderedLength(client, nulls), 7);
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
