#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <serving/engine/engine.hpp>

#include "support/stubs.hpp"

using serving::config::GenerateConfig;
using serving::config::RuntimeConfig;
using serving::engine::Engine;
using serving::engine::EngineConfig;
using serving::engine::ModelExecutor;
using serving::engine::StepCommand;
using serving::engine::StepResult;
using serving::engine::TokenOutput;
using serving::model::TokenizerAdapter;

namespace
{

using serving::testing::CharTokenizer;
using serving::testing::drain;
using serving::testing::ScriptedExecutor;

/// One character per token id, so generated text is readable in assertions.

/**
 * An executor that returns a scripted token per request and can be told to
 * fail; counts registrations, steps and closes.
 */

EngineConfig makeConfig()
{
  EngineConfig config;
  config.runtime.pageSize            = 4;
  config.scheduler.enablePrefixCache = false;
  config.scheduler.maxSeqLen         = 256;
  config.idlePollMicroseconds        = 50;
  return config;
}

/// Drain a stream to completion, returning every update.

} // namespace

TEST(EngineTest, RegistersTheModelBeforeStartingItsThread)
{
  const CharTokenizer tokenizer;
  ScriptedExecutor    executor({'a'});
  Engine              engine(makeConfig(), tokenizer, executor);

  EXPECT_FALSE(engine.isReady());
  EXPECT_EQ(executor.registered.load(), 0);

  engine.start();
  EXPECT_EQ(executor.registered.load(), 1);
  // The ordering matters once the executor is the Python bridge: registerModel
  // forks simpler's children and must run while the process is single-threaded.
  EXPECT_TRUE(engine.isReady());

  engine.stop();
  EXPECT_FALSE(engine.isReady());
  EXPECT_EQ(executor.closed.load(), 1);
}

TEST(EngineTest, StartingTwiceIsANoOp)
{
  const CharTokenizer tokenizer;
  ScriptedExecutor    executor({'a'});
  Engine              engine(makeConfig(), tokenizer, executor);
  engine.start();
  engine.start();
  EXPECT_EQ(executor.registered.load(), 1);
  engine.stop();
}

TEST(EngineTest, StreamsGeneratedTextEndToEnd)
{
  const CharTokenizer tokenizer;
  ScriptedExecutor    executor({'h', 'i', '!'});
  Engine              engine(makeConfig(), tokenizer, executor);
  engine.start();

  const auto prompt            = tokenizer.encode("Say:");
  executor.promptLengths["r1"] = static_cast<int>(prompt.size());

  GenerateConfig generate;
  generate.maxNewTokens = 3;
  generate.ignoreEos    = true;

  auto       stream  = engine.addRequest("r1", prompt, generate);
  const auto updates = drain(stream);

  ASSERT_FALSE(updates.empty());
  EXPECT_EQ(updates.back().text, "hi!");
  EXPECT_TRUE(updates.back().finished);
  EXPECT_EQ(updates.back().finishReason, "FINISHED_LENGTH");

  // Deltas must concatenate to the final text -- that is what SSE sends.
  std::string joined;
  for (const auto &update : updates) { joined += update.delta; }
  EXPECT_EQ(joined, "hi!");

  engine.stop();
}

TEST(EngineTest, StopsAtEos)
{
  const CharTokenizer tokenizer;
  // The tokenizer reports eos == 0, so the second token ends generation.
  ScriptedExecutor executor({'x', 0, 'y'});
  Engine           engine(makeConfig(), tokenizer, executor);
  engine.start();

  const auto prompt            = tokenizer.encode("go");
  executor.promptLengths["r1"] = static_cast<int>(prompt.size());

  GenerateConfig generate;
  generate.maxNewTokens = 8;

  auto       stream  = engine.addRequest("r1", prompt, generate);
  const auto updates = drain(stream);

  ASSERT_FALSE(updates.empty());
  EXPECT_TRUE(updates.back().finished);
  EXPECT_EQ(updates.back().finishReason, "FINISHED_EOS");
  engine.stop();
}

TEST(EngineTest, IgnoreEosKeepsGenerating)
{
  const CharTokenizer tokenizer;
  ScriptedExecutor    executor({0, 0, 0, 0});
  Engine              engine(makeConfig(), tokenizer, executor);
  engine.start();

  const auto prompt            = tokenizer.encode("go");
  executor.promptLengths["r1"] = static_cast<int>(prompt.size());

  GenerateConfig generate;
  generate.maxNewTokens = 4;
  generate.ignoreEos    = true;

  auto       stream  = engine.addRequest("r1", prompt, generate);
  const auto updates = drain(stream);
  EXPECT_EQ(updates.back().finishReason, "FINISHED_LENGTH");
  engine.stop();
}

TEST(EngineTest, ServesSeveralRequestsConcurrently)
{
  const CharTokenizer tokenizer;
  ScriptedExecutor    executor({'a', 'b', 'c'});
  Engine              engine(makeConfig(), tokenizer, executor);
  engine.start();

  GenerateConfig generate;
  generate.maxNewTokens = 3;
  generate.ignoreEos    = true;

  std::vector<std::shared_ptr<serving::engine::RequestStream>> streams;
  for (int i = 0; i < 4; ++i)
  {
    const std::string id       = "r" + std::to_string(i);
    const auto        prompt   = tokenizer.encode("go");
    executor.promptLengths[id] = static_cast<int>(prompt.size());
    streams.push_back(engine.addRequest(id, prompt, generate));
  }

  for (const auto &stream : streams)
  {
    const auto updates = drain(stream);
    ASSERT_FALSE(updates.empty());
    EXPECT_EQ(updates.back().text, "abc");
    EXPECT_TRUE(updates.back().finished);
  }
  engine.stop();
}

TEST(EngineTest, ChunksALongPromptWithoutChangingTheOutput)
{
  const CharTokenizer tokenizer;
  ScriptedExecutor    executor({'o', 'k'});
  auto                config                 = makeConfig();
  config.scheduler.longPrefillTokenThreshold = 4; // force several chunks
  Engine engine(config, tokenizer, executor);
  engine.start();

  const auto prompt            = tokenizer.encode("a rather longer prompt");
  executor.promptLengths["r1"] = static_cast<int>(prompt.size());

  GenerateConfig generate;
  generate.maxNewTokens = 2;
  generate.ignoreEos    = true;

  auto       stream  = engine.addRequest("r1", prompt, generate);
  const auto updates = drain(stream);
  EXPECT_EQ(updates.back().text, "ok");
  // Several steps were needed: the prompt did not fit one chunk.
  EXPECT_GT(executor.steps.load(), 2);
  engine.stop();
}

TEST(EngineTest, AnExecutorFailureFailsTheRequestNotTheEngine)
{
  const CharTokenizer tokenizer;
  ScriptedExecutor    executor({'a'});
  executor.failWith = "device exploded";
  Engine engine(makeConfig(), tokenizer, executor);
  engine.start();

  const auto prompt            = tokenizer.encode("go");
  executor.promptLengths["r1"] = static_cast<int>(prompt.size());

  GenerateConfig generate;
  generate.maxNewTokens = 2;

  auto       stream  = engine.addRequest("r1", prompt, generate);
  const auto updates = drain(stream);

  ASSERT_FALSE(updates.empty());
  EXPECT_TRUE(updates.back().finished);
  // The failure is an error, not a finish reason: reporting it as one turns a
  // crashed step into a successful completion at the HTTP layer.
  EXPECT_EQ(updates.back().error, "device exploded");
  EXPECT_EQ(updates.back().finishReason, "FINISHED_ERROR");
  // The engine survives it.
  EXPECT_TRUE(engine.isReady());

  // The failed request must be gone from the scheduler: a second request
  // completes and r1 is never asked for again.
  executor.failWith.clear();
  executor.clearDecodedIds();
  executor.promptLengths["r2"] = static_cast<int>(prompt.size());
  const auto second            = drain(engine.addRequest("r2", prompt, generate));
  ASSERT_FALSE(second.empty());
  EXPECT_TRUE(second.back().finished);
  EXPECT_NE(second.back().finishReason, "device exploded");
  int tokens = 0;
  for (const auto &u : second) { tokens += u.tokenId.has_value() ? 1 : 0; }
  EXPECT_EQ(tokens, 2);
  for (const auto &id : executor.decodedIds()) { EXPECT_NE(id, "r1") << "a failed request was rescheduled"; }
  engine.stop();
}

TEST(EngineTest, StopsOnAStopString)
{
  const CharTokenizer tokenizer;
  // Prompt "go" then generate "abc": the stop string cuts before "b".
  ScriptedExecutor executor({'a', 'b', 'c'});
  Engine           engine(makeConfig(), tokenizer, executor);
  engine.start();

  const auto prompt            = tokenizer.encode("go");
  executor.promptLengths["r1"] = static_cast<int>(prompt.size());

  GenerateConfig generate;
  generate.maxNewTokens = 8;
  generate.stop         = {"b"};

  const auto updates = drain(engine.addRequest("r1", prompt, generate));
  ASSERT_FALSE(updates.empty());
  EXPECT_TRUE(updates.back().finished);
  EXPECT_EQ(updates.back().finishReason, "FINISHED_STOP");
  // The stop text is not part of the answer, and nothing after it is generated.
  EXPECT_EQ(updates.back().text, "a");
  engine.stop();
}

TEST(EngineTest, OnlyTheFinalUpdateCarriesTheWholeText)
{
  const CharTokenizer tokenizer;
  ScriptedExecutor    executor({'h', 'i', '!'});
  Engine              engine(makeConfig(), tokenizer, executor);
  engine.start();

  const auto prompt            = tokenizer.encode("go");
  executor.promptLengths["r1"] = static_cast<int>(prompt.size());

  GenerateConfig generate;
  generate.maxNewTokens = 3;

  const auto updates = drain(engine.addRequest("r1", prompt, generate));
  ASSERT_GE(updates.size(), 2u);
  // Cumulative text per token is quadratic in the generated length; the deltas
  // carry the stream and the final update carries the whole answer.
  std::string accumulated;
  for (size_t i = 0; i + 1 < updates.size(); ++i)
  {
    EXPECT_TRUE(updates[i].text.empty()) << "intermediate update " << i << " carried the whole text";
    accumulated += updates[i].delta;
  }
  accumulated += updates.back().delta;
  EXPECT_EQ(updates.back().text, "hi!");
  EXPECT_EQ(accumulated, "hi!");
  engine.stop();
}

TEST(EngineTest, AFailedModelLoadLeavesTheEngineUnstarted)
{
  class FailingExecutor : public ModelExecutor
  {
    public:

    int        registerModel() override { throw std::runtime_error("kernel compile failed"); }
    StepResult executeStep(const StepCommand &) override { return {}; }
  };

  const CharTokenizer tokenizer;
  FailingExecutor     executor;
  Engine              engine(makeConfig(), tokenizer, executor);

  EXPECT_THROW(engine.start(), std::runtime_error);
  EXPECT_FALSE(engine.isReady());
  // Admitting here would reach a scheduler that was never built.
  EXPECT_THROW((void)engine.addRequest("r1", {1, 2}, GenerateConfig{}), std::runtime_error);
}

TEST(EngineTest, AbortClosesTheStream)
{
  const CharTokenizer tokenizer;
  ScriptedExecutor    executor({'a'});
  Engine              engine(makeConfig(), tokenizer, executor);
  engine.start();

  const auto prompt            = tokenizer.encode("go");
  executor.promptLengths["r1"] = 9999; // never completes, so it keeps running

  GenerateConfig generate;
  generate.maxNewTokens = 100;

  auto stream = engine.addRequest("r1", prompt, generate);
  engine.abortRequest("r1");
  // A closed stream drains and then returns nothing, rather than blocking.
  while (stream->pop()) {}
  EXPECT_TRUE(stream->closed());
  engine.stop();
}

TEST(EngineTest, RejectsAPromptLongerThanMaxSeqLen)
{
  const CharTokenizer tokenizer;
  ScriptedExecutor    executor({'a'});
  auto                config = makeConfig();
  config.scheduler.maxSeqLen = 8;
  Engine engine(config, tokenizer, executor);
  engine.start();

  GenerateConfig generate;
  EXPECT_THROW((void)engine.addRequest("r1", tokenizer.encode("far too long a prompt"), generate), std::invalid_argument);
  engine.stop();
}

TEST(EngineTest, CarriesEachRequestsSamplingParamsToTheExecutor)
{
  CharTokenizer    tokenizer;
  ScriptedExecutor executor(ScriptedExecutor::fromText("ab"));
  executor.promptLength = 2;

  EngineConfig config;
  config.runtime.pageSize            = 4;
  config.scheduler.enablePrefixCache = false;
  config.scheduler.maxSeqLen         = 64;
  config.idlePollMicroseconds        = 50;

  Engine engine(config, tokenizer, executor);
  engine.start();

  GenerateConfig greedy;
  greedy.maxNewTokens = 2;
  greedy.ignoreEos    = true;

  GenerateConfig sampled = greedy;
  sampled.temperature    = 0.7;
  sampled.topP           = 0.95;
  sampled.topK           = 40;
  sampled.seed           = 1234U;

  // Two requests in flight at once, so the step that carries them is a mixed
  // batch: the executor must see each request's own parameters, not one
  // batch-wide setting.
  auto greedyStream  = engine.addRequest("greedy", tokenizer.encode("hi"), greedy);
  auto sampledStream = engine.addRequest("sampled", tokenizer.encode("yo"), sampled);
  drain(greedyStream);
  drain(sampledStream);

  const auto greedySeen = executor.samplingFor("greedy");
  ASSERT_TRUE(greedySeen.has_value());
  EXPECT_TRUE(greedySeen->isGreedy());
  EXPECT_EQ(greedySeen->temperature, 0.0);
  EXPECT_EQ(greedySeen->topP, 1.0);
  EXPECT_FALSE(greedySeen->topK.has_value());
  EXPECT_FALSE(greedySeen->seed.has_value());

  const auto sampledSeen = executor.samplingFor("sampled");
  ASSERT_TRUE(sampledSeen.has_value());
  EXPECT_FALSE(sampledSeen->isGreedy());
  EXPECT_DOUBLE_EQ(sampledSeen->temperature, 0.7);
  EXPECT_DOUBLE_EQ(sampledSeen->topP, 0.95);
  ASSERT_TRUE(sampledSeen->topK.has_value());
  EXPECT_EQ(*sampledSeen->topK, 40);
  ASSERT_TRUE(sampledSeen->seed.has_value());
  EXPECT_EQ(*sampledSeen->seed, 1234U);

  engine.stop();
}

TEST(EngineTest, GeneratesDistinctRequestIds)
{
  const CharTokenizer tokenizer;
  ScriptedExecutor    executor({'a'});
  Engine              engine(makeConfig(), tokenizer, executor);
  EXPECT_EQ(engine.generateRequestId(), "serving-req-1");
  EXPECT_EQ(engine.generateRequestId(), "serving-req-2");
}
