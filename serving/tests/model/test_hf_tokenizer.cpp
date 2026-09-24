#include <cstdlib>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <serving/engine/detokenizer.hpp>
#include <serving/model/hf_tokenizer.hpp>

using serving::engine::IncrementalDetokenizer;
using serving::model::HfTokenizer;

namespace
{

/**
 * Need a model directory: set SERVING_TEST_MODEL_DIR, otherwise skipped.
 */
std::string modelDir()
{
  const char *dir = std::getenv("SERVING_TEST_MODEL_DIR");
  return dir == nullptr ? std::string{} : std::string{dir};
}

/// The prompt and continuation the Python stack produced, from
/// serving/tests/fixtures/qwen3_generation.json.
const std::vector<int> kGoldenOutput{12095, 13, 3555, 374, 279, 6722, 315, 279};
constexpr const char  *kGoldenText = " Paris. What is the capital of the";

} // namespace

TEST(HfTokenizerTest, RoundTripsThePromptFromTheGoldenRun)
{
  const std::string dir = modelDir();
  if (dir.empty()) { GTEST_SKIP() << "set SERVING_TEST_MODEL_DIR to run"; }

  const auto tokenizer = HfTokenizer::fromModelDir(dir);
  ASSERT_NE(tokenizer, nullptr);

  const auto ids = tokenizer->encode("The capital of France is");
  // A 5-token prompt for this string.
  EXPECT_EQ(ids.size(), 5u);
  EXPECT_EQ(tokenizer->decode(ids), "The capital of France is");
}

TEST(HfTokenizerTest, DecodesTheGoldenContinuation)
{
  const std::string dir = modelDir();
  if (dir.empty()) { GTEST_SKIP() << "set SERVING_TEST_MODEL_DIR to run"; }

  const auto tokenizer = HfTokenizer::fromModelDir(dir);
  EXPECT_EQ(tokenizer->decode(kGoldenOutput), kGoldenText);
}

TEST(HfTokenizerTest, SkipsSpecialTokensByDefault)
{
  const std::string dir = modelDir();
  if (dir.empty()) { GTEST_SKIP() << "set SERVING_TEST_MODEL_DIR to run"; }

  const auto tokenizer = HfTokenizer::fromModelDir(dir);
  ASSERT_TRUE(tokenizer->eosTokenId().has_value()) << "eos id not resolved from tokenizer_config.json";
  const int eos = *tokenizer->eosTokenId();
  EXPECT_TRUE(tokenizer->isSpecial(eos));

  std::vector<int> withEos = kGoldenOutput;
  withEos.push_back(eos);

  // The whole point of the id filter: tokenizers-cpp's public Decode hard-codes
  // skip_special_tokens=false, so without it the eos marker would be streamed.
  EXPECT_EQ(tokenizer->decode(withEos, true), kGoldenText);
  EXPECT_NE(tokenizer->decode(withEos, false), kGoldenText);
}

TEST(HfTokenizerTest, IncrementalDetokenizationMatchesAFullDecodeOnRealText)
{
  const std::string dir = modelDir();
  if (dir.empty()) { GTEST_SKIP() << "set SERVING_TEST_MODEL_DIR to run"; }

  const auto             tokenizer = HfTokenizer::fromModelDir(dir);
  IncrementalDetokenizer detokenizer(*tokenizer);

  std::vector<int> ids;
  std::string      streamed;
  for (const int id : kGoldenOutput)
  {
    ids.push_back(id);
    streamed = detokenizer.append(ids);
  }
  streamed = detokenizer.finalize(ids);

  EXPECT_EQ(streamed, tokenizer->decode(kGoldenOutput));
  EXPECT_EQ(streamed, kGoldenText);
}
