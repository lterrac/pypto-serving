#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <serving/engine/detokenizer.hpp>

using serving::engine::endsWithReplacementChar;
using serving::engine::IncrementalDetokenizer;
using serving::model::TokenizerAdapter;

namespace
{

/// U+FFFD, as a byte-level tokenizer renders an incomplete character.
const std::string kReplacement = "\xEF\xBF\xBD";

/**
 * Tokens 6+7 together render a star; 6 on its own is an incomplete character.
 * Ported from _MultiByteTokenizer in tests/unit/serving/engine/test_output_delivery.py.
 */
class MultiByteTokenizer : public TokenizerAdapter
{
  public:

  mutable int decodeCalls   = 0;
  mutable int decodedTokens = 0;

  [[nodiscard]] std::vector<int> encode(const std::string &) const override { return {}; }

  [[nodiscard]] std::string decode(const std::vector<int> &ids, bool = true) const override
  {
    decodeCalls += 1;
    decodedTokens += static_cast<int>(ids.size());

    static const std::map<int, std::string> table{{1, "He"}, {2, "llo"}, {3, " wor"}, {4, "ld"}, {5, "!"}};
    std::string                             out;
    size_t                                  i = 0;
    while (i < ids.size())
    {
      const int token = ids[i];
      if (token == 6)
      {
        if (i + 1 < ids.size() && ids[i + 1] == 7)
        {
          out += "\xE2\x98\x85"; // U+2605 BLACK STAR
          i += 2;
          continue;
        }
        out += kReplacement;
        i += 1;
        continue;
      }
      if (token == 7)
      {
        out += "\xE2\x98\x85";
        i += 1;
        continue;
      }
      out += table.at(token);
      i += 1;
    }
    return out;
  }
};

/// Token 6 alone is an incomplete character, and it is the last token.
class TrailingByteTokenizer : public TokenizerAdapter
{
  public:

  [[nodiscard]] std::vector<int> encode(const std::string &) const override { return {}; }

  [[nodiscard]] std::string decode(const std::vector<int> &ids, bool = true) const override
  {
    static const std::map<int, std::string> table{{1, "Hi"}, {2, "!"}};
    std::string                             out;
    for (const int token : ids) { out += (token == 6) ? kReplacement : table.at(token); }
    return out;
  }
};

/// One character per token; used for the complexity check.
class CharTokenizer : public TokenizerAdapter
{
  public:

  mutable int decodedTokens = 0;

  [[nodiscard]] std::vector<int> encode(const std::string &) const override { return {}; }

  [[nodiscard]] std::string decode(const std::vector<int> &ids, bool = true) const override
  {
    decodedTokens += static_cast<int>(ids.size());
    std::string out;
    for (const int token : ids) { out += static_cast<char>('a' + (token % 26)); }
    return out;
  }
};

bool contains(const std::string &haystack, const std::string &needle) { return haystack.find(needle) != std::string::npos; }

} // namespace

TEST(DetokenizerTest, DetectsATrailingReplacementCharacter)
{
  EXPECT_TRUE(endsWithReplacementChar(kReplacement));
  EXPECT_TRUE(endsWithReplacementChar("Hi!" + kReplacement));
  EXPECT_FALSE(endsWithReplacementChar("Hi!"));
  EXPECT_FALSE(endsWithReplacementChar(""));
  EXPECT_FALSE(endsWithReplacementChar("\xE2\x98\x85")); // a complete star
}

TEST(DetokenizerTest, MatchesAFullDecodeAndHidesPartialCharacters)
{
  // Ported from test_incremental_detok_matches_full_decode_and_hides_partial_chars.
  const MultiByteTokenizer tokenizer;
  IncrementalDetokenizer   detokenizer(tokenizer);

  const std::vector<int>   sequence{1, 2, 3, 4, 5, 6, 7};
  std::vector<std::string> perStep;
  std::string              cumulative;

  for (size_t k = 1; k <= sequence.size(); ++k)
  {
    const std::vector<int> prefix(sequence.begin(), sequence.begin() + static_cast<long>(k));
    cumulative = detokenizer.append(prefix);
    perStep.push_back(cumulative);
  }

  // A partial character must never reach the stream.
  for (const std::string &text : perStep) { EXPECT_FALSE(contains(text, kReplacement)) << "leaked a partial char: " << text; }

  ASSERT_EQ(perStep.size(), 7u);
  EXPECT_EQ(perStep[4], "Hello world!"); // token 6 arrives next
  EXPECT_EQ(perStep[5], "Hello world!"); // still withheld: the char is incomplete
  EXPECT_EQ(cumulative, "Hello world!\xE2\x98\x85");
  EXPECT_EQ(cumulative, tokenizer.decode(sequence)) << "incremental text diverged from a full decode";
}

TEST(DetokenizerTest, FinalizeFlushesATrailingIncompleteCharacter)
{
  // Ported from test_finalize_detok_flushes_trailing_incomplete_char_at_eos.
  // Guards the truncation bug: the incremental path withholds a trailing U+FFFD
  // forever, because no later token will arrive once generation stops.
  const TrailingByteTokenizer tokenizer;
  IncrementalDetokenizer      detokenizer(tokenizer);

  const std::vector<int> sequence{1, 2, 6};
  std::string            incremental;
  for (size_t k = 1; k <= sequence.size(); ++k)
  {
    const std::vector<int> prefix(sequence.begin(), sequence.begin() + static_cast<long>(k));
    incremental = detokenizer.append(prefix);
  }

  EXPECT_EQ(incremental, "Hi!");
  EXPECT_FALSE(contains(incremental, kReplacement));

  const std::string final = detokenizer.finalize(sequence);
  EXPECT_EQ(final, "Hi!" + kReplacement);
  EXPECT_EQ(final, tokenizer.decode(sequence));
}

TEST(DetokenizerTest, AnEmptyOutputStaysEmpty)
{
  const CharTokenizer    tokenizer;
  IncrementalDetokenizer detokenizer(tokenizer);
  EXPECT_EQ(detokenizer.append({}), "");
  EXPECT_EQ(detokenizer.finalize({}), "");
  EXPECT_EQ(detokenizer.readOffset(), 0);
}

TEST(DetokenizerTest, WorkIsLinearInGeneratedTokens)
{
  // The reason this class exists: decoding the whole output every step is
  // O(N^2). The decode window must stay bounded, so total tokens decoded grows
  // linearly rather than quadratically.
  const CharTokenizer    tokenizer;
  IncrementalDetokenizer detokenizer(tokenizer);

  constexpr int    kSteps = 400;
  std::vector<int> ids;
  for (int i = 0; i < kSteps; ++i)
  {
    ids.push_back(i);
    detokenizer.append(ids);
  }

  // A full re-decode every step would be kSteps*(kSteps+1)/2 = 80,200 tokens.
  // The bounded window decodes a small constant per step instead.
  const int quadratic = kSteps * (kSteps + 1) / 2;
  EXPECT_LT(tokenizer.decodedTokens, quadratic / 10) << "decoded " << tokenizer.decodedTokens << " tokens over " << kSteps << " steps; the window is not bounded";
  EXPECT_LT(tokenizer.decodedTokens, kSteps * 10) << "per-step work should be a small constant";
}

TEST(DetokenizerTest, KeepsAThreeTokenContextWindow)
{
  // Not one or two: a short window loses boundary context and corrupts spacing
  // and multi-token characters for SentencePiece / byte-level BPE tokenizers.
  const CharTokenizer    tokenizer;
  IncrementalDetokenizer detokenizer(tokenizer);

  std::vector<int> ids;
  for (int i = 0; i < 10; ++i)
  {
    ids.push_back(i);
    detokenizer.append(ids);
  }
  EXPECT_EQ(detokenizer.readOffset(), 10);
  EXPECT_EQ(detokenizer.prefixOffset(), 7);
}
