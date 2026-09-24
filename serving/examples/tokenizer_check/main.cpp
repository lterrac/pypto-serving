/**
 * Checks the tokenizer against a model directory, without gtest.
 *
 *   tokenizer-check <model_dir>
 */

#include <cstdio>
#include <string>
#include <vector>

#include <serving/engine/detokenizer.hpp>
#include <serving/model/hf_tokenizer.hpp>

namespace
{

int failures = 0;

void check(bool condition, const std::string &what)
{
  printf("%-52s %s\n", what.c_str(), condition ? "ok" : "FAILED");
  if (!condition) { failures += 1; }
}

} // namespace

int main(int argc, char **argv)
{
  if (argc < 2)
  {
    fprintf(stderr, "usage: %s <model_dir>\n", argv[0]);
    return 2;
  }

  // From serving/tests/fixtures/qwen3_generation.json.
  const std::vector<int> golden{12095, 13, 3555, 374, 279, 6722, 315, 279};
  const std::string      goldenText = " Paris. What is the capital of the";

  const auto tokenizer = serving::model::HfTokenizer::fromModelDir(argv[1]);

  const auto ids = tokenizer->encode("The capital of France is");
  printf("prompt ids:");
  for (int id : ids) { printf(" %d", id); }
  printf("\n");
  check(ids.size() == 5, "prompt encodes to 5 tokens");
  check(tokenizer->decode(ids) == "The capital of France is", "prompt round-trips");
  check(tokenizer->decode(golden) == goldenText, "golden continuation decodes correctly");

  const bool haveEos = tokenizer->eosTokenId().has_value();
  check(haveEos, "eos id resolved from tokenizer_config.json");
  if (haveEos)
  {
    const int eos = *tokenizer->eosTokenId();
    printf("eos id: %d\n", eos);
    check(tokenizer->isSpecial(eos), "eos is marked special");
    std::vector<int> withEos = golden;
    withEos.push_back(eos);
    check(tokenizer->decode(withEos, true) == goldenText, "special tokens skipped by default");
    check(tokenizer->decode(withEos, false) != goldenText, "not skipping them really does differ");
    printf("without skipping: [%s]\n", tokenizer->decode(withEos, false).c_str());
  }

  serving::engine::IncrementalDetokenizer detokenizer(*tokenizer);
  std::vector<int>                        streamed;
  std::string                             text;
  for (const int id : golden)
  {
    streamed.push_back(id);
    text = detokenizer.append(streamed);
  }
  text = detokenizer.finalize(streamed);
  check(text == goldenText, "incremental detok matches a full decode");

  printf("\n%s\n", failures == 0 ? "ALL OK" : "FAILURES");
  return failures == 0 ? 0 : 1;
}
