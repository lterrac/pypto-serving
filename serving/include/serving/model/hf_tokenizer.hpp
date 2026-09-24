#pragma once

/**
 * TokenizerAdapter over tokenizers-cpp, from a model directory's
 * tokenizer.json; special-token ids from tokenizer_config.json. Compiled only
 * with the tokenizers feature.
 */

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include <serving/model/tokenizer.hpp>

namespace tokenizers
{
class Tokenizer;
}

namespace serving::model
{

class HfTokenizer : public TokenizerAdapter
{
  public:

  /// Load from a model directory containing tokenizer.json (+ tokenizer_config.json).
  static std::unique_ptr<HfTokenizer> fromModelDir(const std::string &modelDir);

  HfTokenizer(std::unique_ptr<tokenizers::Tokenizer> tokenizer, std::optional<int> bos, std::optional<int> eos, std::unordered_set<int> specialIds);
  ~HfTokenizer() override;

  [[nodiscard]] std::vector<int> encode(const std::string &text) const override;
  [[nodiscard]] std::string      decode(const std::vector<int> &tokenIds, bool skipSpecialTokens = true) const override;

  [[nodiscard]] std::optional<int> bosTokenId() const override { return _bos; }
  [[nodiscard]] std::optional<int> eosTokenId() const override { return _eos; }

  [[nodiscard]] size_t vocabSize() const;

  /// Whether an id is an added special token (and so skipped when decoding).
  [[nodiscard]] bool isSpecial(int tokenId) const;

  private:

  std::unique_ptr<tokenizers::Tokenizer> _tokenizer;
  std::optional<int>                     _bos;
  std::optional<int>                     _eos;
  std::unordered_set<int>                _specialIds;
};

} // namespace serving::model
