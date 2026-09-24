#pragma once

/**
 * The tokenizer interface the engine uses. Dependency-free; hf_tokenizer.hpp
 * is the tokenizers-cpp implementation.
 */

#include <optional>
#include <string>
#include <vector>

namespace serving::model
{

/// Minimal tokenizer surface required by the generation engine.
class TokenizerAdapter
{
  public:

  virtual ~TokenizerAdapter() = default;

  /// Encode text into token ids, without adding prompt specials.
  [[nodiscard]] virtual std::vector<int> encode(const std::string &text) const = 0;

  /// Decode token ids back into text.
  [[nodiscard]] virtual std::string decode(const std::vector<int> &tokenIds, bool skipSpecialTokens = true) const = 0;

  [[nodiscard]] virtual std::optional<int> bosTokenId() const { return std::nullopt; }
  [[nodiscard]] virtual std::optional<int> eosTokenId() const { return std::nullopt; }
};

} // namespace serving::model
