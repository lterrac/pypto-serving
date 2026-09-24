#pragma once

/**
 * Incremental detokenization for a streaming request.
 *
 * Decodes a three-token window per step, so the work over a generation is
 * linear. A decode ending in U+FFFD is a multi-token character in flight and
 * is withheld; finalize() re-decodes the whole output so the final text equals
 * an offline decode. Fewer than three tokens of context corrupts spacing for
 * SentencePiece and byte-level BPE.
 */

#include <string>
#include <vector>

#include <serving/model/tokenizer.hpp>

namespace serving::engine
{

/// Per-request incremental detokenization state.
class IncrementalDetokenizer
{
  public:

  explicit IncrementalDetokenizer(const model::TokenizerAdapter &tokenizer)
    : _tokenizer(tokenizer)
  {}

  /**
   * Fold any newly-completed text into the running string and return it.
   *
   * `outputIds` is the request's full output so far; only the tail beyond what
   * has already been rendered is decoded.
   */
  const std::string &append(const std::vector<int> &outputIds);

  /**
   * Authoritative final text for a finished request.
   *
   * One full decode, O(N) once per request, so a trailing incomplete character
   * that `append` withheld is still resolved the way an offline decode renders it.
   */
  const std::string &finalize(const std::vector<int> &outputIds);

  [[nodiscard]] const std::string &text() const { return _text; }
  [[nodiscard]] int                prefixOffset() const { return _prefixOffset; }
  [[nodiscard]] int                readOffset() const { return _readOffset; }

  private:

  const model::TokenizerAdapter &_tokenizer;

  std::string _text;
  int         _prefixOffset = 0;
  int         _readOffset   = 0;
};

/// Whether a UTF-8 string ends with the replacement character U+FFFD.
[[nodiscard]] bool endsWithReplacementChar(const std::string &text);

} // namespace serving::engine
