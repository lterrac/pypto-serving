#include <serving/engine/detokenizer.hpp>

#include <algorithm>

namespace serving::engine
{

namespace
{

/// The sliding context window, in tokens, kept behind the read offset.
constexpr int kPrefixWindowTokens = 3;

} // namespace

bool endsWithReplacementChar(const std::string &text)
{
  // U+FFFD is EF BF BD in UTF-8.
  if (text.size() < 3) { return false; }
  const auto tail = text.substr(text.size() - 3);
  return static_cast<unsigned char>(tail[0]) == 0xEF && static_cast<unsigned char>(tail[1]) == 0xBF && static_cast<unsigned char>(tail[2]) == 0xBD;
}

const std::string &IncrementalDetokenizer::append(const std::vector<int> &outputIds)
{
  if (outputIds.empty()) { return _text; }

  const int total = static_cast<int>(outputIds.size());
  // Decode a short window: [prefixOffset:] gives the context that makes the
  // delta render identically to a full decode; the delta is the tail beyond what
  // [prefixOffset:readOffset] already covered.
  const int prefixBegin = std::clamp(_prefixOffset, 0, total);
  const int prefixEnd   = std::clamp(_readOffset, prefixBegin, total);

  const std::vector<int> prefixIds(outputIds.begin() + prefixBegin, outputIds.begin() + prefixEnd);
  const std::vector<int> newIds(outputIds.begin() + prefixBegin, outputIds.end());

  const std::string prefixText = prefixIds.empty() ? std::string{} : _tokenizer.decode(prefixIds);
  const std::string newText    = _tokenizer.decode(newIds);

  if (newText.size() <= prefixText.size() || endsWithReplacementChar(newText))
  {
    // No new complete text yet -- mid multi-token character. Wait for more
    // tokens WITHOUT advancing the offsets, or the character would be lost.
    return _text;
  }

  _text.append(newText, prefixText.size(), std::string::npos);
  _readOffset   = total;
  _prefixOffset = std::max(0, _readOffset - kPrefixWindowTokens);
  return _text;
}

const std::string &IncrementalDetokenizer::finalize(const std::vector<int> &outputIds)
{
  if (outputIds.empty()) { return _text; }
  _text         = _tokenizer.decode(outputIds);
  _readOffset   = static_cast<int>(outputIds.size());
  _prefixOffset = std::max(0, _readOffset - kPrefixWindowTokens);
  return _text;
}

} // namespace serving::engine
