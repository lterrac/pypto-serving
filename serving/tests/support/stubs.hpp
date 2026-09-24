#pragma once

/// Stand-ins shared by the engine and server tests: a tokenizer that is its own
/// inverse and an executor that emits a fixed script, so a test can drive the
/// engine end to end with no model.

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <serving/engine/engine.hpp>
#include <serving/engine/executor.hpp>
#include <serving/model/tokenizer.hpp>

namespace serving::testing
{

/// One token per byte, so an expected string is readable in the test.
class CharTokenizer : public model::TokenizerAdapter
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

/// Emits `script` cyclically, one token per decode slot and one per prefill
/// chunk that completes its prompt -- the rule the real worker follows. Set
/// `failWith` to make a step fail.
class ScriptedExecutor : public engine::ModelExecutor
{
  public:

  explicit ScriptedExecutor(std::vector<int> script, int numPages = 64)
    : _script(std::move(script)),
      _numPages(numPages)
  {}

  /// Tokens spelled as text, which is how the server tests read best.
  [[nodiscard]] static std::vector<int> fromText(const std::string &text)
  {
    std::vector<int> ids;
    for (const char c : text) { ids.push_back(static_cast<int>(c)); }
    return ids;
  }

  int registerModel() override
  {
    registered += 1;
    return _numPages;
  }

  engine::StepResult executeStep(const engine::StepCommand &command) override
  {
    steps += 1;
    lastPrefillChunks = 0;
    for (const auto &item : command.prefill) { lastPrefillChunks += static_cast<int>(item.chunkTokens.size()); }

    engine::StepResult result;
    if (!failWith.empty())
    {
      result.error = failWith;
      return result;
    }

    for (const auto &item : command.prefill)
    {
      const auto it      = promptLengths.find(item.requestId);
      const int  wanted  = it == promptLengths.end() ? promptLength : it->second;
      const int  covered = item.numComputedTokens + static_cast<int>(item.chunkTokens.size());
      if (covered >= wanted) { result.newTokens[item.requestId] = {next(item.requestId)}; }
    }
    for (const auto &item : command.decode) { result.newTokens[item.requestId] = {next(item.requestId)}; }
    {
      const std::lock_guard<std::mutex> lock(_decodedMutex);
      for (const auto &item : command.decode) { _decodedIds.push_back(item.requestId); }
    }
    return result;
  }

  void close() override { closed += 1; }

  /// Prompt length per request; `promptLength` is the default for ids not listed.
  std::map<std::string, int> promptLengths;
  int                        promptLength = 0;
  std::string                failWith;
  std::atomic<int>           steps{0};
  std::atomic<int>           registered{0};
  std::atomic<int>           closed{0};
  std::atomic<int>           lastPrefillChunks{0};

  /// Every decode slot asked for, in order. Written on the engine thread.
  std::vector<std::string> decodedIds()
  {
    const std::lock_guard<std::mutex> lock(_decodedMutex);
    return _decodedIds;
  }

  void clearDecodedIds()
  {
    const std::lock_guard<std::mutex> lock(_decodedMutex);
    _decodedIds.clear();
  }

  private:

  int next(const std::string &requestId)
  {
    const size_t index = _emitted[requestId]++;
    return _script[index % _script.size()];
  }

  std::mutex                    _decodedMutex;
  std::vector<std::string>      _decodedIds;
  std::vector<int>              _script;
  int                           _numPages;
  std::map<std::string, size_t> _emitted;
};

/// Every update a request produces, in order.
inline std::vector<engine::TokenOutput> drain(const std::shared_ptr<engine::RequestStream> &stream)
{
  std::vector<engine::TokenOutput> updates;
  while (auto update = stream->pop()) { updates.push_back(*update); }
  return updates;
}

} // namespace serving::testing
