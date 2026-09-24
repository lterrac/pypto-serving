#pragma once

/**
 * The model as the engine drives it: a StepCommand of prefill chunks and
 * decode slots in, a StepResult of tokens out. Token ids, block ids and
 * lengths only.
 */

#include <string>
#include <unordered_map>
#include <vector>

namespace serving::engine
{

/// One request's prefill chunk for this step.
struct PrefillItem
{
  std::string      requestId;
  std::vector<int> chunkTokens;
  /// Tokens already computed before this chunk (a prefix-cache hit, or earlier chunks).
  int numComputedTokens = 0;
  /// Full prompt length. A chunk samples a token only when it reaches the end of
  /// the prompt; the executor needs this to know which chunks did.
  int              promptLen = 0;
  std::vector<int> blockIds;
};

/// One request's decode slot for this step.
struct DecodeItem
{
  std::string requestId;
  int         lastToken = 0;
  /// Sequence length including the token being fed in.
  int              seqLen = 0;
  std::vector<int> blockIds;
};

struct StepCommand
{
  std::vector<PrefillItem> prefill;
  std::vector<DecodeItem>  decode;

  [[nodiscard]] bool empty() const { return prefill.empty() && decode.empty(); }
};

struct StepResult
{
  /// Tokens sampled this step, by request id. A prefill chunk that did not
  /// complete its prompt samples nothing and simply does not appear.
  std::unordered_map<std::string, std::vector<int>> newTokens;

  /// Non-empty if the step failed; the engine fails the affected requests.
  std::string error;
};

class ModelExecutor
{
  public:

  virtual ~ModelExecutor() = default;

  /// Load the model and return the KV page count the scheduler may hand out.
  /// Expensive: kernel compile plus weight staging.
  virtual int registerModel() = 0;

  virtual StepResult executeStep(const StepCommand &command) = 0;

  virtual void close() {}
};

} // namespace serving::engine
