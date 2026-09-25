#pragma once

/**
 * What the engine reads out of `pypto_serving/config/types.py`.
 *
 * Only these two: the rest of that module -- the per-request and per-model
 * descriptions the model layer passes around, and the torch-bearing batch types
 * -- stays in Python until something in C++ needs it.
 */

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace serving::config
{

/// User-facing options that control text generation.
struct GenerateConfig
{
  int                      maxNewTokens = 256;
  double                   temperature  = 0.0;
  double                   topP         = 1.0;
  std::optional<int>       topK         = std::nullopt;
  std::optional<uint64_t>  seed         = std::nullopt;
  std::vector<std::string> stop         = {};
  bool                     stream       = false;
  bool                     ignoreEos    = false;

  bool operator==(const GenerateConfig &) const = default;
};

/// The KV pool's geometry. Scheduling limits live in sched::SchedulerConfig;
/// carrying them here too meant setting the same number twice.
struct RuntimeConfig
{
  /// KV tokens per block. 128 for the Qwen kernels.
  int pageSize = 64;
  /// Compiled batch width of the model's kernels.
  int maxBatchSize = 1;
  int maxSeqLen    = 4096;

  bool operator==(const RuntimeConfig &) const = default;
};

} // namespace serving::config
