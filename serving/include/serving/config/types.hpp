#pragma once

/**
 * Runtime, generation and cache configuration.
 *
 * `pypto_serving/config/types.py` in C++: same fields, defaults and validation
 * messages; validate() replaces __post_init__. The torch-bearing types
 * (RuntimeModel, PrefillBatch, ...) stay in Python.
 */

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace serving::config
{

/// Prefill chunk sizes a model may advertise. Mirrors PREFILL_CHUNK_SIZE_CHOICES.
inline constexpr std::array<int, 4> PREFILL_CHUNK_SIZE_CHOICES{1024, 2048, 4096, 8192};

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

/// Static description of one loaded model, as read from its config.json.
struct ModelConfig
{
  std::string        modelId;
  std::string        architecture;
  int                vocabSize             = 0;
  int                hiddenSize            = 0;
  int                intermediateSize      = 0;
  int                numHiddenLayers       = 0;
  int                numAttentionHeads     = 0;
  int                numKeyValueHeads      = 0;
  int                headDim               = 0;
  int                maxPositionEmbeddings = 0;
  double             rmsNormEps            = 0.0;
  double             ropeTheta             = 0.0;
  std::optional<int> bosTokenId            = std::nullopt;
  std::optional<int> eosTokenId            = std::nullopt;
  std::optional<int> padTokenId            = std::nullopt;
  std::string        torchDtype;

  bool operator==(const ModelConfig &) const = default;
};

/// Source-token and physical-storage layout of one cache block.
struct KVCacheSpec
{
  int blockSize     = 0;
  int pageSizeBytes = 0;
  int compressRatio = 1;

  void validate() const
  {
    if (blockSize <= 0) { throw std::invalid_argument("KV cache block_size must be positive"); }
    if (pageSizeBytes <= 0) { throw std::invalid_argument("KV cache page_size_bytes must be positive"); }
    if (compressRatio <= 0) { throw std::invalid_argument("KV cache compress_ratio must be positive"); }
    if (blockSize % compressRatio != 0) { throw std::invalid_argument("KV cache block_size must be divisible by compress_ratio"); }
  }

  /// Number of source tokens represented by one block.
  [[nodiscard]] int tokenCapacity() const { return blockSize; }

  /// Number of physical rows stored for one source-token block.
  [[nodiscard]] int storageBlockSize() const { return blockSize / compressRatio; }

  bool operator==(const KVCacheSpec &) const = default;
};

/// One independently allocated model-specific cache family.
struct KVCacheGroupSpec
{
  std::string      name;
  std::vector<int> layerIndices;
  KVCacheSpec      spec;
  int              maxBlocksPerSeq = 0;

  /// Fixed kernel layouts may expose a physical pool smaller than the serving
  /// configuration's generic max batch size. In that case this is the source of
  /// truth for scheduler-visible block IDs in one cache partition.
  std::optional<int> numBlocks = std::nullopt;

  /// Some distributed kernels combine rank-local data parallelism with expert
  /// parallelism. Each rank then owns an independent cache namespace whose
  /// physical block IDs start at zero. `numPartitions` describes those
  /// namespaces without flattening them into one conflicting block-ID space.
  int numPartitions = 1;

  /// Source-token tail needed to resume a rolling cache group. `nullopt`
  /// denotes a full-history group whose pages are append-only. Physical ring
  /// capacity remains bounded independently by `maxBlocksPerSeq`.
  std::optional<int> slidingWindow = std::nullopt;

  /// EAGLE/MTP cache groups are shifted by one token: the last KV row in a
  /// matched page depends on the first token after that page. Its cache hash
  /// includes that boundary token, and publication waits until it is known.
  bool isEagleGroup = false;

  void validate() const
  {
    spec.validate();
    if (name.empty()) { throw std::invalid_argument("KV cache group name must not be empty"); }
    if (maxBlocksPerSeq <= 0) { throw std::invalid_argument("KV cache max_blocks_per_seq must be positive"); }
    if (numBlocks.has_value() && *numBlocks <= 0) { throw std::invalid_argument("KV cache num_blocks must be positive when specified"); }
    if (numPartitions <= 0) { throw std::invalid_argument("KV cache num_partitions must be positive"); }
    if (slidingWindow.has_value())
    {
      if (*slidingWindow <= 0) { throw std::invalid_argument("KV cache sliding_window must be positive"); }
      if (*slidingWindow % spec.tokenCapacity() != 0) { throw std::invalid_argument("KV cache sliding_window must be a multiple of the block token capacity"); }
      if (*slidingWindow / spec.tokenCapacity() > maxBlocksPerSeq) { throw std::invalid_argument("KV cache sliding_window requires more blocks than max_blocks_per_seq"); }
    }
  }

  bool operator==(const KVCacheGroupSpec &) const = default;
};

/// Runtime limits and device placement for one loaded model.
struct RuntimeConfig
{
  int pageSize     = 64;
  int maxBatchSize = 1;
  int maxSeqLen    = 4096;

  /// Host-side tensor placement. NPU executors manage device memory through the
  /// DistributedWorker internally -- keep this as "cpu".
  std::string        device       = "cpu";
  std::string        kvDtype      = "bfloat16";
  std::string        weightDtype  = "bfloat16";
  std::optional<int> totalKvPages = std::nullopt;

  double npuMemoryUtilization = 0.90;
  int    maxNumBatchedTokens  = 4096;

  std::optional<int> maxPrefillTokensPerRequest = std::nullopt;
  std::vector<int>   prefillChunkSizeChoices    = {};

  bool supportsChunkedPrefillWithSpeculation = true;
  int  speculativePrefixCacheReplayTokens    = 0;
  bool requiresHomogeneousPrefillDecode      = false;

  int maxNewTokens         = 256;
  int numSpeculativeTokens = 0;

  std::vector<KVCacheGroupSpec> kvCacheGroups = {};

  std::vector<int> ringDepPool    = {};
  std::vector<int> ringTaskWindow = {};
  std::vector<int> ringHeap       = {};

  bool operator==(const RuntimeConfig &) const = default;
};

/// Per-layer geometry handed to the model layer.
struct LayerSpec
{
  int layerIdx          = 0;
  int hiddenSize        = 0;
  int intermediateSize  = 0;
  int numAttentionHeads = 0;
  int numKeyValueHeads  = 0;
  int headDim           = 0;

  bool operator==(const LayerSpec &) const = default;
};

/// Per-request sampling knobs resolved from a GenerateConfig.
struct SamplingParams
{
  double                  temperature = 0.0;
  double                  topP        = 1.0;
  std::optional<int>      topK        = std::nullopt;
  std::optional<uint64_t> seed        = std::nullopt;

  bool operator==(const SamplingParams &) const = default;
};

/// Paged KV-cache allocation assigned to one request.
struct KvAllocation
{
  std::string      requestId;
  std::string      modelId;
  std::vector<int> pageIds;
  int              tokensCapacity = 0;
  int              tokensUsed     = 0;

  bool operator==(const KvAllocation &) const = default;
};

/// Lifecycle of one request as the worker sees it.
enum class RequestStateStatus
{
  Waiting,
  Prefill,
  Decode,
  Finished,
  Aborted,
  Error,
};

/// Mutable per-request state carried alongside a batch.
struct RequestState
{
  std::string                   requestId;
  std::string                   modelId;
  std::string                   prompt;
  std::vector<int>              promptTokenIds;
  std::vector<int>              generatedTokenIds;
  std::optional<SamplingParams> samplingParams  = std::nullopt;
  RequestStateStatus            status          = RequestStateStatus::Waiting;
  int                           maxNewTokens    = 0;
  std::vector<std::string>      stopStrings     = {};
  std::optional<int>            eosTokenId      = std::nullopt;
  int                           seqLen          = 0;
  int                           numPromptTokens = 0;
  std::optional<KvAllocation>   kvAllocation    = std::nullopt;
  std::string                   outputText;

  bool operator==(const RequestState &) const = default;
};

/// Final text, generated IDs, and stop reason for one request.
struct GenerateResult
{
  std::string      text;
  std::vector<int> tokenIds;
  std::string      finishReason;

  bool operator==(const GenerateResult &) const = default;
};

} // namespace serving::config
