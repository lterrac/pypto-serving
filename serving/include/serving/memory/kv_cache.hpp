#pragma once

/**
 * Paged KV cache blocks and the prefix cache.
 *
 * The single-pool half of `pypto_serving/serving/memory/kv_cache.py`. Grouped
 * caches (DeepSeek V4) and the torch host pools are not implemented;
 * hasGroups() is false. Block hashes are process-local: a 64-bit mix, not
 * CPython's hash().
 */

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <serving/config/types.hpp>

namespace serving::memory
{

/// Raised when a cache allocation cannot fit in the physical pools.
class KVCacheCapacityError : public std::runtime_error
{
  public:

  explicit KVCacheCapacityError(const std::string &what)
    : std::runtime_error(what)
  {}
};

using BlockHash = uint64_t;

/// Seed of every hash chain. Mirrors the Python module's NONE_HASH sentinel.
inline constexpr BlockHash NONE_HASH = 0x9e3779b97f4a7c15ULL;

/// Chain one full block of tokens onto a parent hash.
[[nodiscard]] BlockHash hashBlockTokens(BlockHash parentHash, const int *tokens, size_t count);

/// Metadata for one physical KV cache page/block.
struct KVCacheBlock
{
  int                      blockId = 0;
  int                      refCnt  = 0;
  std::optional<BlockHash> blockHash;

  // Intrusive links for FreeKVCacheBlockQueue. Only that class touches them.
  KVCacheBlock *prevFree = nullptr;
  KVCacheBlock *nextFree = nullptr;
};

/// Doubly-linked free block queue in eviction order (LRU: oldest at the head).
class FreeKVCacheBlockQueue
{
  public:

  void append(KVCacheBlock *block);
  void appendN(const std::vector<KVCacheBlock *> &blocks);

  /// Take the least recently freed block, or nullptr when empty.
  KVCacheBlock *popleft();

  /// Unlink a block. A block that is not queued is left alone.
  void remove(KVCacheBlock *block);

  [[nodiscard]] int size() const { return _count; }

  private:

  KVCacheBlock *_head  = nullptr;
  KVCacheBlock *_tail  = nullptr;
  int           _count = 0;
};

/// KV block metadata and the prefix cache for one model.
class KvCacheManager
{
  public:

  explicit KvCacheManager(std::optional<int> numBlocks = std::nullopt, int blockSize = 64, bool enablePrefixCache = true);

  [[nodiscard]] int  numFreeBlocks() const { return _freeQueue.size(); }
  [[nodiscard]] int  numBlocks() const { return static_cast<int>(_blocks.size()); }
  [[nodiscard]] int  blockSize() const { return _blockSize; }
  [[nodiscard]] bool enablePrefixCache() const { return _enablePrefixCache; }

  /**
   * Size the pool from the runtime topology. `numBlocks` is the device-reported
   * capacity; a non-positive count is a failed kernel compile and raises the
   * same message as the Python.
   */
  void initialize(const config::RuntimeConfig &runtime, int numBlocks);

  /// Allocate physical blocks, evicting stale prefix hashes; nullopt if short.
  [[nodiscard]] std::optional<std::vector<int>> allocateBlockIds(int numBlocks);

  /// Release one request reference for each listed physical block.
  void releaseBlocksByIds(const std::vector<int> &blockIds);

  /// Release blocks handed out by getComputedBlocks.
  void releaseCachedBlocks(const std::vector<KVCacheBlock *> &blocks);

  /// Chained hashes for every full block in the sequence.
  [[nodiscard]] std::vector<BlockHash> computeBlockHashes(const std::vector<int> &tokenIds) const;

  /**
   * Longest full-block cached prefix for a token sequence, referenced.
   *
   * With no explicit limit, one token is held back so the scheduler can recompute
   * logits without writing into the final shared prefix-cache block.
   */
  [[nodiscard]] std::vector<KVCacheBlock *> getComputedBlocks(const std::vector<int> &tokenIds, std::optional<int> maxCacheHitTokens = std::nullopt);

  /// The same lookup against hashes computed earlier, for a caller that already
  /// holds them. `numTokens` is the length they were computed over.
  [[nodiscard]] std::vector<KVCacheBlock *> getComputedBlocks(const std::vector<BlockHash> &blockHashes, int numTokens, std::optional<int> maxCacheHitTokens = std::nullopt);

  /// Publish blocks `[start, end)` of a request to the prefix cache.
  void cacheBlockIds(const std::vector<int> &blockIds, const std::vector<BlockHash> &blockHashes, int start, int end);

  /// Always false here: the grouped pools are DeepSeek-only and not ported.
  [[nodiscard]] bool hasGroups() const { return false; }
  [[nodiscard]] bool hasEagleGroups() const { return false; }

  // Exposed for tests, which need to assert on refcounts and hash bookkeeping.
  [[nodiscard]] const KVCacheBlock &blockAt(int blockId) const { return _blocks.at(static_cast<size_t>(blockId)); }
  [[nodiscard]] KVCacheBlock       *cachedBlockFor(BlockHash blockHash);

  private:

  void initBlocks(int numBlocks, int blockSize);

  /// Allocate `numBlocks` blocks or nothing at all.
  [[nodiscard]] std::optional<std::vector<KVCacheBlock *>> allocateBlocks(int numBlocks);

  /// Look a block up in the prefix cache and take a reference to it.
  KVCacheBlock *getCachedBlock(BlockHash blockHash);

  /// Publish one full block to the prefix cache.
  void cacheBlock(KVCacheBlock *block, BlockHash blockHash);

  /// Drop one request reference; a block at zero returns to the free queue.
  void release(KVCacheBlock *block);

  /// Chained hashes at this pool's block granularity.
  [[nodiscard]] std::vector<BlockHash> iterBlockHashes(const std::vector<int> &tokenIds) const;

  int  _blockSize         = 64;
  bool _enablePrefixCache = true;

  // Sized exactly once by initBlocks and never resized, so the pointers the free
  // queue and the hash map hold into it stay valid. A second initialize() with
  // different dimensions throws rather than reallocating.
  std::vector<KVCacheBlock> _blocks;

  FreeKVCacheBlockQueue                         _freeQueue;
  std::unordered_map<BlockHash, KVCacheBlock *> _hashToBlock;
};

} // namespace serving::memory
