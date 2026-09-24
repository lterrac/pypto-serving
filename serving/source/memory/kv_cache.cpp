#include <serving/memory/kv_cache.hpp>

namespace serving::memory
{

namespace
{

/// 64-bit mix, in the shape of boost::hash_combine.
BlockHash mix(BlockHash seed, uint64_t value)
{
  seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
  return seed;
}

} // namespace

BlockHash hashBlockTokens(BlockHash parentHash, const int *tokens, size_t count)
{
  BlockHash hash = mix(parentHash, static_cast<uint64_t>(count));
  for (size_t i = 0; i < count; ++i) { hash = mix(hash, static_cast<uint64_t>(static_cast<int64_t>(tokens[i]))); }
  return hash;
}

// ---------------------------------------------------------------------------
// FreeKVCacheBlockQueue
// ---------------------------------------------------------------------------

void FreeKVCacheBlockQueue::append(KVCacheBlock *block)
{
  block->prevFree = _tail;
  block->nextFree = nullptr;
  if (_tail != nullptr) { _tail->nextFree = block; }
  else { _head = block; }
  _tail = block;
  _count += 1;
}

void FreeKVCacheBlockQueue::appendN(const std::vector<KVCacheBlock *> &blocks)
{
  for (KVCacheBlock *block : blocks) { append(block); }
}

KVCacheBlock *FreeKVCacheBlockQueue::popleft()
{
  if (_head == nullptr) { return nullptr; }
  KVCacheBlock *block = _head;
  remove(block);
  return block;
}

void FreeKVCacheBlockQueue::remove(KVCacheBlock *block)
{
  // A block with no links that is neither end of the queue is simply not queued.
  if (block != _head && block != _tail && block->prevFree == nullptr && block->nextFree == nullptr) { return; }

  KVCacheBlock *prev = block->prevFree;
  KVCacheBlock *next = block->nextFree;
  if (prev != nullptr) { prev->nextFree = next; }
  else { _head = next; }
  if (next != nullptr) { next->prevFree = prev; }
  else { _tail = prev; }
  block->prevFree = nullptr;
  block->nextFree = nullptr;
  _count -= 1;
}

// ---------------------------------------------------------------------------
// KvCacheManager
// ---------------------------------------------------------------------------

KvCacheManager::KvCacheManager(std::optional<int> numBlocks, int blockSize, bool enablePrefixCache)
  : _blockSize(blockSize),
    _enablePrefixCache(enablePrefixCache)
{
  if (numBlocks.has_value()) { initBlocks(*numBlocks, blockSize); }
}

void KvCacheManager::initialize(const config::RuntimeConfig &runtime, int numBlocks)
{
  if (numBlocks <= 0)
  {
    // Same message as the Python: it is what a failed kernel compile surfaces as.
    throw std::runtime_error("Worker reported invalid KV cache block count: " + std::to_string(numBlocks));
  }
  if (!runtime.kvCacheGroups.empty())
  {
    throw std::runtime_error("grouped KV cache pools are not supported by this build "
                             "(kv_cache_groups is DeepSeek-only and out of scope for the C++ port)");
  }
  initBlocks(numBlocks, runtime.pageSize);
}

void KvCacheManager::initBlocks(int numBlocks, int blockSize)
{
  if (!_blocks.empty())
  {
    if (static_cast<int>(_blocks.size()) != numBlocks || _blockSize != blockSize) { throw std::invalid_argument("KV block pool is already initialized with different dimensions"); }
    return;
  }
  _blockSize = blockSize;
  _blocks.resize(static_cast<size_t>(numBlocks));
  for (int i = 0; i < numBlocks; ++i)
  {
    _blocks[static_cast<size_t>(i)].blockId = i;
    _freeQueue.append(&_blocks[static_cast<size_t>(i)]);
  }
}

std::optional<std::vector<KVCacheBlock *>> KvCacheManager::allocateBlocks(int numBlocks)
{
  if (numBlocks <= 0) { return std::vector<KVCacheBlock *>{}; }
  if (numFreeBlocks() < numBlocks) { return std::nullopt; }

  std::vector<KVCacheBlock *> blocks;
  blocks.reserve(static_cast<size_t>(numBlocks));
  for (int i = 0; i < numBlocks; ++i)
  {
    KVCacheBlock *block = _freeQueue.popleft();
    if (block == nullptr)
    {
      for (KVCacheBlock *allocated : blocks) { release(allocated); }
      return std::nullopt;
    }
    if (block->blockHash.has_value())
    {
      _hashToBlock.erase(*block->blockHash);
      block->blockHash.reset();
    }
    block->refCnt = 1;
    blocks.push_back(block);
  }
  return blocks;
}

std::optional<std::vector<int>> KvCacheManager::allocateBlockIds(int numBlocks)
{
  auto blocks = allocateBlocks(numBlocks);
  if (!blocks.has_value()) { return std::nullopt; }
  std::vector<int> ids;
  ids.reserve(blocks->size());
  for (KVCacheBlock *block : *blocks) { ids.push_back(block->blockId); }
  return ids;
}

void KvCacheManager::releaseBlocksByIds(const std::vector<int> &blockIds)
{
  for (int blockId : blockIds) { release(&_blocks.at(static_cast<size_t>(blockId))); }
}

void KvCacheManager::releaseCachedBlocks(const std::vector<KVCacheBlock *> &blocks)
{
  for (KVCacheBlock *block : blocks) { release(block); }
}

KVCacheBlock *KvCacheManager::getCachedBlock(BlockHash blockHash)
{
  if (!_enablePrefixCache) { return nullptr; }
  const auto it = _hashToBlock.find(blockHash);
  if (it == _hashToBlock.end()) { return nullptr; }
  KVCacheBlock *block = it->second;
  // A cached block sitting at zero references is still in the free queue; taking
  // a reference must pull it out so it cannot be handed to someone else.
  if (block->refCnt == 0) { _freeQueue.remove(block); }
  block->refCnt += 1;
  return block;
}

KVCacheBlock *KvCacheManager::cachedBlockFor(BlockHash blockHash)
{
  const auto it = _hashToBlock.find(blockHash);
  return it == _hashToBlock.end() ? nullptr : it->second;
}

void KvCacheManager::cacheBlock(KVCacheBlock *block, BlockHash blockHash)
{
  if (!_enablePrefixCache) { return; }
  if (block->blockHash.has_value()) { _hashToBlock.erase(*block->blockHash); }
  block->blockHash        = blockHash;
  _hashToBlock[blockHash] = block;
}

void KvCacheManager::cacheBlockIds(const std::vector<int> &blockIds, const std::vector<BlockHash> &blockHashes, int start, int end)
{
  if (!_enablePrefixCache) { return; }
  for (int idx = start; idx < end; ++idx)
  {
    if (idx < 0 || idx >= static_cast<int>(blockHashes.size()) || idx >= static_cast<int>(blockIds.size())) { break; }
    cacheBlock(&_blocks.at(static_cast<size_t>(blockIds[static_cast<size_t>(idx)])), blockHashes[static_cast<size_t>(idx)]);
  }
}

void KvCacheManager::release(KVCacheBlock *block)
{
  if (block->refCnt <= 0) { return; }
  block->refCnt -= 1;
  if (block->refCnt == 0) { _freeQueue.append(block); }
}

std::vector<BlockHash> KvCacheManager::iterBlockHashes(const std::vector<int> &tokenIds) const
{
  std::vector<BlockHash> hashes;
  if (_blockSize <= 0) { return hashes; }
  const int numFullBlocks = static_cast<int>(tokenIds.size()) / _blockSize;
  hashes.reserve(static_cast<size_t>(numFullBlocks));
  BlockHash parentHash = NONE_HASH;
  for (int i = 0; i < numFullBlocks; ++i)
  {
    parentHash = hashBlockTokens(parentHash, tokenIds.data() + static_cast<size_t>(i) * static_cast<size_t>(_blockSize), static_cast<size_t>(_blockSize));
    hashes.push_back(parentHash);
  }
  return hashes;
}

std::vector<BlockHash> KvCacheManager::computeBlockHashes(const std::vector<int> &tokenIds) const { return iterBlockHashes(tokenIds); }

std::vector<KVCacheBlock *> KvCacheManager::getComputedBlocks(const std::vector<int> &tokenIds, std::optional<int> maxCacheHitTokens)
{
  if (!_enablePrefixCache) { return {}; }
  return getComputedBlocks(iterBlockHashes(tokenIds), static_cast<int>(tokenIds.size()), maxCacheHitTokens);
}

std::vector<KVCacheBlock *> KvCacheManager::getComputedBlocks(const std::vector<BlockHash> &blockHashes, int numTokens, std::optional<int> maxCacheHitTokens)
{
  std::vector<KVCacheBlock *> hitBlocks;
  if (!_enablePrefixCache) { return hitBlocks; }

  // Hold back the last token by default: the scheduler needs to recompute logits
  // without writing into the final shared prefix-cache block.
  const int limitTokens  = maxCacheHitTokens.value_or(std::max(0, numTokens - 1));
  const int maxHitBlocks = std::max(0, limitTokens) / _blockSize;

  for (const BlockHash blockHash : blockHashes)
  {
    if (static_cast<int>(hitBlocks.size()) >= maxHitBlocks) { break; }
    KVCacheBlock *block = getCachedBlock(blockHash);
    if (block == nullptr) { break; }
    hitBlocks.push_back(block);
  }
  return hitBlocks;
}

} // namespace serving::memory
