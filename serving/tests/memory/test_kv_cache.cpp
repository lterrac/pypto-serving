#include <gtest/gtest.h>

#include <serving/memory/kv_cache.hpp>

using serving::config::RuntimeConfig;
using serving::memory::BlockHash;
using serving::memory::KVCacheBlock;
using serving::memory::KvCacheManager;
using serving::memory::NONE_HASH;

namespace
{

/// Allocate and immediately assert the allocation succeeded.
std::vector<int> mustAllocate(KvCacheManager &manager, int count)
{
  auto ids = manager.allocateBlockIds(count);
  EXPECT_TRUE(ids.has_value()) << "expected " << count << " blocks to be allocatable";
  return ids.value_or(std::vector<int>{});
}

} // namespace

// ---------------------------------------------------------------------------
// Pool sizing
// ---------------------------------------------------------------------------

TEST(KvCacheManagerTest, StartsFullyFree)
{
  KvCacheManager manager(8, 2, false);
  EXPECT_EQ(manager.numBlocks(), 8);
  EXPECT_EQ(manager.numFreeBlocks(), 8);
  EXPECT_EQ(manager.blockSize(), 2);
}

TEST(KvCacheManagerTest, InitializeSizesFromTheRuntimeConfig)
{
  RuntimeConfig runtime;
  runtime.pageSize = 128;

  KvCacheManager manager;
  manager.initialize(runtime, 1302); // the count a real Qwen3-14B run reports
  EXPECT_EQ(manager.numBlocks(), 1302);
  EXPECT_EQ(manager.blockSize(), 128);
  EXPECT_EQ(manager.numFreeBlocks(), 1302);
}

TEST(KvCacheManagerTest, InitializeRejectsANonPositiveBlockCount)
{
  // This is what a failed kernel compile in the worker looks like from here, so
  // the message matters as much as the throw.
  RuntimeConfig  runtime;
  KvCacheManager manager;
  try
  {
    manager.initialize(runtime, 0);
    FAIL() << "expected a throw";
  }
  catch (const std::runtime_error &e)
  {
    EXPECT_STREQ(e.what(), "Worker reported invalid KV cache block count: 0");
  }
}

TEST(KvCacheManagerTest, ReinitializingWithDifferentDimensionsThrows)
{
  RuntimeConfig runtime;
  runtime.pageSize = 2;

  KvCacheManager manager(8, 2, false);
  EXPECT_NO_THROW(manager.initialize(runtime, 8)); // same dimensions: a no-op
  runtime.pageSize = 4;
  EXPECT_THROW(manager.initialize(runtime, 8), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Allocation and reference counting
// ---------------------------------------------------------------------------

TEST(KvCacheManagerTest, AllocationTakesBlocksOutOfTheFreePool)
{
  KvCacheManager manager(4, 2, false);
  const auto     ids = mustAllocate(manager, 3);
  EXPECT_EQ(ids.size(), 3u);
  EXPECT_EQ(manager.numFreeBlocks(), 1);
  for (int id : ids) { EXPECT_EQ(manager.blockAt(id).refCnt, 1); }
}

TEST(KvCacheManagerTest, AllocationIsAllOrNothing)
{
  KvCacheManager manager(4, 2, false);
  EXPECT_FALSE(manager.allocateBlockIds(5).has_value());
  // A refused allocation must not have consumed anything.
  EXPECT_EQ(manager.numFreeBlocks(), 4);
}

TEST(KvCacheManagerTest, AllocatingZeroBlocksSucceedsWithNothing)
{
  KvCacheManager manager(4, 2, false);
  const auto     ids = manager.allocateBlockIds(0);
  ASSERT_TRUE(ids.has_value());
  EXPECT_TRUE(ids->empty());
  EXPECT_EQ(manager.numFreeBlocks(), 4);
}

TEST(KvCacheManagerTest, ReleaseReturnsBlocksToThePool)
{
  KvCacheManager manager(4, 2, false);
  const auto     ids = mustAllocate(manager, 4);
  EXPECT_EQ(manager.numFreeBlocks(), 0);
  manager.releaseBlocksByIds(ids);
  EXPECT_EQ(manager.numFreeBlocks(), 4);
  for (int id : ids) { EXPECT_EQ(manager.blockAt(id).refCnt, 0); }
}

TEST(KvCacheManagerTest, ReleasingAnUnreferencedBlockIsANoOp)
{
  KvCacheManager manager(2, 2, false);
  manager.releaseBlocksByIds({0, 0, 0});
  // Nothing was ever allocated, so the pool must not grow past its size.
  EXPECT_EQ(manager.numFreeBlocks(), 2);
}

TEST(KvCacheManagerTest, BlocksAreReusedInEvictionOrder)
{
  KvCacheManager manager(4, 2, false);
  const auto     first = mustAllocate(manager, 4);
  ASSERT_EQ(first.size(), 4u);

  // Free two, oldest first; the next allocations must come back in that order.
  manager.releaseBlocksByIds({first[2]});
  manager.releaseBlocksByIds({first[0]});
  const auto reused = mustAllocate(manager, 2);
  ASSERT_EQ(reused.size(), 2u);
  EXPECT_EQ(reused[0], first[2]);
  EXPECT_EQ(reused[1], first[0]);
}

// ---------------------------------------------------------------------------
// Block hashing
// ---------------------------------------------------------------------------

TEST(KvCacheManagerTest, HashesCoverOnlyFullBlocks)
{
  KvCacheManager manager(8, 4, true);
  EXPECT_EQ(manager.computeBlockHashes({1, 2, 3}).size(), 0u);       // short of one block
  EXPECT_EQ(manager.computeBlockHashes({1, 2, 3, 4}).size(), 1u);    // exactly one
  EXPECT_EQ(manager.computeBlockHashes({1, 2, 3, 4, 5}).size(), 1u); // partial tail ignored
  EXPECT_EQ(manager.computeBlockHashes({1, 2, 3, 4, 5, 6, 7, 8}).size(), 2u);
}

TEST(KvCacheManagerTest, HashesAreChainedSoAPrefixIsShared)
{
  KvCacheManager manager(8, 2, true);
  const auto     a = manager.computeBlockHashes({1, 2, 3, 4, 5, 6});
  const auto     b = manager.computeBlockHashes({1, 2, 3, 4, 9, 9});
  ASSERT_EQ(a.size(), 3u);
  ASSERT_EQ(b.size(), 3u);
  // Shared prefix hashes identically; the block where they diverge does not.
  EXPECT_EQ(a[0], b[0]);
  EXPECT_EQ(a[1], b[1]);
  EXPECT_NE(a[2], b[2]);
}

TEST(KvCacheManagerTest, ChainingMakesPositionMatter)
{
  KvCacheManager manager(8, 2, true);
  // The same block content at a different offset must not collide, or a prefix
  // hit would splice in KV computed under a different history.
  const auto first  = manager.computeBlockHashes({7, 7, 1, 1});
  const auto second = manager.computeBlockHashes({1, 1, 7, 7});
  ASSERT_EQ(first.size(), 2u);
  ASSERT_EQ(second.size(), 2u);
  EXPECT_NE(first[0], second[1]);
  EXPECT_NE(first[1], second[0]);
}

// ---------------------------------------------------------------------------
// Prefix cache
// ---------------------------------------------------------------------------

TEST(KvCacheManagerTest, PublishedBlocksAreFoundAgain)
{
  KvCacheManager         manager(8, 2, true);
  const std::vector<int> prompt{1, 2, 3, 4, 5};
  const auto             hashes = manager.computeBlockHashes(prompt);
  ASSERT_EQ(hashes.size(), 2u);

  const auto ids = mustAllocate(manager, 2);
  manager.cacheBlockIds(ids, hashes, 0, 2);

  // maxCacheHitTokens defaults to len-1 = 4 tokens = 2 blocks.
  auto hits = manager.getComputedBlocks(prompt);
  ASSERT_EQ(hits.size(), 2u);
  EXPECT_EQ(hits[0]->blockId, ids[0]);
  EXPECT_EQ(hits[1]->blockId, ids[1]);
  manager.releaseCachedBlocks(hits);
}

TEST(KvCacheManagerTest, TheLastTokenIsHeldBackByDefault)
{
  KvCacheManager         manager(8, 2, true);
  const std::vector<int> prompt{1, 2, 3, 4};
  const auto             hashes = manager.computeBlockHashes(prompt);
  const auto             ids    = mustAllocate(manager, 2);
  manager.cacheBlockIds(ids, hashes, 0, 2);

  // Both blocks are published, but the default limit of len-1 == 3 tokens only
  // permits one whole block -- the scheduler must be left something to compute.
  auto hits = manager.getComputedBlocks(prompt);
  EXPECT_EQ(hits.size(), 1u);
  manager.releaseCachedBlocks(hits);

  // Lifting the limit exposes both.
  auto all = manager.getComputedBlocks(prompt, 4);
  EXPECT_EQ(all.size(), 2u);
  manager.releaseCachedBlocks(all);
}

TEST(KvCacheManagerTest, AHitRescuesABlockFromTheFreeQueue)
{
  KvCacheManager         manager(4, 2, true);
  const std::vector<int> prompt{1, 2, 3, 4};
  const auto             hashes = manager.computeBlockHashes(prompt);
  const auto             ids    = mustAllocate(manager, 2);
  manager.cacheBlockIds(ids, hashes, 0, 2);
  manager.releaseBlocksByIds(ids); // published but unreferenced: still cached
  EXPECT_EQ(manager.numFreeBlocks(), 4);

  auto hits = manager.getComputedBlocks(prompt, 4);
  ASSERT_EQ(hits.size(), 2u);
  // Taking a reference must pull them back out, or they could be handed to
  // someone else while this request still depends on their contents.
  EXPECT_EQ(manager.numFreeBlocks(), 2);
  for (const KVCacheBlock *block : hits) { EXPECT_EQ(block->refCnt, 1); }

  manager.releaseCachedBlocks(hits);
  EXPECT_EQ(manager.numFreeBlocks(), 4);
}

TEST(KvCacheManagerTest, EvictionForgetsTheStaleHash)
{
  KvCacheManager         manager(2, 2, true);
  const std::vector<int> prompt{1, 2, 3, 4};
  const auto             hashes = manager.computeBlockHashes(prompt);
  const auto             ids    = mustAllocate(manager, 2);
  manager.cacheBlockIds(ids, hashes, 0, 2);
  manager.releaseBlocksByIds(ids);

  ASSERT_NE(manager.cachedBlockFor(hashes[0]), nullptr);

  // Reallocating the whole pool must evict both published blocks and drop their
  // hashes, otherwise a later lookup would return a block holding other tokens.
  const auto reused = mustAllocate(manager, 2);
  EXPECT_EQ(reused.size(), 2u);
  EXPECT_EQ(manager.cachedBlockFor(hashes[0]), nullptr);
  EXPECT_EQ(manager.cachedBlockFor(hashes[1]), nullptr);
  EXPECT_TRUE(manager.getComputedBlocks(prompt, 4).empty());
}

TEST(KvCacheManagerTest, RepublishingABlockDropsItsPreviousHash)
{
  KvCacheManager manager(4, 2, true);
  const auto     firstHashes  = manager.computeBlockHashes({1, 2});
  const auto     secondHashes = manager.computeBlockHashes({3, 4});
  const auto     ids          = mustAllocate(manager, 1);

  manager.cacheBlockIds(ids, firstHashes, 0, 1);
  ASSERT_NE(manager.cachedBlockFor(firstHashes[0]), nullptr);

  manager.cacheBlockIds(ids, secondHashes, 0, 1);
  EXPECT_EQ(manager.cachedBlockFor(firstHashes[0]), nullptr);
  EXPECT_NE(manager.cachedBlockFor(secondHashes[0]), nullptr);
}

TEST(KvCacheManagerTest, CacheBlockIdsIgnoresARangeBeyondTheInputs)
{
  KvCacheManager manager(4, 2, true);
  const auto     hashes = manager.computeBlockHashes({1, 2});
  const auto     ids    = mustAllocate(manager, 1);
  EXPECT_NO_THROW(manager.cacheBlockIds(ids, hashes, 0, 99));
  EXPECT_NE(manager.cachedBlockFor(hashes[0]), nullptr);
}

TEST(KvCacheManagerTest, DisablingThePrefixCacheSuppressesBothSides)
{
  KvCacheManager         manager(8, 2, false);
  const std::vector<int> prompt{1, 2, 3, 4};
  const auto             hashes = manager.computeBlockHashes(prompt);
  const auto             ids    = mustAllocate(manager, 2);

  manager.cacheBlockIds(ids, hashes, 0, 2);
  EXPECT_EQ(manager.cachedBlockFor(hashes[0]), nullptr);
  EXPECT_TRUE(manager.getComputedBlocks(prompt, 4).empty());
}

TEST(KvCacheManagerTest, LookupStopsAtTheFirstMiss)
{
  KvCacheManager         manager(8, 2, true);
  const std::vector<int> prompt{1, 2, 3, 4, 5, 6};
  const auto             hashes = manager.computeBlockHashes(prompt);
  ASSERT_EQ(hashes.size(), 3u);
  const auto ids = mustAllocate(manager, 3);

  // Publish blocks 0 and 2 but not 1: the run of hits must stop at the gap
  // rather than skipping it, since block 2's KV depends on block 1's.
  manager.cacheBlockIds(ids, hashes, 0, 1);
  manager.cacheBlockIds(ids, hashes, 2, 3);

  auto hits = manager.getComputedBlocks(prompt, 6);
  EXPECT_EQ(hits.size(), 1u);
  manager.releaseCachedBlocks(hits);
}

// Grouped pools are not implemented.

TEST(KvCacheManagerTest, ReportsNoCacheGroups)
{
  KvCacheManager manager(4, 2, true);
  EXPECT_FALSE(manager.hasGroups());
}
