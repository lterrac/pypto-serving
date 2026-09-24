#include <gtest/gtest.h>

#include <serving/config/types.hpp>

using serving::config::GenerateConfig;
using serving::config::KVCacheGroupSpec;
using serving::config::KVCacheSpec;
using serving::config::RuntimeConfig;

namespace
{

KVCacheSpec makeSpec(int blockSize = 128, int compressRatio = 1)
{
  KVCacheSpec spec;
  spec.blockSize     = blockSize;
  spec.pageSizeBytes = 4096;
  spec.compressRatio = compressRatio;
  return spec;
}

KVCacheGroupSpec makeGroup(const KVCacheSpec &spec, int maxBlocksPerSeq = 8)
{
  KVCacheGroupSpec group;
  group.name            = "default";
  group.layerIndices    = {0, 1};
  group.spec            = spec;
  group.maxBlocksPerSeq = maxBlocksPerSeq;
  return group;
}

} // namespace

TEST(GenerateConfigTest, DefaultsMatchThePythonDataclass)
{
  const GenerateConfig config;
  EXPECT_EQ(config.maxNewTokens, 256);
  EXPECT_DOUBLE_EQ(config.temperature, 0.0);
  EXPECT_DOUBLE_EQ(config.topP, 1.0);
  EXPECT_FALSE(config.topK.has_value());
  EXPECT_FALSE(config.seed.has_value());
  EXPECT_TRUE(config.stop.empty());
  EXPECT_FALSE(config.stream);
  EXPECT_FALSE(config.ignoreEos);
}

TEST(RuntimeConfigTest, DefaultsMatchThePythonDataclass)
{
  const RuntimeConfig config;
  EXPECT_EQ(config.pageSize, 64);
  EXPECT_EQ(config.maxBatchSize, 1);
  EXPECT_EQ(config.maxSeqLen, 4096);
  EXPECT_EQ(config.device, "cpu");
  EXPECT_EQ(config.kvDtype, "bfloat16");
  EXPECT_EQ(config.maxNumBatchedTokens, 4096);
  EXPECT_DOUBLE_EQ(config.npuMemoryUtilization, 0.90);
  EXPECT_TRUE(config.supportsChunkedPrefillWithSpeculation);
  EXPECT_FALSE(config.requiresHomogeneousPrefillDecode);
}

TEST(KVCacheSpecTest, AcceptsAWellFormedSpec)
{
  const KVCacheSpec spec = makeSpec();
  EXPECT_NO_THROW(spec.validate());
  EXPECT_EQ(spec.tokenCapacity(), 128);
  EXPECT_EQ(spec.storageBlockSize(), 128);
}

TEST(KVCacheSpecTest, CompressionShrinksStorageButNotTokenCapacity)
{
  const KVCacheSpec spec = makeSpec(128, 4);
  ASSERT_NO_THROW(spec.validate());
  EXPECT_EQ(spec.tokenCapacity(), 128);
  EXPECT_EQ(spec.storageBlockSize(), 32);
}

TEST(KVCacheSpecTest, RejectsNonPositiveAndIndivisibleGeometry)
{
  EXPECT_THROW(makeSpec(0).validate(), std::invalid_argument);

  KVCacheSpec zeroPage   = makeSpec();
  zeroPage.pageSizeBytes = 0;
  EXPECT_THROW(zeroPage.validate(), std::invalid_argument);

  KVCacheSpec zeroRatio   = makeSpec();
  zeroRatio.compressRatio = 0;
  EXPECT_THROW(zeroRatio.validate(), std::invalid_argument);

  // 128 is not divisible by 5.
  EXPECT_THROW(makeSpec(128, 5).validate(), std::invalid_argument);
}

TEST(KVCacheGroupSpecTest, AcceptsAWellFormedGroup) { EXPECT_NO_THROW(makeGroup(makeSpec()).validate()); }

TEST(KVCacheGroupSpecTest, RejectsAnEmptyNameAndNonPositiveCounts)
{
  KVCacheGroupSpec unnamed = makeGroup(makeSpec());
  unnamed.name             = "";
  EXPECT_THROW(unnamed.validate(), std::invalid_argument);

  KVCacheGroupSpec noBlocks = makeGroup(makeSpec());
  noBlocks.maxBlocksPerSeq  = 0;
  EXPECT_THROW(noBlocks.validate(), std::invalid_argument);

  KVCacheGroupSpec zeroPool = makeGroup(makeSpec());
  zeroPool.numBlocks        = 0;
  EXPECT_THROW(zeroPool.validate(), std::invalid_argument);

  KVCacheGroupSpec noParts = makeGroup(makeSpec());
  noParts.numPartitions    = 0;
  EXPECT_THROW(noParts.validate(), std::invalid_argument);
}

TEST(KVCacheGroupSpecTest, SlidingWindowMustBeWholeBlocksAndFitTheRing)
{
  // A window of whole blocks that fits inside max_blocks_per_seq is accepted.
  KVCacheGroupSpec rolling = makeGroup(makeSpec(128), 8);
  rolling.slidingWindow    = 512; // 4 blocks of 128 tokens
  EXPECT_NO_THROW(rolling.validate());

  KVCacheGroupSpec nonPositive = makeGroup(makeSpec(128), 8);
  nonPositive.slidingWindow    = 0;
  EXPECT_THROW(nonPositive.validate(), std::invalid_argument);

  KVCacheGroupSpec ragged = makeGroup(makeSpec(128), 8);
  ragged.slidingWindow    = 200; // not a multiple of the 128-token block
  EXPECT_THROW(ragged.validate(), std::invalid_argument);

  KVCacheGroupSpec oversized = makeGroup(makeSpec(128), 2);
  oversized.slidingWindow    = 512; // 4 blocks, but the ring only holds 2
  EXPECT_THROW(oversized.validate(), std::invalid_argument);
}

TEST(KVCacheGroupSpecTest, PropagatesSpecValidationFailures) { EXPECT_THROW(makeGroup(makeSpec(0)).validate(), std::invalid_argument); }
