#include <gtest/gtest.h>

#include <serving/sched/scheduler.hpp>

using serving::memory::KvCacheManager;
using serving::sched::Request;
using serving::sched::RequestPtr;
using serving::sched::RequestStatus;
using serving::sched::ScheduledRequest;
using serving::sched::Scheduler;
using serving::sched::SchedulerConfig;
using serving::sched::SchedulerOutput;

namespace
{

/// The chunk sizes a model may advertise (config/types.py PREFILL_CHUNK_SIZE_CHOICES).
const std::vector<int> kChunkChoices{1024, 2048, 4096, 8192};

RequestPtr makeRequest(const std::string &id, std::vector<int> prompt, int maxNewTokens = 8)
{
  auto request            = std::make_shared<Request>();
  request->requestId      = id;
  request->promptTokenIds = std::move(prompt);
  request->maxNewTokens   = maxNewTokens;
  return request;
}

/// A RUNNING request that finished prefill and has one decoded token, so its
/// next step is a decode of exactly one token.
RequestPtr runningDecodeRequest(const std::string &id = "r", std::vector<int> prompt = {1, 2}, int firstOutput = 99)
{
  auto request               = makeRequest(id, prompt);
  request->numComputedTokens = static_cast<int>(prompt.size());
  request->outputTokenIds    = {firstOutput};
  request->status            = RequestStatus::Running;
  return request;
}

SchedulerConfig baseConfig()
{
  SchedulerConfig config;
  config.enablePrefixCache = false;
  return config;
}

} // namespace

// ---------------------------------------------------------------------------
// Config validation
// ---------------------------------------------------------------------------

TEST(SchedulerConfigTest, AcceptsEveryAdvertisedChunkSize)
{
  for (const int chunkSize : kChunkChoices)
  {
    SchedulerConfig config;
    config.longPrefillTokenThreshold = chunkSize;
    config.prefillChunkSizeChoices   = kChunkChoices;
    EXPECT_NO_THROW(config.validate()) << "chunk size " << chunkSize;
  }
}

TEST(SchedulerConfigTest, RejectsAChunkSizeTheModelDoesNotAdvertise)
{
  for (const int chunkSize : {128, 3072, 8193})
  {
    SchedulerConfig config;
    config.longPrefillTokenThreshold = chunkSize;
    config.prefillChunkSizeChoices   = kChunkChoices;
    EXPECT_THROW(config.validate(), std::invalid_argument) << "chunk size " << chunkSize;
  }
}

TEST(SchedulerConfigTest, SpeculativeDepthMustFitTheTokenBudget)
{
  SchedulerConfig config;
  config.maxNumScheduledTokens = 4;
  config.numSpeculativeTokens  = 4;
  EXPECT_THROW(config.validate(), std::invalid_argument);

  config.numSpeculativeTokens = -1;
  EXPECT_THROW(config.validate(), std::invalid_argument);
}

TEST(SchedulerConfigTest, RejectsANonPositivePerRequestPrefillLimit)
{
  SchedulerConfig config;
  config.maxPrefillTokensPerRequest = 0;
  EXPECT_THROW(config.validate(), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Admission
// ---------------------------------------------------------------------------

TEST(SchedulerAdmissionTest, RejectsAPromptLongerThanMaxSeqLen)
{
  KvCacheManager manager(8, 2, false);
  auto           config = baseConfig();
  config.maxSeqLen      = 4;
  Scheduler scheduler(config, manager);

  // Rejected rather than truncated: it could never be served.
  EXPECT_THROW(scheduler.addRequest(makeRequest("long", {1, 2, 3, 4, 5})), std::invalid_argument);
}

TEST(SchedulerAdmissionTest, RejectsAPromptThatLeavesNoRoomToGenerate)
{
  KvCacheManager manager(8, 2, false);
  auto           config = baseConfig();
  config.maxSeqLen      = 4;
  Scheduler scheduler(config, manager);

  EXPECT_THROW(scheduler.addRequest(makeRequest("exact", {1, 2, 3, 4})), std::invalid_argument);
}

TEST(SchedulerAdmissionTest, CapsGenerationToFitMaxSeqLen)
{
  KvCacheManager manager(64, 2, false);
  auto           config = baseConfig();
  config.maxSeqLen      = 10;
  Scheduler scheduler(config, manager);

  auto request = makeRequest("capped", {1, 2, 3}, 100);
  scheduler.addRequest(request);
  EXPECT_EQ(request->maxNewTokens, 7); // 10 - 3
}

TEST(SchedulerAdmissionTest, AdmittedRequestsQueueAndAreFindable)
{
  KvCacheManager manager(64, 2, false);
  Scheduler      scheduler(baseConfig(), manager);

  EXPECT_FALSE(scheduler.hasWork());
  auto request = makeRequest("a", {1, 2, 3});
  scheduler.addRequest(request);
  EXPECT_TRUE(scheduler.hasWork());
  EXPECT_EQ(scheduler.waiting().size(), 1u);
  EXPECT_EQ(scheduler.lookup("a"), request);
  EXPECT_EQ(scheduler.lookup("missing"), nullptr);
}

TEST(SchedulerAdmissionTest, RejectsAnUnchunkablePromptWhenChunkingIsOff)
{
  KvCacheManager manager(64, 2, false);
  auto           config        = baseConfig();
  config.enableChunkPrefill    = false;
  config.maxNumScheduledTokens = 4;
  config.maxSeqLen             = 64;
  Scheduler scheduler(config, manager);

  EXPECT_NO_THROW(scheduler.addRequest(makeRequest("fits", {1, 2, 3, 4})));
  EXPECT_THROW(scheduler.addRequest(makeRequest("toolong", {1, 2, 3, 4, 5})), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Scheduling
// ---------------------------------------------------------------------------

TEST(SchedulerScheduleTest, SchedulesAWaitingPromptAsPrefill)
{
  KvCacheManager manager(64, 2, false);
  Scheduler      scheduler(baseConfig(), manager);
  scheduler.addRequest(makeRequest("a", {1, 2, 3, 4}));

  const SchedulerOutput output = scheduler.schedule();
  ASSERT_EQ(output.scheduledRequests.size(), 1u);
  EXPECT_TRUE(output.scheduledRequests[0].isPrefill);
  EXPECT_EQ(output.scheduledRequests[0].numNewTokens, 4);
  EXPECT_EQ(output.numPrefillTokens, 4);
  EXPECT_EQ(output.numDecodeTokens, 0);
  EXPECT_FALSE(output.isEmpty());
  EXPECT_EQ(scheduler.running().size(), 1u);
  EXPECT_TRUE(scheduler.waiting().empty());

  // Two blocks of two tokens each.
  EXPECT_EQ(output.scheduledRequests[0].blockIds.size(), 2u);
}

TEST(SchedulerScheduleTest, ChunksALongPromptAcrossSteps)
{
  KvCacheManager manager(64, 2, false);
  auto           config            = baseConfig();
  config.longPrefillTokenThreshold = 4;
  config.prefillChunkSizeChoices   = {};
  Scheduler scheduler(config, manager);
  scheduler.addRequest(makeRequest("a", {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));

  const auto first = scheduler.schedule();
  ASSERT_EQ(first.scheduledRequests.size(), 1u);
  EXPECT_EQ(first.scheduledRequests[0].numNewTokens, 4);

  // The engine confirms the chunk, then the next step continues the prompt.
  auto outputs = scheduler.updateFromOutput(first, {});
  EXPECT_TRUE(outputs.empty()); // an incomplete prefill chunk samples nothing

  const auto second = scheduler.schedule();
  ASSERT_EQ(second.scheduledRequests.size(), 1u);
  EXPECT_TRUE(second.scheduledRequests[0].isPrefill);
  EXPECT_EQ(second.scheduledRequests[0].numNewTokens, 4);
  EXPECT_EQ(second.scheduledRequests[0].numComputedTokens, 4);
}

TEST(SchedulerScheduleTest, HonoursTheStepTokenBudget)
{
  KvCacheManager manager(64, 2, false);
  auto           config            = baseConfig();
  config.maxNumScheduledTokens     = 6;
  config.longPrefillTokenThreshold = 0; // no per-request chunk limit
  Scheduler scheduler(config, manager);
  scheduler.addRequest(makeRequest("a", {1, 2, 3, 4}));
  scheduler.addRequest(makeRequest("b", {5, 6, 7, 8}));

  const auto output = scheduler.schedule();
  // The budget is spent, not rationed per request: the first prompt takes its
  // four tokens and the second is CHUNKED into the remaining two rather than
  // being deferred whole. (Confirmed against the Python scheduler, which yields
  // exactly [("a", 4), ("b", 2)].)
  ASSERT_EQ(output.scheduledRequests.size(), 2u);
  EXPECT_EQ(output.scheduledRequests[0].request->requestId, "a");
  EXPECT_EQ(output.scheduledRequests[0].numNewTokens, 4);
  EXPECT_EQ(output.scheduledRequests[1].request->requestId, "b");
  EXPECT_EQ(output.scheduledRequests[1].numNewTokens, 2);
  EXPECT_EQ(output.numPrefillTokens, 6);
  EXPECT_EQ(output.numDecodeTokens, 0);
  EXPECT_TRUE(scheduler.waiting().empty());
}

TEST(SchedulerScheduleTest, HonoursTheRunningRequestCeiling)
{
  KvCacheManager manager(64, 2, false);
  auto           config    = baseConfig();
  config.maxNumRunningReqs = 2;
  Scheduler scheduler(config, manager);
  for (const char *id : {"a", "b", "c"}) { scheduler.addRequest(makeRequest(id, {1, 2})); }

  const auto output = scheduler.schedule();
  EXPECT_EQ(output.scheduledRequests.size(), 2u);
  EXPECT_EQ(scheduler.running().size(), 2u);
  EXPECT_EQ(scheduler.waiting().size(), 1u);
}

TEST(SchedulerScheduleTest, SchedulesDecodeForARunningRequest)
{
  KvCacheManager manager(64, 2, false);
  Scheduler      scheduler(baseConfig(), manager);
  scheduler.addRequest(makeRequest("a", {1, 2}));

  const auto prefill = scheduler.schedule();
  auto       outs    = scheduler.updateFromOutput(prefill, {{"a", {42}}});
  ASSERT_EQ(outs.size(), 1u);
  EXPECT_EQ(outs[0].newTokenId, 42);

  const auto decode = scheduler.schedule();
  ASSERT_EQ(decode.scheduledRequests.size(), 1u);
  EXPECT_FALSE(decode.scheduledRequests[0].isPrefill);
  EXPECT_EQ(decode.scheduledRequests[0].numNewTokens, 1);
  EXPECT_EQ(decode.numDecodeTokens, 1);
}

// ---------------------------------------------------------------------------
// Finishing
// ---------------------------------------------------------------------------

TEST(SchedulerFinishTest, StopsOnEos)
{
  KvCacheManager manager(64, 2, false);
  Scheduler      scheduler(baseConfig(), manager);
  auto           request = makeRequest("a", {1, 2});
  request->eosTokenId    = 7;
  scheduler.addRequest(request);

  const auto prefill = scheduler.schedule();
  const auto outputs = scheduler.updateFromOutput(prefill, {{"a", {7}}});
  ASSERT_EQ(outputs.size(), 1u);
  EXPECT_TRUE(outputs[0].finished);
  EXPECT_EQ(outputs[0].finishReason, "FINISHED_EOS");
  EXPECT_EQ(request->status, RequestStatus::FinishedEos);
  EXPECT_TRUE(scheduler.running().empty());
  // Its blocks went back to the pool.
  EXPECT_EQ(manager.numFreeBlocks(), manager.numBlocks());
}

TEST(SchedulerFinishTest, StopsOnLength)
{
  KvCacheManager manager(64, 2, false);
  Scheduler      scheduler(baseConfig(), manager);
  scheduler.addRequest(makeRequest("a", {1, 2}, 1));

  const auto prefill = scheduler.schedule();
  const auto outputs = scheduler.updateFromOutput(prefill, {{"a", {42}}});
  ASSERT_EQ(outputs.size(), 1u);
  EXPECT_TRUE(outputs[0].finished);
  EXPECT_EQ(outputs[0].finishReason, "FINISHED_LENGTH");
}

TEST(SchedulerFinishTest, SpeculativeOutputCountsOnlyTokensRetainedBeforeEos)
{
  // Ported from test_scheduler_speculative_output_counts_only_tokens_retained_before_eos.
  KvCacheManager manager(4, 2, false);
  Scheduler      scheduler(baseConfig(), manager);

  auto request               = makeRequest("speculative", {1}, 4);
  request->eosTokenId        = 7;
  request->numComputedTokens = 1;
  request->status            = RequestStatus::Running;
  scheduler.addRequest(request);
  // The Python test reaches into scheduler.running directly to stage a request
  // that is already mid-decode. Same here: addRequest can only queue it as
  // waiting, and going through a full prefill first would test something else.
  // Safe -- the underlying vector is not const, only the accessor is.
  const_cast<std::vector<RequestPtr> &>(scheduler.running()).push_back(request);

  SchedulerOutput  scheduled;
  ScheduledRequest step;
  step.request      = request;
  step.numNewTokens = 1;
  step.isPrefill    = false;
  scheduled.scheduledRequests.push_back(step);

  const auto outputs = scheduler.updateFromOutput(scheduled, {{"speculative", {7, 8}}});

  // The token after EOS is dropped and must not count as retained.
  EXPECT_EQ(request->outputTokenIds, std::vector<int>{7});
  EXPECT_EQ(request->numComputedTokens, 2);
  EXPECT_EQ(request->status, RequestStatus::FinishedEos);
  ASSERT_EQ(outputs.size(), 1u);
  EXPECT_EQ(outputs[0].newTokenId, 7);
  EXPECT_TRUE(outputs[0].finished);
}

TEST(SchedulerFinishTest, AbortReleasesEverything)
{
  KvCacheManager manager(64, 2, false);
  Scheduler      scheduler(baseConfig(), manager);
  scheduler.addRequest(makeRequest("a", {1, 2, 3, 4}));
  (void)scheduler.schedule();
  EXPECT_LT(manager.numFreeBlocks(), manager.numBlocks());

  scheduler.abortRequest("a");
  EXPECT_EQ(manager.numFreeBlocks(), manager.numBlocks());
  EXPECT_TRUE(scheduler.running().empty());
  EXPECT_TRUE(scheduler.waiting().empty());
  EXPECT_EQ(scheduler.lookup("a"), nullptr);
  EXPECT_FALSE(scheduler.hasWork());
}

// ---------------------------------------------------------------------------
// Preemption
// ---------------------------------------------------------------------------

TEST(SchedulerPreemptionTest, PreemptsTheLatestArrivalAndRequeuesItFirst)
{
  // Four blocks of one token: enough to admit two short requests, not to grow both.
  KvCacheManager manager(4, 1, false);
  auto           config            = baseConfig();
  config.maxNumScheduledTokens     = 8;
  config.longPrefillTokenThreshold = 0;
  Scheduler scheduler(config, manager);

  auto older           = makeRequest("older", {1, 2});
  auto younger         = makeRequest("younger", {3, 4});
  older->arrivalTime   = 1.0;
  younger->arrivalTime = 2.0;
  scheduler.addRequest(older);
  scheduler.addRequest(younger);

  const auto first = scheduler.schedule();
  ASSERT_EQ(first.scheduledRequests.size(), 2u);
  EXPECT_EQ(manager.numFreeBlocks(), 0);

  auto outs = scheduler.updateFromOutput(first, {{"older", {10}}, {"younger", {20}}});
  EXPECT_EQ(outs.size(), 2u);

  // Both now need another block and the pool is empty, so one must be preempted.
  const auto second = scheduler.schedule();
  ASSERT_EQ(second.preemptedRequests.size(), 1u);
  EXPECT_EQ(second.preemptedRequests[0]->requestId, "younger");
  EXPECT_EQ(younger->status, RequestStatus::Preempted);
  EXPECT_EQ(younger->numComputedTokens, 0);
  EXPECT_TRUE(younger->allocatedBlockIds.empty());

  // The victim goes to the FRONT of the waiting queue, and is not re-admitted in
  // the same step -- a stale in-flight result must drain first.
  ASSERT_FALSE(scheduler.waiting().empty());
  EXPECT_EQ(scheduler.waiting().front()->requestId, "younger");
  for (const auto &scheduledReq : second.scheduledRequests) { EXPECT_NE(scheduledReq.request->requestId, "younger"); }
}

// ---------------------------------------------------------------------------
// Prefix cache
// ---------------------------------------------------------------------------

TEST(SchedulerPrefixCacheTest, AHitSkipsTheCachedPrefix)
{
  KvCacheManager manager(64, 2, true);
  auto           config    = baseConfig();
  config.enablePrefixCache = true;
  Scheduler scheduler(config, manager);

  const std::vector<int> prompt{1, 2, 3, 4, 5, 6};
  auto                   first = makeRequest("first", prompt);
  scheduler.addRequest(first);
  const auto firstOut = scheduler.schedule();
  ASSERT_EQ(firstOut.scheduledRequests.size(), 1u);
  auto outs = scheduler.updateFromOutput(firstOut, {{"first", {42}}});
  ASSERT_EQ(outs.size(), 1u);

  // A second identical prompt must reuse the published blocks.
  auto second = makeRequest("second", prompt);
  scheduler.addRequest(second);
  const auto secondOut = scheduler.schedule();
  ASSERT_FALSE(secondOut.scheduledRequests.empty());

  const ScheduledRequest *step = nullptr;
  for (const auto &s : secondOut.scheduledRequests)
  {
    if (s.request->requestId == "second") { step = &s; }
  }
  ASSERT_NE(step, nullptr);
  EXPECT_GT(step->numComputedTokens, 0) << "expected a prefix-cache hit";
  EXPECT_LT(step->numNewTokens, static_cast<int>(prompt.size()));
}

TEST(SchedulerPrefixCacheTest, AFullHitStillPrefillsOneToken)
{
  KvCacheManager manager(64, 2, true);
  auto           config    = baseConfig();
  config.enablePrefixCache = true;
  Scheduler scheduler(config, manager);

  const std::vector<int> prompt{1, 2, 3, 4};
  auto                   first = makeRequest("first", prompt);
  scheduler.addRequest(first);
  const auto firstOut = scheduler.schedule();
  auto       outs     = scheduler.updateFromOutput(firstOut, {{"first", {42}}});
  ASSERT_EQ(outs.size(), 1u);

  auto second = makeRequest("second", prompt);
  scheduler.addRequest(second);
  const auto secondOut = scheduler.schedule();

  const ScheduledRequest *step = nullptr;
  for (const auto &s : secondOut.scheduledRequests)
  {
    if (s.request->requestId == "second") { step = &s; }
  }
  ASSERT_NE(step, nullptr);
  // One token is always left for prefill so the first generated token comes from
  // the same kernel as a cold run.
  EXPECT_GE(step->numNewTokens, 1);
  EXPECT_TRUE(step->isPrefill);
}

TEST(SchedulerPrefixCacheTest, DisablingThePrefixCacheComputesNoHashes)
{
  KvCacheManager manager(64, 2, false);
  Scheduler      scheduler(baseConfig(), manager);
  auto           request = makeRequest("a", {1, 2, 3, 4});
  scheduler.addRequest(request);
  EXPECT_TRUE(request->blockHashes.empty());
}

// ---------------------------------------------------------------------------
// Async scheduling
// ---------------------------------------------------------------------------

TEST(SchedulerAsyncTest, PlaceholdersKeepTheNextStepConsistent)
{
  KvCacheManager manager(64, 2, false);
  auto           config  = baseConfig();
  config.asyncScheduling = true;
  Scheduler scheduler(config, manager);
  scheduler.addRequest(makeRequest("a", {1, 2}));

  auto prefill = scheduler.schedule();
  scheduler.advanceAfterSchedule(prefill);

  const RequestPtr request = scheduler.lookup("a");
  ASSERT_NE(request, nullptr);
  // The prompt is covered and one token is reserved for what this step will sample.
  EXPECT_EQ(request->numComputedTokens, 2);
  EXPECT_EQ(request->numOutputPlaceholders, 1);
  // With the placeholder counted, nothing further is owed yet.
  EXPECT_EQ(request->numNewTokensNeeded(), 1);

  const auto outputs = scheduler.updateFromOutput(prefill, {{"a", {42}}});
  ASSERT_EQ(outputs.size(), 1u);
  EXPECT_EQ(request->numOutputPlaceholders, 0);
  EXPECT_EQ(request->outputTokenIds, std::vector<int>{42});
}

TEST(SchedulerAsyncTest, ReachesTheSameEndStateAsSynchronousScheduling)
{
  // The core invariant of async scheduling: optimistic advancement plus
  // reconciliation must land exactly where the synchronous path lands.
  const std::vector<int> prompt{1, 2, 3, 4};
  const std::vector<int> sampled{11, 12, 13};

  const auto run = [&](bool async) {
    KvCacheManager manager(64, 2, false);
    auto           config  = baseConfig();
    config.asyncScheduling = async;
    Scheduler scheduler(config, manager);
    scheduler.addRequest(makeRequest("a", prompt, 8));

    for (const int token : sampled)
    {
      auto output = scheduler.schedule();
      if (output.isEmpty()) { break; }
      if (async) { scheduler.advanceAfterSchedule(output); }
      (void)scheduler.updateFromOutput(output, {{"a", {token}}});
    }
    const RequestPtr request = scheduler.lookup("a");
    return std::tuple{request->numComputedTokens, request->outputTokenIds, request->numOutputPlaceholders, manager.numFreeBlocks()};
  };

  const auto sync  = run(false);
  const auto async = run(true);

  // The absolute values the Python scheduler reaches for this script.
  EXPECT_EQ(std::get<0>(sync), 6);
  EXPECT_EQ(std::get<1>(sync), (std::vector<int>{11, 12, 13}));
  EXPECT_EQ(std::get<2>(sync), 0);
  EXPECT_EQ(std::get<3>(sync), 61);

  EXPECT_EQ(std::get<0>(sync), std::get<0>(async)) << "num_computed_tokens diverged";
  EXPECT_EQ(std::get<1>(sync), std::get<1>(async)) << "output tokens diverged";
  EXPECT_EQ(std::get<2>(sync), std::get<2>(async)) << "placeholders left over";
  EXPECT_EQ(std::get<3>(sync), std::get<3>(async)) << "free block count diverged";
}

TEST(SchedulerAsyncTest, AStaleResultForAFinishedRequestIsDiscarded)
{
  KvCacheManager manager(64, 2, false);
  auto           config  = baseConfig();
  config.asyncScheduling = true;
  Scheduler scheduler(config, manager);

  auto request        = makeRequest("a", {1, 2}, 1);
  request->eosTokenId = 7;
  scheduler.addRequest(request);

  auto prefill = scheduler.schedule();
  scheduler.advanceAfterSchedule(prefill);
  const auto outputs = scheduler.updateFromOutput(prefill, {{"a", {7}}});
  ASSERT_EQ(outputs.size(), 1u);
  EXPECT_EQ(request->status, RequestStatus::FinishedEos);

  // Replaying the same step must change nothing: the request has left running
  // and its blocks are freed, so applying tokens would corrupt the bookkeeping.
  const auto before = request->outputTokenIds;
  const auto stale  = scheduler.updateFromOutput(prefill, {{"a", {99}}});
  EXPECT_TRUE(stale.empty());
  EXPECT_EQ(request->outputTokenIds, before);
}

// ---------------------------------------------------------------------------
// Homogeneous prefill/decode kernels
// ---------------------------------------------------------------------------

TEST(SchedulerPhaseTest, KeepsAStepSinglePhaseAndRotates)
{
  KvCacheManager manager(64, 2, false);
  auto           config                   = baseConfig();
  config.requiresHomogeneousPrefillDecode = true;
  Scheduler scheduler(config, manager);

  // One running decode plus one waiting prefill: the step must contain only one
  // kind of work, and the other kind must get its turn next.
  auto decoder = runningDecodeRequest("decoder");
  scheduler.addRequest(decoder);
  auto firstOut = scheduler.schedule();
  (void)scheduler.updateFromOutput(firstOut, {{"decoder", {5}}});

  scheduler.addRequest(makeRequest("fresh", {7, 8}));
  const auto second     = scheduler.schedule();
  bool       sawPrefill = false;
  bool       sawDecode  = false;
  for (const auto &s : second.scheduledRequests)
  {
    sawPrefill |= s.isPrefill;
    sawDecode |= !s.isPrefill;
  }
  EXPECT_FALSE(sawPrefill && sawDecode) << "a homogeneous step mixed prefill and decode";
}

TEST(SchedulerPhaseTest, MixedWorkIsUnrestrictedWithoutTheFlag)
{
  KvCacheManager manager(64, 2, false);
  Scheduler      scheduler(baseConfig(), manager);

  scheduler.addRequest(makeRequest("a", {1, 2}));
  auto first = scheduler.schedule();
  (void)scheduler.updateFromOutput(first, {{"a", {5}}});

  scheduler.addRequest(makeRequest("b", {7, 8}));
  const auto second = scheduler.schedule();
  // a decodes while b prefills, in the same step.
  EXPECT_EQ(second.scheduledRequests.size(), 2u);
  EXPECT_GT(second.numPrefillTokens, 0);
  EXPECT_GT(second.numDecodeTokens, 0);
}
