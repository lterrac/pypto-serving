#include <serving/sched/scheduler.hpp>

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace serving::sched
{

bool isFinished(RequestStatus status)
{
  switch (status)
  {
  case RequestStatus::FinishedEos:
  case RequestStatus::FinishedLength:
  case RequestStatus::FinishedStop:
  case RequestStatus::FinishedAborted: return true;
  default: return false;
  }
}

const char *statusName(RequestStatus status)
{
  switch (status)
  {
  case RequestStatus::Waiting: return "WAITING";
  case RequestStatus::Running: return "RUNNING";
  case RequestStatus::Preempted: return "PREEMPTED";
  case RequestStatus::FinishedEos: return "FINISHED_EOS";
  case RequestStatus::FinishedLength: return "FINISHED_LENGTH";
  case RequestStatus::FinishedStop: return "FINISHED_STOP";
  case RequestStatus::FinishedAborted: return "FINISHED_ABORTED";
  }
  return "UNKNOWN";
}

std::vector<int> Request::allTokenIds() const
{
  std::vector<int> all = promptTokenIds;
  all.insert(all.end(), outputTokenIds.begin(), outputTokenIds.end());
  return all;
}

void SchedulerConfig::validate() const
{
  if (numSpeculativeTokens < 0) { throw std::invalid_argument("num_speculative_tokens must be non-negative"); }
  if (numSpeculativeTokens + 1 > maxNumScheduledTokens) { throw std::invalid_argument("max_num_scheduled_tokens must fit one decode token plus num_speculative_tokens"); }
  if (maxPrefillTokensPerRequest.has_value() && *maxPrefillTokensPerRequest <= 0) { throw std::invalid_argument("max_prefill_tokens_per_request must be positive when specified"); }
  if (!prefillChunkSizeChoices.empty())
  {
    const bool valid = std::find(prefillChunkSizeChoices.begin(), prefillChunkSizeChoices.end(), longPrefillTokenThreshold) != prefillChunkSizeChoices.end();
    if (!valid)
    {
      std::ostringstream choices;
      for (size_t i = 0; i < prefillChunkSizeChoices.size(); ++i)
      {
        if (i != 0) { choices << ", "; }
        choices << prefillChunkSizeChoices[i];
      }
      throw std::invalid_argument("long_prefill_token_threshold must be one of (" + choices.str() + "), got " + std::to_string(longPrefillTokenThreshold));
    }
  }
}

Scheduler::Scheduler(SchedulerConfig config, memory::KvCacheManager &kvCacheManager)
  : _config(std::move(config)),
    _kvCacheManager(kvCacheManager)
{
  _config.validate();
  // The grouped speculative/MTP prefix-cache guard the Python constructor carries
  // cannot fire here: hasGroups() is always false in this build.
}

RequestPtr Scheduler::lookup(const std::string &requestId) const
{
  const auto it = _requests.find(requestId);
  return it == _requests.end() ? nullptr : it->second;
}

// ---------------------------------------------------------------------------
// Admission
// ---------------------------------------------------------------------------

void Scheduler::addRequest(const RequestPtr &request)
{
  const int promptLen = request->numPromptTokens();
  const int maxSeqLen = _config.maxSeqLen;
  // A prompt that fills or exceeds max_seq_len can never produce a token, so it
  // is rejected rather than truncated. The Python checks `>` and then `== 0`
  // remaining with two messages; one check covers both.
  if (promptLen >= maxSeqLen)
  {
    throw std::invalid_argument("Request " + request->requestId + " prompt length " + std::to_string(promptLen) + " leaves no room for generation within max_seq_len " +
                                std::to_string(maxSeqLen) + "; request rejected.");
  }
  const int remaining = maxSeqLen - promptLen;
  // Cap generation so prompt + generated never exceeds max_seq_len, keeping every
  // request inside its per-request KV budget and avoiding overflow preemption.
  if (request->maxNewTokens > remaining) { request->maxNewTokens = remaining; }

  if (!_config.enablePrefixCache)
  {
    const auto rejection = singlePrefillRejection(*request);
    if (rejection.has_value()) { throw std::invalid_argument(*rejection); }
  }
  if (_config.enablePrefixCache) { request->blockHashes = _kvCacheManager.computeBlockHashes(request->promptTokenIds); }

  request->status = RequestStatus::Waiting;
  _waiting.push_back(request);
  _requests[request->requestId] = request;
}

void Scheduler::abortRequest(const std::string &requestId)
{
  const auto it = _requests.find(requestId);
  if (it == _requests.end()) { return; }
  const RequestPtr request = it->second;
  request->status          = RequestStatus::FinishedAborted;
  freeRequestBlocks(*request);
  _running.erase(std::remove_if(_running.begin(), _running.end(), [&](const RequestPtr &r) { return r->requestId == requestId; }), _running.end());
  std::deque<RequestPtr> kept;
  for (const auto &r : _waiting)
  {
    if (r->requestId != requestId) { kept.push_back(r); }
  }
  _waiting = std::move(kept);
  _requests.erase(it);
}

void Scheduler::finishRequest(const std::string &requestId, RequestStatus status)
{
  const auto it = _requests.find(requestId);
  if (it == _requests.end()) { return; }
  it->second->status = status;
  freeRequestBlocks(*it->second);
  _running.erase(std::remove_if(_running.begin(), _running.end(), [&](const RequestPtr &r) { return r->requestId == requestId; }), _running.end());
}

// ---------------------------------------------------------------------------
// Chunking limits
// ---------------------------------------------------------------------------

std::optional<int> Scheduler::prefillChunkLimit() const
{
  std::vector<int> limits;
  if (_config.maxPrefillTokensPerRequest.has_value()) { limits.push_back(*_config.maxPrefillTokensPerRequest); }
  if (_config.enableChunkPrefill && _config.longPrefillTokenThreshold > 0) { limits.push_back(_config.longPrefillTokenThreshold); }
  if (limits.empty()) { return std::nullopt; }
  return *std::min_element(limits.begin(), limits.end());
}

int Scheduler::singlePrefillDispatchLimit() const
{
  const auto limit = prefillChunkLimit();
  if (!limit.has_value()) { return _config.maxNumScheduledTokens; }
  return std::min(*limit, _config.maxNumScheduledTokens);
}

bool Scheduler::requiresSinglePrefillDispatch() const
{
  return !_config.enableChunkPrefill || (_config.numSpeculativeTokens > 0 && !_config.supportsChunkedPrefillWithSpeculation);
}

std::optional<std::string> Scheduler::singlePrefillRejection(const Request &request) const
{
  if (!requiresSinglePrefillDispatch()) { return std::nullopt; }
  const int uncachedTokens = std::max(1, request.numPromptTokens() - request.numComputedTokens);
  const int limit          = singlePrefillDispatchLimit();
  if (uncachedTokens <= limit) { return std::nullopt; }
  if (!_config.enableChunkPrefill)
  {
    return "Request " + request.requestId + " uncached prompt length " + std::to_string(uncachedTokens) + " exceeds the effective single-dispatch prefill limit " +
           std::to_string(limit) + " while chunked prefill is disabled.";
  }
  return "Request " + request.requestId + " uncached prompt length " + std::to_string(uncachedTokens) +
         " requires chunked prefill, which is not supported with speculative decoding for this model; "
         "the single-dispatch limit is " +
         std::to_string(limit) + ". Disable speculative decoding or shorten the uncached prompt suffix.";
}

int Scheduler::limitScheduledTokens(const Request &request, int tokenBudget) const
{
  const int needed = request.numNewTokensNeeded();
  int       limit  = tokenBudget;
  if (request.isPrefill())
  {
    const auto chunkLimit = prefillChunkLimit();
    if (chunkLimit.has_value()) { limit = std::min(limit, *chunkLimit); }
    if (requiresSinglePrefillDispatch() && needed > limit) { return 0; }
  }
  return std::min(needed, limit);
}

std::optional<std::string> Scheduler::groupedCachePhase()
{
  if (!_config.requiresHomogeneousPrefillDecode) { return std::nullopt; }

  const auto eligible               = [](const RequestPtr &r) { return r->status != RequestStatus::Preempted && !r->terminalPrefillInFlight && r->numNewTokensNeeded() > 0; };
  const bool hasRunningPrefill      = std::any_of(_running.begin(), _running.end(), [&](const RequestPtr &r) { return eligible(r) && r->isPrefill(); });
  const bool canAdmitWaitingPrefill = !_waiting.empty() && static_cast<int>(_running.size()) < _config.maxNumRunningReqs;
  const bool hasPrefill             = hasRunningPrefill || canAdmitWaitingPrefill;
  const bool hasDecode              = std::any_of(_running.begin(), _running.end(), [&](const RequestPtr &r) { return eligible(r) && !r->isPrefill(); });

  if (!hasPrefill && !hasDecode) { return std::nullopt; }

  std::string phase;
  if (hasPrefill && hasDecode) { phase = _nextGroupedCachePhase; }
  else if (hasPrefill) { phase = "prefill"; }
  else { phase = "decode"; }

  // Rotate on selection rather than completion: if the selected phase cannot
  // allocate this pass, the other still gets its turn next call.
  _nextGroupedCachePhase = (phase == "prefill") ? "decode" : "prefill";
  return phase;
}

// ---------------------------------------------------------------------------
// schedule
// ---------------------------------------------------------------------------

SchedulerOutput Scheduler::schedule()
{
  SchedulerOutput output;
  int             tokenBudget  = _config.maxNumScheduledTokens;
  const auto      groupedPhase = groupedCachePhase();

  // Phase 1: RUNNING requests (decode, or a resumed prefill chunk).
  std::vector<std::string>   scheduledReqIds;
  std::map<std::string, int> numScheduledTokens;
  std::vector<RequestPtr>    runningToKeep;

  // Python iterates the original list object while preemption rebinds self.running,
  // so the loop walks a snapshot. Copy to reproduce that exactly.
  const std::vector<RequestPtr> runningSnapshot = _running;

  for (const RequestPtr &request : runningSnapshot)
  {
    // A request later in this snapshot may have been preempted while scheduling
    // an earlier one. Do not schedule it again from the stale snapshot.
    if (request->status == RequestStatus::Preempted) { continue; }
    if (request->terminalPrefillInFlight)
    {
      runningToKeep.push_back(request);
      continue;
    }
    if (groupedPhase.has_value() && request->isPrefill() != (*groupedPhase == "prefill"))
    {
      runningToKeep.push_back(request);
      continue;
    }
    if (request->numNewTokensNeeded() <= 0)
    {
      runningToKeep.push_back(request);
      continue;
    }

    const int numNew = limitScheduledTokens(*request, tokenBudget);
    if (numNew <= 0)
    {
      runningToKeep.push_back(request);
      continue;
    }

    const bool isPrefill         = request->isPrefill();
    const int  speculativeTokens = (!isPrefill && request->temperature <= 0.0) ? _config.numSpeculativeTokens : 0;
    const int  scheduledTokens   = numNew + speculativeTokens;
    if (scheduledTokens > tokenBudget)
    {
      runningToKeep.push_back(request);
      continue;
    }

    if (!tryAllocateRequestBlocks(*request, scheduledTokens))
    {
      auto preempted = preemptLowestPriority(*request, scheduledReqIds, numScheduledTokens, output);
      if (!preempted.has_value())
      {
        runningToKeep.push_back(request);
        continue;
      }
      tokenBudget += preempted->returnedTokens;
      output.preemptedRequests.push_back(preempted->request);
      if (!tryAllocateRequestBlocks(*request, scheduledTokens))
      {
        runningToKeep.push_back(request);
        continue;
      }
    }

    std::vector<int> allBlockIds = request->cachedBlockIds;
    allBlockIds.insert(allBlockIds.end(), request->allocatedBlockIds.begin(), request->allocatedBlockIds.end());

    ScheduledRequest scheduled;
    scheduled.request           = request;
    scheduled.numNewTokens      = numNew;
    scheduled.isPrefill         = isPrefill;
    scheduled.numComputedTokens = request->numComputedTokens;
    scheduled.blockIds          = allBlockIds;
    output.scheduledRequests.push_back(std::move(scheduled));

    scheduledReqIds.push_back(request->requestId);
    numScheduledTokens[request->requestId] = scheduledTokens;
    if (isPrefill) { output.numPrefillTokens += numNew; }
    else { output.numDecodeTokens += scheduledTokens; }
    tokenBudget -= scheduledTokens;
    runningToKeep.push_back(request);
  }

  // Victims seen earlier in the iteration may already be in runningToKeep. They
  // now live in the waiting queue and must not be retained in both.
  _running.clear();
  for (const RequestPtr &request : runningToKeep)
  {
    if (request->status != RequestStatus::Preempted) { _running.push_back(request); }
  }

  // Keep a single-phase kernel's commands homogeneous: prefill gets the next step.
  if (groupedPhase.has_value() && *groupedPhase == "decode") { return output; }

  // Phase 2: WAITING requests (new prefill).
  std::vector<std::string> preemptedThisStep;
  for (const RequestPtr &request : output.preemptedRequests) { preemptedThisStep.push_back(request->requestId); }

  std::deque<RequestPtr> remainingWaiting;
  while (!_waiting.empty() && tokenBudget > 0)
  {
    if (static_cast<int>(_running.size()) >= _config.maxNumRunningReqs) { break; }

    const RequestPtr request = _waiting.front();
    _waiting.pop_front();

    // Keep a victim PREEMPTED until any older in-flight result has drained.
    // Re-admitting it in this same schedule() call would make that stale result
    // indistinguishable from output for the restarted request.
    if (std::find(preemptedThisStep.begin(), preemptedThisStep.end(), request->requestId) != preemptedThisStep.end())
    {
      remainingWaiting.push_back(request);
      continue;
    }

    if (_config.enablePrefixCache)
    {
      auto cachedBlocks = _kvCacheManager.getComputedBlocks(request->promptTokenIds);
      if (!cachedBlocks.empty())
      {
        request->cachedBlockIds.clear();
        for (const memory::KVCacheBlock *block : cachedBlocks) { request->cachedBlockIds.push_back(block->blockId); }
        request->numComputedTokens = static_cast<int>(cachedBlocks.size()) * _kvCacheManager.blockSize();
        request->numBlocksCached   = static_cast<int>(cachedBlocks.size());
      }
    }

    const auto rejection = singlePrefillRejection(*request);
    if (rejection.has_value())
    {
      releaseWaitingPrefixBlocks(request);
      request->status = RequestStatus::FinishedAborted;
      _requests.erase(request->requestId);
      output.rejectedRequests[request->requestId] = *rejection;
      continue;
    }

    int numNew = limitScheduledTokens(*request, tokenBudget);
    if (numNew <= 0)
    {
      // Full prefix-cache hit: leave one token for prefill so the output uses the
      // SAME kernel as a cold run and produces an identical first token.
      if (request->numComputedTokens >= request->numPromptTokens())
      {
        request->numComputedTokens = std::max(0, request->numPromptTokens() - 1);
        numNew                     = 1;
      }
      else
      {
        releaseWaitingPrefixBlocks(request);
        remainingWaiting.push_back(request);
        continue;
      }
    }

    if (!tryAllocateRequestBlocks(*request, numNew))
    {
      releaseWaitingPrefixBlocks(request);
      remainingWaiting.push_back(request);
      break;
    }

    request->status = RequestStatus::Running;
    _running.push_back(request);

    std::vector<int> allBlockIds = request->cachedBlockIds;
    allBlockIds.insert(allBlockIds.end(), request->allocatedBlockIds.begin(), request->allocatedBlockIds.end());

    ScheduledRequest scheduled;
    scheduled.request           = request;
    scheduled.numNewTokens      = numNew;
    scheduled.isPrefill         = true;
    scheduled.numComputedTokens = request->numComputedTokens;
    scheduled.blockIds          = allBlockIds;
    output.scheduledRequests.push_back(std::move(scheduled));

    output.numPrefillTokens += numNew;
    tokenBudget -= numNew;
  }

  for (const auto &request : _waiting) { remainingWaiting.push_back(request); }
  _waiting = std::move(remainingWaiting);

  return output;
}

void Scheduler::releaseWaitingPrefixBlocks(const RequestPtr &request)
{
  freeRequestBlocks(*request);
  request->numComputedTokens = 0;
}

// ---------------------------------------------------------------------------
// Async advance / reconcile
// ---------------------------------------------------------------------------

void Scheduler::advanceAfterSchedule(SchedulerOutput &output)
{
  if (!_config.asyncScheduling) { return; }
  for (ScheduledRequest &scheduled : output.scheduledRequests)
  {
    Request   &request         = *scheduled.request;
    const bool completesPrompt = request.numComputedTokens + scheduled.numNewTokens >= request.numPromptTokens();
    request.numComputedTokens += scheduled.numNewTokens;

    // Prefix-cache blocks are not published here: this runs at schedule time,
    // before the tokens exist.
    if (!scheduled.isPrefill || completesPrompt)
    {
      // Reserve the MAXIMUM this step can emit. Speculative/MTP decode returns
      // 1..1+num_speculative_tokens, known only once the worker replies, so
      // reserve the upper bound (matching the blocks schedule() allocated) and
      // subtract the shortfall in reconcileAsyncOutput.
      const int reserved = 1 + speculativeTokensFor(request, scheduled);
      request.numOutputPlaceholders += reserved;
      request.numComputedTokens += reserved - 1;
    }
    if (scheduled.isPrefill && completesPrompt && _config.numSpeculativeTokens > 0) { request.terminalPrefillInFlight = true; }
  }
}

int Scheduler::speculativeTokensFor(const Request &request, const ScheduledRequest &scheduled) const
{
  // Mirrors the accounting schedule() uses when allocating: only greedy decode
  // steps get speculative capacity.
  if (scheduled.isPrefill || request.temperature > 0.0) { return 0; }
  return _config.numSpeculativeTokens;
}

std::vector<RequestOutput> Scheduler::updateFromOutput(const SchedulerOutput &output, const std::unordered_map<std::string, std::vector<int>> &newTokenIds)
{
  std::vector<RequestOutput> outputs;

  for (const ScheduledRequest &scheduled : output.scheduledRequests)
  {
    const RequestPtr request = scheduled.request;

    // Async pipelining: a request that finished, was aborted, or was preempted at
    // step N may still have step N+1 in flight. Discard that stale result -- the
    // request has left `running` (blocks freed) or had its state reset, so
    // applying tokens would corrupt the bookkeeping.
    if (isFinished(request->status) || request->status == RequestStatus::Preempted) { continue; }

    if (scheduled.isPrefill && scheduled.numComputedTokens + scheduled.numNewTokens >= request->numPromptTokens()) { request->terminalPrefillInFlight = false; }

    std::vector<int> tokenIds;
    const auto       it = newTokenIds.find(request->requestId);
    if (it != newTokenIds.end()) { tokenIds = it->second; }

    if (_config.asyncScheduling)
    {
      // advanceAfterSchedule already advanced optimistically; this applies the real
      // tokens and releases the placeholders.
      reconcileAsyncOutput(request, scheduled, tokenIds, outputs);
    }
    else if (scheduled.isPrefill)
    {
      request->numComputedTokens += scheduled.numNewTokens;
      cacheCompletedBlocks(*request);
      if (request->numComputedTokens < request->numPromptTokens()) { continue; }
      for (const int tokenId : tokenIds)
      {
        request->outputTokenIds.push_back(tokenId);
        outputs.push_back(RequestOutput{request->requestId, tokenId, false, ""});
        if (checkFinish(*request).has_value()) { break; }
      }
    }
    else
    {
      int retainedTokens = 0;
      for (const int tokenId : tokenIds)
      {
        request->outputTokenIds.push_back(tokenId);
        retainedTokens += 1;
        outputs.push_back(RequestOutput{request->requestId, tokenId, false, ""});
        if (checkFinish(*request).has_value()) { break; }
      }
      request->numComputedTokens += retainedTokens;
      cacheCompletedBlocks(*request);
    }
  }

  std::vector<std::string> finishedIds;
  for (const RequestPtr &request : _running)
  {
    if (isFinished(request->status)) { continue; }
    const auto finishReason = checkFinish(*request);
    if (!finishReason.has_value()) { continue; }
    request->status = *finishReason;
    finishedIds.push_back(request->requestId);

    // Mark the request's most recent output finished; emit a bare one if it
    // produced nothing this step.
    bool marked = false;
    for (auto out = outputs.rbegin(); out != outputs.rend(); ++out)
    {
      if (out->requestId == request->requestId)
      {
        out->finished     = true;
        out->finishReason = statusName(*finishReason);
        marked            = true;
        break;
      }
    }
    if (!marked) { outputs.push_back(RequestOutput{request->requestId, std::nullopt, true, statusName(*finishReason)}); }
  }

  for (const std::string &requestId : finishedIds)
  {
    const auto it = _requests.find(requestId);
    if (it != _requests.end()) { freeRequestBlocks(*it->second); }
    _running.erase(std::remove_if(_running.begin(), _running.end(), [&](const RequestPtr &r) { return r->requestId == requestId; }), _running.end());
  }

  return outputs;
}

void Scheduler::reconcileAsyncOutput(const RequestPtr &requestPtr, const ScheduledRequest &scheduled, const std::vector<int> &tokenIds, std::vector<RequestOutput> &outputs)
{
  Request &request = *requestPtr;

  // This step reserved placeholders iff it sampled: a decode step, or a prefill
  // chunk that completed the prompt. numComputedTokens was already advanced, so
  // "completed the prompt" means numComputed >= prompt.
  const bool sampledThisStep = !scheduled.isPrefill || request.numComputedTokens >= request.numPromptTokens();
  const int  reserved        = sampledThisStep ? 1 + speculativeTokensFor(request, scheduled) : 0;

  // Publish through the token count this step actually confirmed. A newer
  // in-flight chunk may already have advanced the request's rolling state, so
  // reading current state here would publish too much.
  const int confirmedTokens = scheduled.numComputedTokens + scheduled.numNewTokens;
  cacheCompletedBlocks(request, confirmedTokens);

  int retainedTokens = 0;
  for (const int tokenId : tokenIds)
  {
    request.outputTokenIds.push_back(tokenId);
    retainedTokens += 1;
    outputs.push_back(RequestOutput{request.requestId, tokenId, false, ""});
    // Tokens after a finish are dropped, mirroring the sync path, so they must
    // not count as retained.
    if (checkFinish(request).has_value()) { break; }
  }

  if (reserved != 0)
  {
    request.numOutputPlaceholders = std::max(0, request.numOutputPlaceholders - reserved);
    // Reclaim only the SPECULATIVE positions that produced no retained token.
    // advanceAfterSchedule added `reserved - 1` on top of numNewTokens; the latter
    // is this step's real KV work and must never be reverted, or the same prefill
    // chunk would be re-scheduled and decoded twice.
    const int speculativePositions = reserved - 1;
    const int unusedSpeculative    = std::max(0, speculativePositions - std::max(0, retainedTokens - 1));
    if (unusedSpeculative > 0) { request.numComputedTokens = std::max(0, request.numComputedTokens - unusedSpeculative); }
  }
}

std::optional<RequestStatus> Scheduler::checkFinish(const Request &request) const
{
  if (request.outputTokenIds.empty()) { return std::nullopt; }
  const int lastToken = request.outputTokenIds.back();
  if (request.eosTokenId.has_value() && lastToken == *request.eosTokenId) { return RequestStatus::FinishedEos; }
  if (static_cast<int>(request.outputTokenIds.size()) >= request.maxNewTokens) { return RequestStatus::FinishedLength; }
  return std::nullopt;
}

// ---------------------------------------------------------------------------
// Blocks
// ---------------------------------------------------------------------------

int Scheduler::blocksNeeded(const Request &request, int numNewTokens) const
{
  const int currentTotalTokens = request.numComputedTokens + numNewTokens;
  const int currentBlocks      = static_cast<int>(request.cachedBlockIds.size() + request.allocatedBlockIds.size());
  const int blockSize          = _kvCacheManager.blockSize();
  const int neededBlocks       = (currentTotalTokens + blockSize - 1) / blockSize;
  return std::max(0, neededBlocks - currentBlocks);
}

bool Scheduler::tryAllocateBlocks(Request &request, int numBlocks)
{
  if (numBlocks <= 0) { return true; }
  if (_kvCacheManager.numFreeBlocks() < numBlocks) { return false; }
  auto blockIds = _kvCacheManager.allocateBlockIds(numBlocks);
  if (!blockIds.has_value()) { return false; }
  request.allocatedBlockIds.insert(request.allocatedBlockIds.end(), blockIds->begin(), blockIds->end());
  return true;
}

bool Scheduler::tryAllocateRequestBlocks(Request &request, int numNewTokens) { return tryAllocateBlocks(request, blocksNeeded(request, numNewTokens)); }

std::optional<Scheduler::Preemption> Scheduler::preemptLowestPriority(const Request              &exclude,
                                                                      std::vector<std::string>   &scheduledReqIds,
                                                                      std::map<std::string, int> &numScheduledTokens,
                                                                      SchedulerOutput            &output)
{
  if (_running.empty()) { return std::nullopt; }

  std::vector<RequestPtr> candidates;
  for (const RequestPtr &r : _running)
  {
    if (r->requestId != exclude.requestId) { candidates.push_back(r); }
  }
  if (candidates.empty()) { return std::nullopt; }

  // Lowest priority is the most recently arrived.
  const RequestPtr victim = *std::max_element(candidates.begin(), candidates.end(), [](const RequestPtr &a, const RequestPtr &b) { return a->arrivalTime < b->arrivalTime; });

  int        returnedTokens = 0;
  const auto scheduledIt    = std::find(scheduledReqIds.begin(), scheduledReqIds.end(), victim->requestId);
  if (scheduledIt != scheduledReqIds.end())
  {
    scheduledReqIds.erase(scheduledIt);
    const auto tokensIt = numScheduledTokens.find(victim->requestId);
    if (tokensIt != numScheduledTokens.end())
    {
      returnedTokens = tokensIt->second;
      numScheduledTokens.erase(tokensIt);
    }
    output.scheduledRequests.erase(
      std::remove_if(output.scheduledRequests.begin(), output.scheduledRequests.end(), [&](const ScheduledRequest &s) { return s.request->requestId == victim->requestId; }),
      output.scheduledRequests.end());
    if (victim->isPrefill()) { output.numPrefillTokens -= returnedTokens; }
    else { output.numDecodeTokens -= returnedTokens; }
  }

  freeRequestBlocks(*victim);
  victim->status            = RequestStatus::Preempted;
  victim->numComputedTokens = 0;
  victim->cachedBlockIds.clear();
  victim->allocatedBlockIds.clear();
  victim->numBlocksCached = 0;
  // Drop any optimistic placeholder so the re-queued request restarts from a
  // clean prefill state; its in-flight step, if any, is discarded engine-side.
  victim->numOutputPlaceholders   = 0;
  victim->terminalPrefillInFlight = false;

  _running.erase(std::remove_if(_running.begin(), _running.end(), [&](const RequestPtr &r) { return r->requestId == victim->requestId; }), _running.end());
  _waiting.push_front(victim);

  return Preemption{victim, returnedTokens};
}

void Scheduler::freeRequestBlocks(Request &request)
{
  _kvCacheManager.releaseBlocksByIds(request.cachedBlockIds);
  _kvCacheManager.releaseBlocksByIds(request.allocatedBlockIds);
  request.cachedBlockIds.clear();
  request.allocatedBlockIds.clear();
}

void Scheduler::cacheCompletedBlocks(Request &request, std::optional<int> numComputedTokens)
{
  if (!_config.enablePrefixCache) { return; }
  const int confirmedTokens = numComputedTokens.value_or(request.numComputedTokens);

  const int totalBlocksComputed = std::min(confirmedTokens / _kvCacheManager.blockSize(), static_cast<int>(request.blockHashes.size()));
  const int alreadyCached       = request.numBlocksCached;
  if (totalBlocksComputed <= alreadyCached) { return; }

  std::vector<int> allBlockIds = request.cachedBlockIds;
  allBlockIds.insert(allBlockIds.end(), request.allocatedBlockIds.begin(), request.allocatedBlockIds.end());
  _kvCacheManager.cacheBlockIds(allBlockIds, request.blockHashes, alreadyCached, totalBlocksComputed);
  request.numBlocksCached = totalBlocksComputed;
}

} // namespace serving::sched
