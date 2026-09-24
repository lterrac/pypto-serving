#pragma once

/**
 * Continuous-batching scheduler with chunked prefill, prefix caching and
 * preemption (`pypto_serving/serving/sched/scheduler.py`).
 *
 * Grouped-cache branches are not implemented; hasGroups() is false. Requests
 * are shared_ptr: the waiting queue, the running list, the id map and the
 * scheduled output alias them, and preemption moves them between those.
 */

#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <serving/memory/kv_cache.hpp>

namespace serving::sched
{

enum class RequestStatus
{
  Waiting,
  Running,
  Preempted,
  FinishedEos,
  FinishedLength,
  FinishedStop,
  FinishedAborted,
};

/// Whether a status is terminal.
[[nodiscard]] bool isFinished(RequestStatus status);

/// The status name, as the engine reports it as a finish reason.
[[nodiscard]] const char *statusName(RequestStatus status);

struct SchedulerConfig
{
  int maxNumRunningReqs         = 32;
  int maxNumScheduledTokens     = 4096;
  int longPrefillTokenThreshold = 2048;

  std::optional<int> maxPrefillTokensPerRequest = std::nullopt;
  std::vector<int>   prefillChunkSizeChoices    = {};
  int                maxSeqLen                  = 4096;

  bool enablePrefixCache                     = true;
  bool enableChunkPrefill                    = true;
  int  numSpeculativeTokens                  = 0;
  bool supportsChunkedPrefillWithSpeculation = true;
  bool requiresHomogeneousPrefillDecode      = false;

  /// Schedule step N+1 before step N's sampled token returns, advancing request
  /// state optimistically through placeholders.
  bool asyncScheduling = false;

  /// Mirrors the Python `__post_init__`; throws std::invalid_argument.
  void validate() const;
};

struct Request
{
  std::string      requestId;
  std::vector<int> promptTokenIds;
  int              maxNewTokens = 0;
  double           arrivalTime  = 0.0;

  RequestStatus    status            = RequestStatus::Waiting;
  int              numComputedTokens = 0;
  std::vector<int> outputTokenIds;

  std::vector<std::string> stopStrings;
  std::optional<int>       eosTokenId = std::nullopt;

  double                  temperature = 0.0;
  double                  topP        = 1.0;
  std::optional<int>      topK        = std::nullopt;
  std::optional<uint64_t> seed        = std::nullopt;

  std::vector<int>               cachedBlockIds;
  std::vector<int>               allocatedBlockIds;
  std::vector<memory::BlockHash> blockHashes;

  /// How many blocks have been published to the prefix cache.
  int numBlocksCached = 0;

  /// Async scheduling: tokens scheduled optimistically but not yet sampled.
  int numOutputPlaceholders = 0;

  /// Async scheduling must not turn a terminal prefill into a decode wave until
  /// the worker has confirmed it. Per request, so unrelated requests keep the
  /// depth-2 pipeline.
  bool terminalPrefillInFlight = false;

  [[nodiscard]] int numPromptTokens() const { return static_cast<int>(promptTokenIds.size()); }

  /// Placeholders count as not-yet-materialised output tokens, so this stays
  /// consistent when the next step is scheduled before the in-flight token lands.
  [[nodiscard]] int numTokens() const { return numPromptTokens() + static_cast<int>(outputTokenIds.size()) + numOutputPlaceholders; }

  [[nodiscard]] int              numNewTokensNeeded() const { return numTokens() - numComputedTokens; }
  [[nodiscard]] bool             isPrefill() const { return numComputedTokens < numPromptTokens(); }
  [[nodiscard]] std::vector<int> allTokenIds() const;
};

using RequestPtr = std::shared_ptr<Request>;

struct ScheduledRequest
{
  RequestPtr       request;
  int              numNewTokens      = 0;
  bool             isPrefill         = false;
  int              numComputedTokens = 0;
  std::vector<int> blockIds;
  bool             resumedFromPreemption = false;
};

struct SchedulerOutput
{
  std::vector<ScheduledRequest>      scheduledRequests;
  std::vector<RequestPtr>            preemptedRequests;
  std::map<std::string, std::string> rejectedRequests;
  int                                numPrefillTokens = 0;
  int                                numDecodeTokens  = 0;

  [[nodiscard]] bool isEmpty() const { return scheduledRequests.empty(); }
};

struct RequestOutput
{
  std::string        requestId;
  std::optional<int> newTokenId = std::nullopt;
  bool               finished   = false;
  std::string        finishReason;
};

class Scheduler
{
  public:

  Scheduler(SchedulerConfig config, memory::KvCacheManager &kvCacheManager);

  /// Admit a request. Throws std::invalid_argument on a prompt that can never be served.
  void addRequest(const RequestPtr &request);

  void abortRequest(const std::string &requestId);
  void finishRequest(const std::string &requestId, RequestStatus status);

  [[nodiscard]] bool hasWork() const { return !_running.empty() || !_waiting.empty(); }

  /// Select the next step's work.
  [[nodiscard]] SchedulerOutput schedule();

  /// Optimistically advance state for a just-scheduled step (async mode only).
  void advanceAfterSchedule(SchedulerOutput &output);

  /// Apply worker results. `newTokenIds` maps request id to the tokens it sampled.
  [[nodiscard]] std::vector<RequestOutput> updateFromOutput(const SchedulerOutput &output, const std::unordered_map<std::string, std::vector<int>> &newTokenIds);

  [[nodiscard]] const SchedulerConfig         &config() const { return _config; }
  [[nodiscard]] memory::KvCacheManager        &kvCacheManager() { return _kvCacheManager; }
  [[nodiscard]] const std::vector<RequestPtr> &running() const { return _running; }
  [[nodiscard]] const std::deque<RequestPtr>  &waiting() const { return _waiting; }
  [[nodiscard]] RequestPtr                     lookup(const std::string &requestId) const;

  private:

  void                                     releaseWaitingPrefixBlocks(const RequestPtr &request);
  [[nodiscard]] std::optional<int>         prefillChunkLimit() const;
  [[nodiscard]] int                        singlePrefillDispatchLimit() const;
  [[nodiscard]] bool                       requiresSinglePrefillDispatch() const;
  [[nodiscard]] std::optional<std::string> singlePrefillRejection(const Request &request) const;
  [[nodiscard]] int                        limitScheduledTokens(const Request &request, int tokenBudget) const;

  /// "prefill", "decode" or nullopt when the kernel imposes no single-phase contract.
  [[nodiscard]] std::optional<std::string> groupedCachePhase();

  [[nodiscard]] int speculativeTokensFor(const Request &request, const ScheduledRequest &scheduled) const;

  void reconcileAsyncOutput(const RequestPtr &request, const ScheduledRequest &scheduled, const std::vector<int> &tokenIds, std::vector<RequestOutput> &outputs);

  [[nodiscard]] std::optional<RequestStatus> checkFinish(const Request &request) const;

  [[nodiscard]] int  blocksNeeded(const Request &request, int numNewTokens) const;
  [[nodiscard]] bool tryAllocateBlocks(Request &request, int numBlocks);
  [[nodiscard]] bool tryAllocateRequestBlocks(Request &request, int numNewTokens);

  struct Preemption
  {
    RequestPtr request;
    int        returnedTokens = 0;
  };

  [[nodiscard]] std::optional<Preemption> preemptLowestPriority(const Request              &exclude,
                                                                std::vector<std::string>   &scheduledReqIds,
                                                                std::map<std::string, int> &numScheduledTokens,
                                                                SchedulerOutput            &output);

  void freeRequestBlocks(Request &request);
  void cacheCompletedBlocks(Request &request, std::optional<int> numComputedTokens = std::nullopt);

  SchedulerConfig         _config;
  memory::KvCacheManager &_kvCacheManager;

  std::deque<RequestPtr>                      _waiting;
  std::vector<RequestPtr>                     _running;
  std::unordered_map<std::string, RequestPtr> _requests;

  /// Decode goes first when both kinds of work initially coexist.
  std::string _nextGroupedCachePhase = "decode";
};

} // namespace serving::sched
