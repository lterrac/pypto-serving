#pragma once

/**
 * The engine: one thread running scheduler -> executor -> detokenization.
 *
 * Callers hand over token ids and drain a per-request RequestStream. start()
 * registers the model -- the load, during which simpler forks its children --
 * and only then starts the thread.
 */

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <serving/config/types.hpp>
#include <serving/util/blocking_queue.hpp>
#include <serving/engine/detokenizer.hpp>
#include <serving/engine/executor.hpp>
#include <serving/memory/kv_cache.hpp>
#include <serving/model/tokenizer.hpp>
#include <serving/sched/scheduler.hpp>

namespace serving::engine
{

/// One streamed update for a request.
struct TokenOutput
{
  std::string requestId;
  /// Text produced since the previous update.
  std::string delta;
  /// The whole text, on the final update only: carrying it per token is
  /// quadratic in the generated length. Accumulate `delta` for a running value.
  std::string        text;
  std::optional<int> tokenId;
  bool               finished = false;
  std::string        finishReason;
  /// Set when the request failed. The answer is incomplete and the caller must
  /// report it as an error rather than a normal finish.
  std::string error;
};

/// A request's updates in order: the engine thread pushes, a caller thread pops
/// until nullopt.
using RequestStream = util::BlockingQueue<TokenOutput>;

struct EngineConfig
{
  sched::SchedulerConfig scheduler;
  config::RuntimeConfig  runtime;

  /// How long the engine thread waits when there is nothing to do.
  int idlePollMicroseconds = 200;

  /// How long a step may take before the engine calls itself stalled.
  ///
  /// Liveness needs a progress signal, not a pulse: a kernel or collective that
  /// hangs blocks the loop thread without it exiting, so a check that asks only
  /// whether the thread exists answers yes for ever and nothing rebuilds the
  /// replica. Measured only against scheduled work -- an idle engine is healthy,
  /// however long it has been idle -- and set well above the longest legitimate
  /// step, because declaring a slow prefill dead is worse than the hang.
  double stepStallSeconds = 300.0;
};

class Engine
{
  public:

  Engine(EngineConfig config, const model::TokenizerAdapter &tokenizer, ModelExecutor &executor);
  ~Engine();

  Engine(const Engine &)            = delete;
  Engine &operator=(const Engine &) = delete;

  /// Register the model, size the KV cache, then start the engine thread.
  void start();

  void stop();

  /// Whether the engine is up and its loop alive.
  [[nodiscard]] bool isReady() const;

  /// Seconds the current step has been outstanding, or 0 when the engine is
  /// idle or between steps. A step is outstanding from the moment it is handed
  /// to the executor until its result comes back.
  [[nodiscard]] double stepElapsedSeconds() const;

  /// Up, and getting somewhere -- what /health reports. False when a step has
  /// been outstanding longer than `stepStallSeconds`: the loop is alive but
  /// wedged, which is the state a liveness check exists to find.
  [[nodiscard]] bool isHealthy() const;

  /// Admit a prompt that the caller has already tokenized.
  std::shared_ptr<RequestStream> addRequest(const std::string &requestId, const std::vector<int> &promptTokenIds, const config::GenerateConfig &generateConfig);

  [[nodiscard]] std::string generateRequestId();

  /// Tokens admitted but not yet generated: the prompt still to prefill plus
  /// the generation still to come. The prefill half is the expensive one and
  /// the dominant part of a backlog, so a load signal that omitted it would
  /// report a replica sitting on thousands of queued prompt tokens as idle.
  [[nodiscard]] int pendingTokenLoad() const;

  /// Requests admitted and not yet finished. The scaling policy's queue-depth
  /// signal: unlike a token count it does not move with how long each answer
  /// happens to be.
  [[nodiscard]] int outstandingRequests() const;

  /// Requests preempted since start. A running request is preempted when the KV
  /// pool cannot grow it, and it then replays its prompt and everything it had
  /// generated -- so a non-zero and rising count means the deployment is
  /// thrashing and wants either fewer concurrent requests or more pages.
  [[nodiscard]] int preemptions() const { return _preemptions.load(); }

  void abortRequest(const std::string &requestId);

  private:

  struct RequestContext
  {
    sched::RequestPtr                       request;
    std::shared_ptr<RequestStream>          stream;
    std::shared_ptr<IncrementalDetokenizer> detokenizer;
    config::GenerateConfig                  generateConfig;
    std::string                             lastText;
  };

  void                      loop();
  void                      runOneStep();
  [[nodiscard]] StepCommand buildStepCommand(const sched::SchedulerOutput &output) const;
  void                      deliver(const std::vector<sched::RequestOutput> &outputs);
  void                      failRequest(const std::string &requestId, const std::string &reason);

  EngineConfig                   _config;
  const model::TokenizerAdapter &_tokenizer;
  ModelExecutor                 &_executor;

  memory::KvCacheManager            _kvCacheManager;
  std::unique_ptr<sched::Scheduler> _scheduler;

  mutable std::mutex                              _mutex;
  std::unordered_map<std::string, RequestContext> _contexts;

  std::thread _thread;
  /// Signalled by the engine thread once its loop is actually running, so
  /// start() can return a genuinely ready engine rather than one whose /health
  /// flaps for the first few microseconds.
  std::mutex              _startMutex;
  std::condition_variable _startCv;

  std::atomic<bool> _running{false};
  std::atomic<bool> _started{false};
  std::atomic<bool> _loopAlive{false};
  std::atomic<int>  _preemptions{0};
  std::atomic<int>  _requestCounter{0};

  /// When the step now in flight was handed to the executor, as steady-clock
  /// seconds; 0 when no step is outstanding. Written by the engine thread,
  /// read by whichever thread is answering /health.
  std::atomic<double> _stepStartedAt{0.0};
};

} // namespace serving::engine
