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

/**
 * A request's output channel.
 *
 * The engine thread pushes; a caller thread pops. `pop` blocks until an update
 * arrives or the stream finishes, so an SSE handler is a plain loop.
 */
class RequestStream
{
  public:

  void push(TokenOutput output);

  /// Next update, or nullopt once the stream is finished and drained.
  std::optional<TokenOutput> pop();

  /// Mark the stream complete; a blocked `pop` returns once drained.
  void close();

  [[nodiscard]] bool closed() const;

  private:

  mutable std::mutex      _mutex;
  std::condition_variable _cv;
  std::deque<TokenOutput> _queue;
  bool                    _closed = false;
};

struct EngineConfig
{
  sched::SchedulerConfig scheduler;
  config::RuntimeConfig  runtime;
  config::GenerateConfig defaults;

  /// How long the engine thread waits when there is nothing to do.
  int idlePollMicroseconds = 200;
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

  /// Whether the engine is up and its loop alive -- what /health reports.
  [[nodiscard]] bool isReady() const;

  /// Admit a prompt that the caller has already tokenized.
  std::shared_ptr<RequestStream> addRequest(const std::string &requestId, const std::vector<int> &promptTokenIds, const config::GenerateConfig &generateConfig);

  [[nodiscard]] std::string generateRequestId();

  /// Tokens admitted but not yet generated; the router's load signal.
  [[nodiscard]] int pendingTokenLoad() const;

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
  std::atomic<int>  _requestCounter{0};
};

} // namespace serving::engine
