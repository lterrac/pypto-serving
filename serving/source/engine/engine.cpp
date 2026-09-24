#include <serving/engine/engine.hpp>

#include <chrono>
#include <stdexcept>

namespace serving::engine
{

// ---------------------------------------------------------------------------
// RequestStream
// ---------------------------------------------------------------------------

void RequestStream::push(TokenOutput output)
{
  {
    const std::lock_guard<std::mutex> lock(_mutex);
    if (_closed) { return; }
    _queue.push_back(std::move(output));
  }
  _cv.notify_one();
}

std::optional<TokenOutput> RequestStream::pop()
{
  std::unique_lock<std::mutex> lock(_mutex);
  _cv.wait(lock, [this] { return !_queue.empty() || _closed; });
  if (_queue.empty()) { return std::nullopt; }
  TokenOutput output = std::move(_queue.front());
  _queue.pop_front();
  return output;
}

void RequestStream::close()
{
  {
    const std::lock_guard<std::mutex> lock(_mutex);
    _closed = true;
  }
  _cv.notify_all();
}

bool RequestStream::closed() const
{
  const std::lock_guard<std::mutex> lock(_mutex);
  return _closed;
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

Engine::Engine(EngineConfig config, const model::TokenizerAdapter &tokenizer, ModelExecutor &executor)
  : _config(std::move(config)),
    _tokenizer(tokenizer),
    _executor(executor),
    _kvCacheManager(std::nullopt, _config.runtime.pageSize, _config.scheduler.enablePrefixCache)
{}

Engine::~Engine() { stop(); }

void Engine::start()
{
  if (_started.exchange(true)) { return; }

  // Model first, thread second. Once the executor is the Python bridge this is
  // the call that forks simpler's per-chip children, and that must happen while
  // the process is still single-threaded.
  try
  {
    const int numPages = _executor.registerModel();
    _kvCacheManager.initialize(_config.runtime, numPages);
    _scheduler = std::make_unique<sched::Scheduler>(_config.scheduler, _kvCacheManager);
  }
  catch (...)
  {
    // Loading the model is the common failure. Leaving _started set would make
    // addRequest admit into a null scheduler.
    _started = false;
    throw;
  }

  _running = true;
  _thread  = std::thread([this] { loop(); });

  // Wait for the loop to come up. Without this, isReady() -- and so /health --
  // reports false for a moment after start() returns, which reads as a dead
  // engine to anything polling it.
  std::unique_lock<std::mutex> lock(_startMutex);
  _startCv.wait(lock, [this] { return _loopAlive.load() || !_running.load(); });
}

void Engine::stop()
{
  if (!_running.exchange(false)) { return; }
  if (_thread.joinable()) { _thread.join(); }

  std::vector<std::shared_ptr<RequestStream>> streams;
  {
    const std::lock_guard<std::mutex> lock(_mutex);
    for (auto &[id, ctx] : _contexts) { streams.push_back(ctx.stream); }
    _contexts.clear();
  }
  for (const auto &stream : streams) { stream->close(); }
  _executor.close();
}

bool Engine::isReady() const { return _started.load() && _running.load() && _loopAlive.load(); }

std::string Engine::generateRequestId() { return "serving-req-" + std::to_string(++_requestCounter); }

int Engine::pendingTokenLoad() const
{
  const std::lock_guard<std::mutex> lock(_mutex);
  int                               pending = 0;
  for (const auto &[id, ctx] : _contexts) { pending += std::max(0, ctx.request->maxNewTokens - static_cast<int>(ctx.request->outputTokenIds.size())); }
  return pending;
}

std::shared_ptr<RequestStream> Engine::addRequest(const std::string &requestId, const std::vector<int> &promptTokenIds, const config::GenerateConfig &generateConfig)
{
  if (!isReady()) { throw std::runtime_error("engine is not ready"); }

  auto request            = std::make_shared<sched::Request>();
  request->requestId      = requestId;
  request->promptTokenIds = promptTokenIds;
  request->maxNewTokens   = generateConfig.maxNewTokens;
  request->temperature    = generateConfig.temperature;
  request->topP           = generateConfig.topP;
  request->topK           = generateConfig.topK;
  request->seed           = generateConfig.seed;
  request->stopStrings    = generateConfig.stop;
  request->arrivalTime    = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  if (!generateConfig.ignoreEos) { request->eosTokenId = _tokenizer.eosTokenId(); }

  auto stream = std::make_shared<RequestStream>();

  RequestContext ctx;
  ctx.request        = request;
  ctx.stream         = stream;
  ctx.detokenizer    = std::make_shared<IncrementalDetokenizer>(_tokenizer);
  ctx.generateConfig = generateConfig;

  {
    const std::lock_guard<std::mutex> lock(_mutex);
    // addRequest validates and can throw; do it under the lock so a rejected
    // request never leaves a half-registered context behind.
    _scheduler->addRequest(request);
    _contexts.emplace(requestId, std::move(ctx));
  }
  return stream;
}

void Engine::abortRequest(const std::string &requestId)
{
  std::shared_ptr<RequestStream> stream;
  {
    const std::lock_guard<std::mutex> lock(_mutex);
    const auto                        it = _contexts.find(requestId);
    if (it == _contexts.end()) { return; }
    stream = it->second.stream;
    _scheduler->abortRequest(requestId);
    _contexts.erase(it);
  }
  stream->close();
}

void Engine::loop()
{
  {
    const std::lock_guard<std::mutex> lock(_startMutex);
    _loopAlive = true;
  }
  _startCv.notify_all();

  while (_running.load())
  {
    try
    {
      runOneStep();
    }
    catch (const std::exception &)
    {
      // A step must not take the loop down: the requests it touched are failed
      // in runOneStep, and the engine stays up for everyone else.
    }
  }
  _loopAlive = false;
}

void Engine::runOneStep()
{
  sched::SchedulerOutput output;
  {
    const std::lock_guard<std::mutex> lock(_mutex);
    if (!_scheduler->hasWork())
    {
      // Nothing to do; drop the lock before sleeping.
    }
    else { output = _scheduler->schedule(); }
  }

  if (output.scheduledRequests.empty())
  {
    std::this_thread::sleep_for(std::chrono::microseconds(_config.idlePollMicroseconds));
    // Rejections still need delivering even when nothing was scheduled.
    if (!output.rejectedRequests.empty())
    {
      for (const auto &[requestId, reason] : output.rejectedRequests) { failRequest(requestId, reason); }
    }
    return;
  }

  for (const auto &[requestId, reason] : output.rejectedRequests) { failRequest(requestId, reason); }

  const StepCommand command = buildStepCommand(output);
  const StepResult  result  = _executor.executeStep(command);

  if (!result.error.empty())
  {
    for (const auto &scheduled : output.scheduledRequests) { failRequest(scheduled.request->requestId, result.error); }
    return;
  }

  std::vector<sched::RequestOutput> requestOutputs;
  {
    const std::lock_guard<std::mutex> lock(_mutex);
    requestOutputs = _scheduler->updateFromOutput(output, result.newTokens);
  }
  deliver(requestOutputs);
}

StepCommand Engine::buildStepCommand(const sched::SchedulerOutput &output) const
{
  StepCommand command;
  for (const auto &scheduled : output.scheduledRequests)
  {
    const sched::Request &request = *scheduled.request;
    if (scheduled.isPrefill)
    {
      PrefillItem item;
      item.requestId         = request.requestId;
      item.blockIds          = scheduled.blockIds;
      item.numComputedTokens = scheduled.numComputedTokens;
      item.promptLen         = request.numPromptTokens();

      item.chunkTokens = request.tokenRange(scheduled.numComputedTokens, scheduled.numComputedTokens + scheduled.numNewTokens);
      command.prefill.push_back(std::move(item));
    }
    else
    {
      DecodeItem item;
      item.requestId = request.requestId;
      item.blockIds  = scheduled.blockIds;
      item.lastToken = request.outputTokenIds.empty() ? 0 : request.outputTokenIds.back();
      item.seqLen    = request.numPromptTokens() + static_cast<int>(request.outputTokenIds.size());
      command.decode.push_back(std::move(item));
    }
  }
  return command;
}

namespace
{

/// Offset of the earliest stop string in `text`, or npos. The stop text itself
/// is not part of the answer, so the caller truncates there.
size_t firstStopOffset(const std::string &text, const std::vector<std::string> &stopStrings)
{
  size_t earliest = std::string::npos;
  for (const std::string &stop : stopStrings)
  {
    if (stop.empty()) { continue; }
    const size_t at = text.find(stop);
    if (at != std::string::npos && at < earliest) { earliest = at; }
  }
  return earliest;
}

} // namespace

void Engine::deliver(const std::vector<sched::RequestOutput> &outputs)
{
  for (const sched::RequestOutput &out : outputs)
  {
    std::shared_ptr<RequestStream>          stream;
    std::shared_ptr<IncrementalDetokenizer> detokenizer;
    sched::RequestPtr                       request;
    std::string                             lastText;

    {
      const std::lock_guard<std::mutex> lock(_mutex);
      const auto                        it = _contexts.find(out.requestId);
      if (it == _contexts.end()) { continue; }
      stream      = it->second.stream;
      detokenizer = it->second.detokenizer;
      request     = it->second.request;
      lastText    = it->second.lastText;
    }

    // Unlocked: with a Python tokenizer this takes the GIL, and holding _mutex
    // across that inverts the lock order against any Python thread calling in.
    // Only this thread touches the detokenizer or the request, and the shared
    // pointers keep both alive if the context is erased meanwhile.
    std::string text;
    try
    {
      text = out.finished ? detokenizer->finalize(request->outputTokenIds) : detokenizer->append(request->outputTokenIds);
    }
    catch (const std::exception &e)
    {
      failRequest(out.requestId, std::string("detokenization failed: ") + e.what());
      continue;
    }

    bool        finished     = out.finished;
    std::string finishReason = out.finishReason;
    if (!finished)
    {
      // Stop strings match the decoded text, so the scheduler cannot check them.
      const size_t cut = firstStopOffset(text, request->stopStrings);
      if (cut != std::string::npos)
      {
        text.resize(cut);
        finished     = true;
        finishReason = "FINISHED_STOP";
      }
    }

    TokenOutput update;
    update.requestId    = out.requestId;
    update.delta        = text.size() >= lastText.size() ? text.substr(lastText.size()) : std::string{};
    update.tokenId      = out.newTokenId;
    update.finished     = finished;
    update.finishReason = finishReason;
    if (finished) { update.text = text; }

    {
      const std::lock_guard<std::mutex> lock(_mutex);
      const auto                        it = _contexts.find(out.requestId);
      if (it == _contexts.end()) { continue; }
      if (!finished) { it->second.lastText = std::move(text); }
      else
      {
        // A stop string ends the request here rather than in the scheduler, so
        // its blocks and its slot have to be released explicitly.
        if (!out.finished && _scheduler != nullptr) { _scheduler->finishRequest(out.requestId, sched::RequestStatus::FinishedStop); }
        _contexts.erase(it);
      }
    }

    stream->push(std::move(update));
    if (finished) { stream->close(); }
  }
}

void Engine::failRequest(const std::string &requestId, const std::string &reason)
{
  std::shared_ptr<RequestStream> stream;
  {
    const std::lock_guard<std::mutex> lock(_mutex);
    const auto                        it = _contexts.find(requestId);
    if (it == _contexts.end()) { return; }
    stream = it->second.stream;
    // Abort it in the scheduler too, or it stays running: rescheduled every step,
    // its blocks never freed, failing whatever is batched with it.
    _scheduler->abortRequest(requestId);
    _contexts.erase(it);
  }
  TokenOutput update;
  update.requestId    = requestId;
  update.finished     = true;
  update.finishReason = "FINISHED_ERROR";
  update.error        = reason.empty() ? std::string{"request failed"} : reason;
  stream->push(std::move(update));
  stream->close();
}

} // namespace serving::engine
