/**
 * Native Python bindings for the serving platform runtime and its channels.
 *
 * The Python process *is* the MPI rank: the platform runs inside it as a library.
 * See platform/docs/python-channel-contract.md for the rationale and the interface
 * this file implements.
 *
 * Three rules are load-bearing here:
 *   1. Input::read() copies the payload into a Python `bytes` object and pops in a
 *      single call, because Message::getData() points into the consumer ring buffer
 *      and is only valid until popMessage().
 *   2. Every call that blocks, or that a caller may poll, releases the GIL. See the
 *      per-method table in the contract document.
 *   3. Teardown is ordered and enforced, not assumed. The channels are MPI RMA windows
 *      and the channel controller runs on TaskR fibers; destroying either out of order
 *      is a segmentation fault rather than an exception. Platform.stop() is therefore
 *      idempotent, runs from the destructor as a last resort, and Runtime.finalize()
 *      stops every live platform before MPI_Finalize.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <mpi.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <hicr/core/instance.hpp>

#include <modules/channelController/module.hpp>
#include <modules/configuration/deployment.hpp>
#include <modules/service/module.hpp>
#include <system/channels/input.hpp>
#include <system/channels/message.hpp>
#include <system/channels/output.hpp>
#include <system/engine.hpp>

#include <channels/helpers.hpp>
#include <deployment/helpers.hpp>
#include <runtime/helpers.hpp>

namespace py = pybind11;

namespace
{

/// Raised as Python's built-in TimeoutError.
class TimeoutError final : public std::runtime_error
{
  public:

  using std::runtime_error::runtime_error;
};

/**
 * py::gil_scoped_release, but tolerant of not holding the GIL in the first place.
 *
 * The teardown paths run from destructors. pybind11 invokes those with the GIL held, but
 * they are also reachable from an unwind or from interpreter shutdown, and PyEval_SaveThread
 * on a thread that does not hold the GIL is a fatal error rather than a no-op. Ask the one
 * question that actually answers that -- do we hold it -- rather than whether the
 * interpreter is initialized, which is a different question with a misleading answer during
 * finalization.
 */
class gilRelease_t final
{
  public:

  gilRelease_t()
  {
    if (PyGILState_Check() != 0) _state = PyEval_SaveThread();
  }

  gilRelease_t(const gilRelease_t &)            = delete;
  gilRelease_t &operator=(const gilRelease_t &) = delete;

  ~gilRelease_t()
  {
    if (_state != nullptr) PyEval_RestoreThread(_state);
  }

  private:

  PyThreadState *_state = nullptr;
};

/**
 * RAII wrapper over serving::system::channels::Base::lock()/unlock().
 *
 * Output::pushMessageLocking takes that same mutex, but Output::isFull and the whole of
 * Input do not. As long as the GIL was held for the duration of every channel call, the
 * interpreter serialised them for free; now that the polled and blocking methods release
 * it, two Python threads on one handle would race for real. The channels are SPSC by
 * construction -- one producer thread, one consumer thread, per edge -- and that is still
 * the contract, but taking the channel mutex here means a misuse degrades to contention
 * instead of to corruption.
 */
class channelLock_t final
{
  public:

  explicit channelLock_t(serving::system::channels::Base &channel)
    : _channel(channel)
  {
    _channel.lock();
  }

  channelLock_t(const channelLock_t &)            = delete;
  channelLock_t &operator=(const channelLock_t &) = delete;

  ~channelLock_t() { _channel.unlock(); }

  private:

  serving::system::channels::Base &_channel;
};

/**
 * MPI_Init requires an argv that outlives the call, and the interpreter's own argv
 * is not stable C storage. Keep a process-wide copy of sys.argv for that purpose.
 */
struct processArguments_t
{
  std::vector<std::string> storage;
  std::vector<char *>      pointers;
  int                      count = 0;
};

__INLINE__ processArguments_t *makeProcessArguments()
{
  // Intentionally leaked: MPI may retain the pointers for the lifetime of the process.
  auto *arguments = new processArguments_t();

  auto systemArgv = py::module_::import("sys").attr("argv");
  for (const auto &argument : systemArgv) arguments->storage.push_back(py::cast<std::string>(argument));
  if (arguments->storage.empty()) arguments->storage.push_back("python");

  for (auto &argument : arguments->storage) arguments->pointers.push_back(argument.data());
  arguments->pointers.push_back(nullptr);
  arguments->count = static_cast<int>(arguments->storage.size());
  return arguments;
}

class PyPlatform;

/**
 * MPI + HiCR bootstrap. Thin wrapper over the shared example helper makeRuntime().
 */
class PyRuntime final
{
  public:

  explicit PyRuntime(const size_t computeResourceCount)
  {
    // makeRuntime() hands exactly two compute resources to TaskR and throws otherwise, and
    // it walks the device's compute resource iterator without a bound, so a larger value
    // runs off the end before it ever gets to that check. Refuse here where it can be said.
    if (computeResourceCount != 2) HICR_THROW_LOGIC("compute_resource_count must be 2; the underlying runtime supports no other value.");

    auto *arguments = makeProcessArguments();
    auto *argv      = arguments->pointers.data();

    // MPI_Init_thread is collective and taskr/hwloc bring-up is not instantaneous.
    {
      py::gil_scoped_release release;
      _runtime = std::make_unique<::Runtime>(makeRuntime(&arguments->count, &argv, computeResourceCount));
    }

    const auto &instance = _runtime->instanceManager->getCurrentInstance();
    _instanceId          = instance->getId();
    _isRoot              = instance->isRootInstance();
    _instanceCount       = _runtime->instanceManager->getInstances().size();
  }

  /**
   * Last-resort cleanup for a runtime the caller never finalized. Anything that throws
   * here would terminate the process, and the interpreter may already be shutting down,
   * so this is strictly best effort.
   */
  ~PyRuntime()
  {
    try
    {
      finalize();
    }
    catch (const std::exception &error)
    {
      fprintf(stderr, "[serving/platform] Ignoring error while finalizing the runtime from its destructor: %s\n", error.what());
    }
    catch (...)
    {
      fprintf(stderr, "[serving/platform] Ignoring unknown error while finalizing the runtime from its destructor.\n");
    }
  }

  [[nodiscard]] __INLINE__ ::Runtime &raw() const
  {
    if (_runtime == nullptr) HICR_THROW_RUNTIME("The platform runtime has already been finalized.");
    return *_runtime;
  }

  [[nodiscard]] __INLINE__ bool isFinalized() const { return _runtime == nullptr; }

  // Cached at construction: these are immutable facts about this rank, and error paths need
  // to be able to name the rank they are reporting on after the runtime has been finalized.
  [[nodiscard]] __INLINE__ HiCR::Instance::instanceId_t getInstanceId() const { return _instanceId; }
  [[nodiscard]] __INLINE__ bool                         isRoot() const { return _isRoot; }
  [[nodiscard]] __INLINE__ size_t                       getInstanceCount() const { return _instanceCount; }

  /**
   * Weakly, so a dropped Platform does not keep the runtime's teardown list growing, and
   * pruning as we go so a long-lived runtime does not accumulate dead control blocks.
   */
  __INLINE__ void registerPlatform(const std::shared_ptr<PyPlatform> &platform)
  {
    std::erase_if(_platforms, [](const std::weak_ptr<PyPlatform> &entry) { return entry.expired(); });
    _platforms.push_back(platform);
    _anyPlatformCreated = true;
  }

  [[nodiscard]] __INLINE__ bool hasAnyPlatformCreated() const { return _anyPlatformCreated; }

  __INLINE__ void notePlatformStarted() { _anyPlatformStarted = true; }

  [[nodiscard]] __INLINE__ bool hasAnyPlatformStarted() const { return _anyPlatformStarted; }

  /**
   * MPI_Finalize, after stopping every platform still standing on top of this runtime.
   *
   * The channels are MPI RMA windows and the channel controller runs on TaskR fibers, so
   * finalizing underneath a live platform does not raise -- it segfaults, either here or
   * later when the channels are freed past MPI_Finalize. The contract states the ordering
   * rule; this enforces it.
   */
  __INLINE__ void finalize()
  {
    if (_runtime == nullptr) return;

    stopPlatforms();

    {
      gilRelease_t release;
      _runtime->instanceManager->finalize();
    }

    // Member destruction order of ::Runtime puts the Engine before the RPC engine it
    // registered targets on, which is the order this must happen in.
    _runtime.reset();
  }

  /**
   * MPI_Abort. The only way for one rank to fail without wedging the others: its peers
   * would otherwise sit in Engine::await() forever, or in the collective MPI_Finalize.
   */
  [[noreturn]] __INLINE__ void abort(const int exitCode)
  {
    if (_runtime == nullptr)
    {
      fprintf(stderr, "[serving/platform] Runtime.abort() after finalize; exiting with %d.\n", exitCode);
      std::exit(exitCode);
    }
    gilRelease_t release;
    _runtime->instanceManager->abort(exitCode);
    std::exit(exitCode); // Not reached; abort() does not return.
  }

  private:

  __INLINE__ void stopPlatforms();

  std::unique_ptr<::Runtime>             _runtime;
  HiCR::Instance::instanceId_t           _instanceId         = 0;
  bool                                   _isRoot             = false;
  size_t                                 _instanceCount      = 0;
  bool                                   _anyPlatformCreated = false;
  bool                                   _anyPlatformStarted = false;
  std::vector<std::weak_ptr<PyPlatform>> _platforms;
};

/**
 * The parsed policy, its runtime managers and the rank -> partition assignment.
 */
class PyDeployment final
{
  public:

  PyDeployment() = default;

  [[nodiscard]] static __INLINE__ std::shared_ptr<PyDeployment> fromJsonFile(const std::string &path)
  {
    auto deployment = std::make_shared<PyDeployment>();
    loadDeploymentFromFile(path, deployment->_deployment);
    return deployment;
  }

  [[nodiscard]] __INLINE__ serving::configuration::Deployment &raw() { return _deployment; }

  __INLINE__ void assignEdgeManagersFrom(const std::shared_ptr<PyRuntime> &runtime)
  {
    auto &runtimeObject = runtime->raw();
    assignEdgeManagers(_deployment, runtimeObject.communicationManager.get(), runtimeObject.memoryManager.get(), runtimeObject.bufferMemorySpace);
    _edgeManagersAssigned = true;
  }

  __INLINE__ void assignInstancesFrom(const std::shared_ptr<PyRuntime> &runtime)
  {
    assignInstancesToPartitions(_deployment, runtime->raw().instanceManager);
    _instancesAssigned = true;
  }

  /**
   * Both steps are silent when skipped and fatal when needed: without edge managers the
   * channels have no memory manager to allocate from, and without instances every
   * partition keeps coordinator id 0, so rank 0 owns every edge and every other rank owns
   * none -- which deadlocks the reconciliation collective rather than reporting anything.
   */
  __INLINE__ void checkPrepared() const
  {
    if (_edgeManagersAssigned == false) HICR_THROW_LOGIC("Deployment.assign_edge_managers(runtime) must be called before creating a Platform.");
    if (_instancesAssigned == false) HICR_THROW_LOGIC("Deployment.assign_instances(runtime) must be called before creating a Platform.");
  }

  [[nodiscard]] __INLINE__ std::vector<std::string> getEdgeNames() const
  {
    std::vector<std::string> names;
    names.reserve(_deployment.getEdges().size());
    for (const auto &edge : _deployment.getEdges()) names.push_back(edge->getName());
    return names;
  }

  private:

  serving::configuration::Deployment _deployment;
  bool                               _edgeManagersAssigned = false;
  bool                               _instancesAssigned    = false;
};

/**
 * Consumer-side handle. Holds the channel weakly: the channel controller owns it, and
 * Platform.stop() must be able to tear it down deterministically before MPI finalizes.
 */
class PyInput final
{
  public:

  PyInput(std::shared_ptr<PyPlatform> platform, std::weak_ptr<serving::system::channels::Input> channel, std::string edgeName, const size_t bufferSize, const size_t capacity)
    : _platform(std::move(platform)),
      _channel(std::move(channel)),
      _edgeName(std::move(edgeName)),
      _bufferSize(bufferSize),
      _capacity(capacity)
  {}

  [[nodiscard]] __INLINE__ bool hasMessage() const
  {
    auto channel = lock();
    // Order matters: the channel mutex is taken after the GIL is dropped and released
    // before it is reacquired, so no thread ever waits on the channel while holding the GIL.
    py::gil_scoped_release release;
    channelLock_t          guard(*channel);
    return channel->hasMessage();
  }

  /**
   * getMessage + copy + popMessage, as one step. The GIL is deliberately held: neither
   * getMessage nor popMessage blocks, and the copy out of the ring buffer needs the GIL
   * anyway. The pointer handed out by getMessage() dies at popMessage().
   */
  [[nodiscard]] __INLINE__ py::bytes read() const
  {
    auto          channel = lock();
    channelLock_t guard(*channel);

    const auto message = channel->getMessage();
    py::bytes  payload(reinterpret_cast<const char *>(message.getData()), message.getSize());
    channel->popMessage();
    return payload;
  }

  [[nodiscard]] __INLINE__ bool isReady() const { return lock()->isReady(); }

  [[nodiscard]] __INLINE__ const std::string &getEdgeName() const { return _edgeName; }
  [[nodiscard]] __INLINE__ size_t             getBufferSize() const { return _bufferSize; }
  [[nodiscard]] __INLINE__ size_t             getCapacity() const { return _capacity; }

  private:

  [[nodiscard]] __INLINE__ std::shared_ptr<serving::system::channels::Input> lock() const
  {
    auto channel = _channel.lock();
    if (channel == nullptr) HICR_THROW_RUNTIME("Input channel '%s' has been closed.", _edgeName.c_str());
    return channel;
  }

  std::shared_ptr<PyPlatform>                     _platform;
  std::weak_ptr<serving::system::channels::Input> _channel;
  const std::string                               _edgeName;
  const size_t                                    _bufferSize;
  const size_t                                    _capacity;
};

/**
 * Producer-side handle. Same weak ownership as PyInput.
 */
class PyOutput final
{
  public:

  PyOutput(std::shared_ptr<PyPlatform> platform, std::weak_ptr<serving::system::channels::Output> channel, std::string edgeName, const size_t bufferSize, const size_t capacity)
    : _platform(std::move(platform)),
      _channel(std::move(channel)),
      _edgeName(std::move(edgeName)),
      _bufferSize(bufferSize),
      _capacity(capacity)
  {}

  /**
   * Drives two updateDepth() calls, i.e. MPI RMA progress, so the GIL is released for the
   * same reason Input::hasMessage() releases it: callers poll this for backpressure, and a
   * poll loop that never drops the GIL starves every other thread in the process.
   */
  [[nodiscard]] __INLINE__ bool isFull(const size_t messageSize) const
  {
    auto                   channel = lock();
    py::gil_scoped_release release;
    channelLock_t          guard(*channel);
    return channel->isFull(messageSize);
  }

  /**
   * pushMessageLocking spins with a 1 us sleep until the ring has room, so the GIL is
   * released around it. The payload pointer stays valid: `payload` keeps a reference to
   * the (immutable) bytes object alive across the release.
   *
   * Only `bytes` is accepted. bytearray and memoryview raise TypeError -- deliberately, so
   * that nothing mutable can be handed to a producer that reads it after the GIL is gone.
   */
  __INLINE__ void push(const py::bytes                                        &payload,
                       const serving::system::channels::Message::messageType_t messageType,
                       const serving::system::channels::Message::groupId_t     groupId,
                       const serving::system::channels::Message::sequenceId_t  sequenceId) const
  {
    auto channel = lock();

    char      *data = nullptr;
    Py_ssize_t size = 0;
    if (PYBIND11_BYTES_AS_STRING_AND_SIZE(payload.ptr(), &data, &size) != 0) throw py::error_already_set();

    // A payload larger than the whole payload ring can never fit, so pushMessageLocking
    // would spin on it forever -- and with the GIL released the interpreter never reaches a
    // bytecode boundary, so not even SIGINT would get the caller out of it.
    if (static_cast<size_t>(size) > _bufferSize)
      throw std::invalid_argument("Payload of " + std::to_string(size) + " bytes exceeds the '" + _edgeName + "' edge buffer size of " + std::to_string(_bufferSize) +
                                  " bytes; it can never be pushed. Raise 'Buffer Size' for this edge in the deployment policy.");

    const serving::system::channels::Message::metadata_t metadata{.type = messageType, .groupId = groupId, .sequenceId = sequenceId};
    if (metadata.isValid() == false) throw std::invalid_argument("Message metadata out of range (type/group/sequence).");

    const serving::system::channels::Message message(reinterpret_cast<const uint8_t *>(data), static_cast<size_t>(size), metadata);

    py::gil_scoped_release release;
    channel->pushMessageLocking(message);
  }

  [[nodiscard]] __INLINE__ bool isReady() const { return lock()->isReady(); }

  [[nodiscard]] __INLINE__ const std::string &getEdgeName() const { return _edgeName; }
  [[nodiscard]] __INLINE__ size_t             getBufferSize() const { return _bufferSize; }
  [[nodiscard]] __INLINE__ size_t             getCapacity() const { return _capacity; }

  private:

  [[nodiscard]] __INLINE__ std::shared_ptr<serving::system::channels::Output> lock() const
  {
    auto channel = _channel.lock();
    if (channel == nullptr) HICR_THROW_RUNTIME("Output channel '%s' has been closed.", _edgeName.c_str());
    return channel;
  }

  std::shared_ptr<PyPlatform>                      _platform;
  std::weak_ptr<serving::system::channels::Output> _channel;
  const std::string                                _edgeName;
  const size_t                                     _bufferSize;
  const size_t                                     _capacity;
};

/**
 * Engine + channelController + service module, wired exactly as
 * examples/modules/channelController does it.
 */
class PyPlatform final : public std::enable_shared_from_this<PyPlatform>
{
  public:

  PyPlatform(std::shared_ptr<PyRuntime> runtime, std::shared_ptr<PyDeployment> deployment)
    : _runtime(std::move(runtime)),
      _deployment(std::move(deployment))
  {
    _deployment->checkPrepared();

    auto &runtimeObject = _runtime->raw();
    _instanceId         = _runtime->getInstanceId();
    _isRoot             = _runtime->isRoot();

    const std::vector<HiCR::CommunicationManager *> managerOrder = {runtimeObject.communicationManager.get()};
    _channelController                                           = std::make_shared<serving::modules::channelController::Module>(_instanceId, managerOrder);

    // Endpoint metadata (source/target instance, channel id, edge) for this rank's edges.
    auto localChannels = buildLocalChannelInfos(_deployment->raw(), _instanceId, defaultChannelKeyBuilder);
    for (auto &input : localChannels.inputs)
    {
      // The helper eagerly builds a throwaway channel; the controller owns the real one.
      input.channel                       = nullptr;
      _localInputs[input.edge->getName()] = input;
    }
    for (auto &output : localChannels.outputs)
    {
      output.channel                        = nullptr;
      _localOutputs[output.edge->getName()] = output;
    }
  }

  /**
   * Last-resort teardown. Without it, an uncaught Python exception between start() and
   * stop() unwinds straight into ~Engine while TaskR service workers are still inside
   * reconcile() on boost fibers, which is a segmentation fault, not an exception -- and
   * the peers then wait forever for a STOP RPC that the dead root never sends.
   */
  ~PyPlatform()
  {
    try
    {
      stop();
    }
    catch (const std::exception &error)
    {
      fprintf(stderr, "[serving/platform] Ignoring error while stopping the platform from its destructor: %s\n", error.what());
    }
    catch (...)
    {
      fprintf(stderr, "[serving/platform] Ignoring unknown error while stopping the platform from its destructor.\n");
    }
  }

  [[nodiscard]] __INLINE__ std::shared_ptr<PyInput> openInput(const std::string &edgeName)
  {
    checkUsable();
    if (_startAttempted) HICR_THROW_LOGIC("Channels must be opened before Platform.start().");
    if (_openedInputs.contains(edgeName) == false)
    {
      if (_localInputs.contains(edgeName) == false) HICR_THROW_LOGIC("Edge '%s' is not consumed by this instance's partition.", edgeName.c_str());
      _openedInputs[edgeName] = createDesiredInput(_channelController, _localInputs.at(edgeName), defaultChannelKeyBuilder);
    }
    const auto &edge = _localInputs.at(edgeName).edge;
    return std::make_shared<PyInput>(shared_from_this(), _openedInputs.at(edgeName), edgeName, edge->getBufferSize(), edge->getBufferCapacity());
  }

  [[nodiscard]] __INLINE__ std::shared_ptr<PyOutput> openOutput(const std::string &edgeName)
  {
    checkUsable();
    if (_startAttempted) HICR_THROW_LOGIC("Channels must be opened before Platform.start().");
    if (_openedOutputs.contains(edgeName) == false)
    {
      if (_localOutputs.contains(edgeName) == false) HICR_THROW_LOGIC("Edge '%s' is not produced by this instance's partition.", edgeName.c_str());
      _openedOutputs[edgeName] = createDesiredOutput(_channelController, _localOutputs.at(edgeName), defaultChannelKeyBuilder);
    }
    const auto &edge = _localOutputs.at(edgeName).edge;
    return std::make_shared<PyOutput>(shared_from_this(), _openedOutputs.at(edgeName), edgeName, edge->getBufferSize(), edge->getBufferCapacity());
  }

  /**
   * Registers the modules and starts the engine. Blocks: initialize() runs the channel
   * controller's reconciliation collective, and on non-root instances run() waits for the
   * root's start RPC.
   */
  __INLINE__ void start()
  {
    checkUsable();
    if (_startAttempted) HICR_THROW_LOGIC("Platform.start() called twice.");

    agreeOnChannelClaims();
    _startAttempted = true;
    _runtime->notePlatformStarted();

    auto &runtimeObject = _runtime->raw();

    _serviceModule = std::make_shared<serving::modules::service::Module>(runtimeObject.taskr);
    _serviceModule->addService("ChannelController", _channelController->getService());

    runtimeObject.serving->addModule("ChannelController", _channelController);
    runtimeObject.serving->addModule("Service", _serviceModule);

    {
      py::gil_scoped_release release;
      runtimeObject.serving->initialize();
      runtimeObject.serving->run();
    }

    // Only now is there TaskR state to unwind. Setting this before the calls above would
    // make a throwing initialize() look like a running engine, and stop() would then await
    // a TaskR runtime that was never started.
    _engineRunning = true;
  }

  __INLINE__ void waitUntilReady(const double timeoutSeconds)
  {
    checkUsable();

    std::vector<std::shared_ptr<serving::system::channels::Base>> channels;
    for (const auto &[_, input] : _openedInputs) channels.push_back(input);
    for (const auto &[_, output] : _openedOutputs) channels.push_back(output);

    bool timedOut = false;
    {
      py::gil_scoped_release release;
      const auto             deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeoutSeconds);
      for (const auto &channel : channels)
      {
        while (channel->isReady() == false)
        {
          if (std::chrono::steady_clock::now() >= deadline)
          {
            timedOut = true;
            break;
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (timedOut) break;
      }
    }
    if (timedOut) throw TimeoutError("Timed out waiting for the platform channels to become ready.");
  }

  /**
   * Idempotent teardown, and re-entrant on failure. Drops the desired channels, then -- if
   * the engine actually got running -- terminates it (root only) and awaits it, which joins
   * the TaskR workers and destroys every channel this platform created.
   *
   * Safe before start() too: open_input/open_output already allocated MPI memory slots, so
   * an opened-then-abandoned platform must still release them before MPI_Finalize.
   *
   * The completion flag is set at the end, not the start: if the await below throws, the
   * destructor's last-resort stop() has to be able to run the rest of the unwind rather
   * than see a platform that claims to be stopped and return.
   */
  __INLINE__ void stop()
  {
    if (_stopped) return;
    _stopRequested = true;

    if (_channelController != nullptr)
    {
      for (const auto &[edgeName, _] : _openedOutputs) _channelController->removeDesiredProducer(edgeName);
      for (const auto &[edgeName, _] : _openedInputs) _channelController->removeDesiredConsumer(edgeName);
    }
    _openedOutputs.clear();
    _openedInputs.clear();

    if (_engineRunning && _runtime->isFinalized() == false)
    {
      // Consumed before the call that can throw, so a retry from the destructor takes the
      // branch below instead of awaiting an engine that has already been torn down.
      _engineRunning      = false;
      auto &runtimeObject = _runtime->raw();

      gilRelease_t release;
      if (_isRoot) runtimeObject.serving->terminate();
      runtimeObject.serving->await();
    }
    else if (_channelController != nullptr)
    {
      // The engine never ran, so nothing has cleared the controller -- and if start() threw
      // after addModule(), the Engine is holding it with no removeModule to take it back.
      // Emptying it here is what releases the MPI memory slots before MPI_Finalize; the
      // husk the Engine keeps owns nothing.
      _channelController->finalize();
    }

    _engineRunning = false;
    _channelController.reset();
    _serviceModule.reset();
    _stopped = true;
  }

  [[nodiscard]] __INLINE__ bool isStopped() const { return _stopped; }

  [[nodiscard]] __INLINE__ HiCR::Instance::instanceId_t getInstanceId() const { return _instanceId; }
  [[nodiscard]] __INLINE__ bool                         isRoot() const { return _isRoot; }

  private:

  __INLINE__ void checkUsable() const
  {
    if (_stopRequested) HICR_THROW_LOGIC("This Platform has been stopped and cannot be reused.");
  }

  /**
   * Agree who is opening what, before anyone commits to the collective that acts on it.
   *
   * channelController::reconcile() runs exchangeGlobalMemorySlots/fence, which is collective
   * over the whole communicator, but it only enters them when *this* rank has channels to
   * create -- and the slots it registers are per edge. Two different disagreements follow:
   * a rank that opened nothing skips a collective its peers are sitting in and the job
   * wedges silently; and a rank that opened a *different set* of edges than its peer enters
   * the collective and then asks for a global key nobody registered, which is a
   * segmentation fault rather than an exception.
   *
   * Counting channels only catches the first. Agreeing the claim set catches both, in the
   * same single collective: every rank walks the deployment's edges in the same order (same
   * JSON), marks the ones it opened as producer and as consumer, and sums. An edge must
   * then be claimed by exactly one producer and exactly one consumer, or by neither side --
   * skipping an edge entirely is legitimate and consistent, as long as *both* ends skip it.
   *
   * MPI is used directly because the runtime these bindings wrap is the MPI backend by
   * construction (makeRuntime hardcodes it), and HiCR exposes no allreduce.
   */
  __INLINE__ void agreeOnChannelClaims() const
  {
    const auto  &edges     = _deployment->raw().getEdges();
    const size_t edgeCount = edges.size();

    // [2i] = producers of edge i, [2i + 1] = consumers of edge i, [2 * edgeCount] = ranks
    // that opened anything at all.
    const size_t     participationIndex = 2 * edgeCount;
    std::vector<int> localClaims(participationIndex + 1, 0);
    std::vector<int> globalClaims(participationIndex + 1, 0);

    for (size_t edgeIndex = 0; edgeIndex < edgeCount; edgeIndex++)
    {
      const auto &edgeName = edges[edgeIndex]->getName();
      if (_openedOutputs.contains(edgeName)) localClaims[2 * edgeIndex] = 1;
      if (_openedInputs.contains(edgeName)) localClaims[2 * edgeIndex + 1] = 1;
    }
    localClaims[participationIndex] = (_openedInputs.empty() && _openedOutputs.empty()) ? 0 : 1;

    {
      py::gil_scoped_release release;
      MPI_Allreduce(localClaims.data(), globalClaims.data(), static_cast<int>(localClaims.size()), MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }

    const int participatingInstances = globalClaims[participationIndex];
    const int instanceCount          = static_cast<int>(_runtime->getInstanceCount());

    if (participatingInstances == 0)
      HICR_THROW_LOGIC("No instance opened any channel before Platform.start(); there is nothing to reconcile. Check the deployment policy's partitions and edges.");

    if (participatingInstances != instanceCount)
      HICR_THROW_LOGIC("Instance %lu: only %d of %d instances opened any channel, but the channel exchange is collective over all of them, and the ones that opened nothing "
                       "would never enter it. Every instance must own and open at least one edge. Check the deployment policy's partitions and edges.",
                       _instanceId,
                       participatingInstances,
                       instanceCount);

    for (size_t edgeIndex = 0; edgeIndex < edgeCount; edgeIndex++)
    {
      const int producerClaims = globalClaims[2 * edgeIndex];
      const int consumerClaims = globalClaims[2 * edgeIndex + 1];

      // Opened by both ends, or by neither. Anything else means the ranks disagree about
      // which memory slots exist.
      if (producerClaims == consumerClaims && producerClaims <= 1) continue;

      HICR_THROW_LOGIC("Instance %lu: edge '%s' was opened by %d producer(s) and %d consumer(s) across the deployment, but every edge must be opened by exactly one of each, "
                       "or by neither. The instances disagree about which channels exist, and the memory slot exchange would fault rather than fail. Open the edge on both "
                       "ends or on neither.",
                       _instanceId,
                       edges[edgeIndex]->getName().c_str(),
                       producerClaims,
                       consumerClaims);
    }
  }

  std::shared_ptr<PyRuntime>    _runtime;
  std::shared_ptr<PyDeployment> _deployment;
  HiCR::Instance::instanceId_t  _instanceId     = 0;
  bool                          _isRoot         = false;
  bool                          _startAttempted = false;
  bool                          _engineRunning  = false;
  bool                          _stopRequested  = false;
  bool                          _stopped        = false;

  std::shared_ptr<serving::modules::channelController::Module> _channelController;
  std::shared_ptr<serving::modules::service::Module>           _serviceModule;

  std::map<std::string, localInput_t>  _localInputs;
  std::map<std::string, localOutput_t> _localOutputs;

  std::map<std::string, std::shared_ptr<serving::system::channels::Input>>  _openedInputs;
  std::map<std::string, std::shared_ptr<serving::system::channels::Output>> _openedOutputs;
};

__INLINE__ void PyRuntime::stopPlatforms()
{
  for (const auto &weakPlatform : _platforms)
  {
    auto platform = weakPlatform.lock();
    if (platform == nullptr || platform->isStopped()) continue;
    fprintf(stderr, "[serving/platform] Runtime.finalize() with a live Platform; stopping it first.\n");

    // Best effort, per platform. This runs on the way to MPI_Finalize, and one platform
    // that cannot unwind must not stop the others from trying or leave MPI un-finalized.
    try
    {
      platform->stop();
    }
    catch (const std::exception &error)
    {
      fprintf(stderr, "[serving/platform] Ignoring error while stopping a platform during finalize: %s\n", error.what());
    }
    catch (...)
    {
      fprintf(stderr, "[serving/platform] Ignoring unknown error while stopping a platform during finalize.\n");
    }
  }
  _platforms.clear();
}

} // namespace

PYBIND11_MODULE(_native, module)
{
  module.doc() = "Native bindings for the PyPTO serving platform runtime and its channels.";

  py::register_exception_translator([](std::exception_ptr exception) {
    try
    {
      if (exception) std::rethrow_exception(exception);
    }
    catch (const TimeoutError &error)
    {
      PyErr_SetString(PyExc_TimeoutError, error.what());
    }
  });

  py::class_<PyRuntime, std::shared_ptr<PyRuntime>>(module, "Runtime")
    .def(py::init<size_t>(), py::arg("compute_resource_count") = 2)
    .def_property_readonly("instance_id", &PyRuntime::getInstanceId)
    .def_property_readonly("is_root", &PyRuntime::isRoot)
    .def_property_readonly("instance_count", &PyRuntime::getInstanceCount)
    .def_property_readonly("is_finalized", &PyRuntime::isFinalized)
    .def("finalize", &PyRuntime::finalize)
    .def("abort", &PyRuntime::abort, py::arg("exit_code") = -1)
    .def("__enter__", [](const std::shared_ptr<PyRuntime> &runtime) { return runtime; })
    .def("__exit__", [](PyRuntime &runtime, const py::object &exceptionType, const py::object &, const py::object &) {
      // Leaving the block on an exception once a Platform exists means this rank is walking
      // away from collectives its peers are already in -- the claim agreement in start(),
      // the slot exchange, or MPI_Finalize itself. Finalizing here would park them forever,
      // so take the job down instead. abort() does not return.
      if (exceptionType.is_none() == false && runtime.hasAnyPlatformCreated()) runtime.abort(1);
      runtime.finalize();
      return false;
    });

  py::class_<PyDeployment, std::shared_ptr<PyDeployment>>(module, "Deployment")
    .def_static("from_json_file", &PyDeployment::fromJsonFile, py::arg("path"))
    .def("assign_edge_managers", &PyDeployment::assignEdgeManagersFrom, py::arg("runtime"))
    .def("assign_instances", &PyDeployment::assignInstancesFrom, py::arg("runtime"))
    .def("edge_names", &PyDeployment::getEdgeNames);

  py::class_<PyInput, std::shared_ptr<PyInput>>(module, "Input")
    .def("has_message", &PyInput::hasMessage)
    .def("read", &PyInput::read)
    .def("is_ready", &PyInput::isReady)
    .def_property_readonly("edge_name", &PyInput::getEdgeName)
    .def_property_readonly("buffer_size", &PyInput::getBufferSize)
    .def_property_readonly("max_message_size", &PyInput::getBufferSize)
    .def_property_readonly("capacity", &PyInput::getCapacity);

  py::class_<PyOutput, std::shared_ptr<PyOutput>>(module, "Output")
    .def("is_full", &PyOutput::isFull, py::arg("message_size"))
    .def("push", &PyOutput::push, py::arg("payload"), py::arg("message_type") = 0, py::arg("group_id") = 0, py::arg("sequence_id") = 0)
    .def("is_ready", &PyOutput::isReady)
    .def_property_readonly("edge_name", &PyOutput::getEdgeName)
    .def_property_readonly("buffer_size", &PyOutput::getBufferSize)
    .def_property_readonly("max_message_size", &PyOutput::getBufferSize)
    .def_property_readonly("capacity", &PyOutput::getCapacity);

  py::class_<PyPlatform, std::shared_ptr<PyPlatform>>(module, "Platform")
    .def(py::init([](const std::shared_ptr<PyRuntime> &runtime, const std::shared_ptr<PyDeployment> &deployment) {
           auto platform = std::make_shared<PyPlatform>(runtime, deployment);
           runtime->registerPlatform(platform);
           return platform;
         }),
         py::arg("runtime"),
         py::arg("deployment"))
    .def("open_input", &PyPlatform::openInput, py::arg("edge_name"))
    .def("open_output", &PyPlatform::openOutput, py::arg("edge_name"))
    .def("start", &PyPlatform::start)
    .def("wait_until_ready", &PyPlatform::waitUntilReady, py::arg("timeout_s") = 30.0)
    .def("stop", &PyPlatform::stop)
    .def_property_readonly("is_stopped", &PyPlatform::isStopped)
    .def_property_readonly("instance_id", &PyPlatform::getInstanceId)
    .def_property_readonly("is_root", &PyPlatform::isRoot)
    .def("__enter__", [](const std::shared_ptr<PyPlatform> &platform) { return platform; })
    .def("__exit__", [](PyPlatform &platform, const py::object &, const py::object &, const py::object &) {
      platform.stop();
      return false;
    });
}
