/**
 * Native Python bindings for the serving platform runtime and its channels.
 *
 * The Python process *is* the MPI rank: the platform runs inside it as a library.
 * See platform/docs/python-channel-contract.md for the rationale and the interface
 * this file implements.
 *
 * Two rules are load-bearing here:
 *   1. Input::read() copies the payload into a Python `bytes` object and pops in a
 *      single call, because Message::getData() points into the consumer ring buffer
 *      and is only valid until popMessage().
 *   2. Every call that can block (pushing to a full channel, the collective run by
 *      the channel controller's reconciliation, the readiness spin, MPI finalize)
 *      releases the GIL, or the caller's asyncio loop deadlocks against it.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cstdint>
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

/**
 * MPI + HiCR bootstrap. Thin wrapper over the shared example helper makeRuntime().
 */
class PyRuntime final
{
  public:

  explicit PyRuntime(const size_t computeResourceCount)
  {
    auto *arguments = makeProcessArguments();
    auto *argv      = arguments->pointers.data();

    // MPI_Init_thread is collective and taskr/hwloc bring-up is not instantaneous.
    py::gil_scoped_release release;
    _runtime = std::make_unique<::Runtime>(makeRuntime(&arguments->count, &argv, computeResourceCount));
  }

  ~PyRuntime() = default;

  [[nodiscard]] __INLINE__ ::Runtime &raw() const
  {
    if (_runtime == nullptr) HICR_THROW_RUNTIME("The platform runtime has already been finalized.");
    return *_runtime;
  }

  [[nodiscard]] __INLINE__ HiCR::Instance::instanceId_t getInstanceId() const { return raw().instanceManager->getCurrentInstance()->getId(); }
  [[nodiscard]] __INLINE__ bool                         isRoot() const { return raw().instanceManager->getCurrentInstance()->isRootInstance(); }
  [[nodiscard]] __INLINE__ size_t                       getInstanceCount() const { return raw().instanceManager->getInstances().size(); }

  __INLINE__ void finalize()
  {
    if (_runtime == nullptr) return;
    _runtime->instanceManager->finalize();
    // The Engine holds RPC targets registered on the RPC engine, so it must go first;
    // that is the member destruction order of ::Runtime.
    _runtime.reset();
  }

  private:

  std::unique_ptr<::Runtime> _runtime;
};

/**
 * The parsed policy plus the rank -> partition assignment.
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
  }

  __INLINE__ void assignInstancesFrom(const std::shared_ptr<PyRuntime> &runtime) { assignInstancesToPartitions(_deployment, runtime->raw().instanceManager); }

  [[nodiscard]] __INLINE__ std::vector<std::string> getEdgeNames() const
  {
    std::vector<std::string> names;
    names.reserve(_deployment.getEdges().size());
    for (const auto &edge : _deployment.getEdges()) names.push_back(edge->getName());
    return names;
  }

  private:

  serving::configuration::Deployment _deployment;
};

class PyPlatform;

/**
 * Consumer-side handle. Holds the channel weakly: the channel controller owns it, and
 * Platform.stop() must be able to tear it down deterministically before MPI finalizes.
 */
class PyInput final
{
  public:

  PyInput(std::shared_ptr<PyPlatform> platform, std::weak_ptr<serving::system::channels::Input> channel, std::string edgeName)
    : _platform(std::move(platform)),
      _channel(std::move(channel)),
      _edgeName(std::move(edgeName))
  {}

  [[nodiscard]] __INLINE__ bool hasMessage() const
  {
    auto                   channel = lock();
    py::gil_scoped_release release;
    return channel->hasMessage();
  }

  /**
   * getMessage + copy + popMessage, as one step. The GIL is deliberately held: neither
   * getMessage nor popMessage blocks, and the copy out of the ring buffer needs the GIL
   * anyway. The pointer handed out by getMessage() dies at popMessage().
   */
  [[nodiscard]] __INLINE__ py::bytes read() const
  {
    auto       channel = lock();
    const auto message = channel->getMessage();
    py::bytes  payload(reinterpret_cast<const char *>(message.getData()), message.getSize());
    channel->popMessage();
    return payload;
  }

  [[nodiscard]] __INLINE__ bool isReady() const { return lock()->isReady(); }

  [[nodiscard]] __INLINE__ const std::string &getEdgeName() const { return _edgeName; }

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
};

/**
 * Producer-side handle. Same weak ownership as PyInput.
 */
class PyOutput final
{
  public:

  PyOutput(std::shared_ptr<PyPlatform> platform, std::weak_ptr<serving::system::channels::Output> channel, std::string edgeName)
    : _platform(std::move(platform)),
      _channel(std::move(channel)),
      _edgeName(std::move(edgeName))
  {}

  [[nodiscard]] __INLINE__ bool isFull(const size_t messageSize) const { return lock()->isFull(messageSize); }

  /**
   * pushMessageLocking spins with a 1 us sleep until the ring has room, so the GIL is
   * released around it. The payload pointer stays valid: `payload` keeps a reference to
   * the (immutable) bytes object alive across the release.
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

    const serving::system::channels::Message::metadata_t metadata{.type = messageType, .groupId = groupId, .sequenceId = sequenceId};
    if (metadata.isValid() == false) throw std::invalid_argument("Message metadata out of range (type/group/sequence).");

    const serving::system::channels::Message message(reinterpret_cast<const uint8_t *>(data), static_cast<size_t>(size), metadata);

    py::gil_scoped_release release;
    channel->pushMessageLocking(message);
  }

  [[nodiscard]] __INLINE__ bool isReady() const { return lock()->isReady(); }

  [[nodiscard]] __INLINE__ const std::string &getEdgeName() const { return _edgeName; }

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

  ~PyPlatform() = default;

  [[nodiscard]] __INLINE__ std::shared_ptr<PyInput> openInput(const std::string &edgeName)
  {
    if (_started) HICR_THROW_LOGIC("Channels must be opened before Platform.start().");
    if (_openedInputs.contains(edgeName) == false)
    {
      if (_localInputs.contains(edgeName) == false) HICR_THROW_LOGIC("Edge '%s' is not consumed by this instance's partition.", edgeName.c_str());
      _openedInputs[edgeName] = createDesiredInput(_channelController, _localInputs.at(edgeName), defaultChannelKeyBuilder);
    }
    return std::make_shared<PyInput>(shared_from_this(), _openedInputs.at(edgeName), edgeName);
  }

  [[nodiscard]] __INLINE__ std::shared_ptr<PyOutput> openOutput(const std::string &edgeName)
  {
    if (_started) HICR_THROW_LOGIC("Channels must be opened before Platform.start().");
    if (_openedOutputs.contains(edgeName) == false)
    {
      if (_localOutputs.contains(edgeName) == false) HICR_THROW_LOGIC("Edge '%s' is not produced by this instance's partition.", edgeName.c_str());
      _openedOutputs[edgeName] = createDesiredOutput(_channelController, _localOutputs.at(edgeName), defaultChannelKeyBuilder);
    }
    return std::make_shared<PyOutput>(shared_from_this(), _openedOutputs.at(edgeName), edgeName);
  }

  /**
   * Registers the modules and starts the engine. Blocks: initialize() runs the channel
   * controller's reconciliation collective, and on non-root instances run() waits for the
   * root's start RPC.
   */
  __INLINE__ void start()
  {
    if (_started) HICR_THROW_LOGIC("Platform.start() called twice.");
    _started = true;

    auto &runtimeObject = _runtime->raw();

    _serviceModule = std::make_shared<serving::modules::service::Module>(runtimeObject.taskr);
    _serviceModule->addService("ChannelController", _channelController->getService());

    runtimeObject.serving->addModule("ChannelController", _channelController);
    runtimeObject.serving->addModule("Service", _serviceModule);

    py::gil_scoped_release release;
    runtimeObject.serving->initialize();
    runtimeObject.serving->run();
  }

  __INLINE__ void waitUntilReady(const double timeoutSeconds)
  {
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
   * Drops the desired channels, terminates the engine (root only) and awaits it. After
   * this returns every channel this platform created is destroyed, so the runtime can be
   * finalized without freeing MPI memory past MPI_Finalize.
   */
  __INLINE__ void stop()
  {
    if (_started == false || _stopped) return;
    _stopped = true;

    auto &runtimeObject = _runtime->raw();

    for (const auto &[edgeName, _] : _openedOutputs) _channelController->removeDesiredProducer(edgeName);
    for (const auto &[edgeName, _] : _openedInputs) _channelController->removeDesiredConsumer(edgeName);
    _openedOutputs.clear();
    _openedInputs.clear();

    py::gil_scoped_release release;
    if (_isRoot) runtimeObject.serving->terminate();
    runtimeObject.serving->await();
  }

  [[nodiscard]] __INLINE__ HiCR::Instance::instanceId_t getInstanceId() const { return _instanceId; }
  [[nodiscard]] __INLINE__ bool                         isRoot() const { return _isRoot; }

  private:

  std::shared_ptr<PyRuntime>    _runtime;
  std::shared_ptr<PyDeployment> _deployment;
  HiCR::Instance::instanceId_t  _instanceId = 0;
  bool                          _isRoot     = false;
  bool                          _started    = false;
  bool                          _stopped    = false;

  std::shared_ptr<serving::modules::channelController::Module> _channelController;
  std::shared_ptr<serving::modules::service::Module>           _serviceModule;

  std::map<std::string, localInput_t>  _localInputs;
  std::map<std::string, localOutput_t> _localOutputs;

  std::map<std::string, std::shared_ptr<serving::system::channels::Input>>  _openedInputs;
  std::map<std::string, std::shared_ptr<serving::system::channels::Output>> _openedOutputs;
};

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
    .def("finalize", &PyRuntime::finalize, py::call_guard<py::gil_scoped_release>());

  py::class_<PyDeployment, std::shared_ptr<PyDeployment>>(module, "Deployment")
    .def_static("from_json_file", &PyDeployment::fromJsonFile, py::arg("path"))
    .def("assign_edge_managers", &PyDeployment::assignEdgeManagersFrom, py::arg("runtime"))
    .def("assign_instances", &PyDeployment::assignInstancesFrom, py::arg("runtime"))
    .def("edge_names", &PyDeployment::getEdgeNames);

  py::class_<PyInput, std::shared_ptr<PyInput>>(module, "Input")
    .def("has_message", &PyInput::hasMessage)
    .def("read", &PyInput::read)
    .def("is_ready", &PyInput::isReady)
    .def_property_readonly("edge_name", &PyInput::getEdgeName);

  py::class_<PyOutput, std::shared_ptr<PyOutput>>(module, "Output")
    .def("is_full", &PyOutput::isFull, py::arg("message_size"))
    .def("push", &PyOutput::push, py::arg("payload"), py::arg("message_type") = 0, py::arg("group_id") = 0, py::arg("sequence_id") = 0)
    .def("is_ready", &PyOutput::isReady)
    .def_property_readonly("edge_name", &PyOutput::getEdgeName);

  py::class_<PyPlatform, std::shared_ptr<PyPlatform>>(module, "Platform")
    .def(py::init<std::shared_ptr<PyRuntime>, std::shared_ptr<PyDeployment>>(), py::arg("runtime"), py::arg("deployment"))
    .def("open_input", &PyPlatform::openInput, py::arg("edge_name"))
    .def("open_output", &PyPlatform::openOutput, py::arg("edge_name"))
    .def("start", &PyPlatform::start)
    .def("wait_until_ready", &PyPlatform::waitUntilReady, py::arg("timeout_s") = 30.0)
    .def("stop", &PyPlatform::stop)
    .def_property_readonly("instance_id", &PyPlatform::getInstanceId)
    .def_property_readonly("is_root", &PyPlatform::isRoot);
}
