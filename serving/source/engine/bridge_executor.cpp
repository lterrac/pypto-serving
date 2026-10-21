#include <serving/engine/bridge_executor.hpp>

#include <Python.h>

#include <stdexcept>

#include <nlohmann/json.hpp>

namespace serving::engine
{

namespace
{

using Json = nlohmann::json;

/// Format and clear the pending Python exception.
std::string takePythonError(const char *what)
{
  if (PyErr_Occurred() == nullptr) { return {}; }
  PyObject *type  = nullptr;
  PyObject *value = nullptr;
  PyObject *tb    = nullptr;
  PyErr_Fetch(&type, &value, &tb);
  PyErr_NormalizeException(&type, &value, &tb);

  std::string message = std::string{what} + ": ";
  if (value != nullptr)
  {
    PyObject *str = PyObject_Str(value);
    if (str != nullptr)
    {
      const char *utf8 = PyUnicode_AsUTF8(str);
      if (utf8 != nullptr) { message += utf8; }
      Py_DECREF(str);
    }
  }
  Py_XDECREF(type);
  Py_XDECREF(value);
  Py_XDECREF(tb);
  return message;
}

} // namespace

/// Holds the interpreter state so <Python.h> stays out of the header.
struct BridgeExecutor::Impl
{
  PyObject *bridge      = nullptr;
  bool      initialized = false;
  /// Saved when the GIL is handed back after model bring-up; see registerModel.
  PyThreadState *mainState = nullptr;
};

BridgeExecutor::BridgeExecutor(BridgeConfig config)
  : _impl(std::make_unique<Impl>()),
    _config(std::move(config))
{
  if (Py_IsInitialized() == 0)
  {
    // Pre-initialised the way the `python3` binary does, not with a bare
    // Py_Initialize.
    //
    // `python3` applies PEP 538/540 during startup: under a C or POSIX locale
    // it turns UTF-8 mode on, so `open()` reads UTF-8 whatever the environment
    // says. Embedding skips that, and the interpreter is left using the
    // locale's encoding -- ASCII under the task broker, which sets no LANG. A
    // model is then loaded by reading files that are UTF-8 by definition:
    // Qwen3-14B's tokenizer.json has a non-ASCII byte 7483 in, and the load
    // died there with "'ascii' codec can't decode byte 0xc2", nowhere near
    // anything that looked like an encoding problem.
    //
    // Asked for explicitly rather than inferred, because the right answer does
    // not depend on the locale: every file this reads is UTF-8.
    PyPreConfig preConfig;
    PyPreConfig_InitPythonConfig(&preConfig);
    preConfig.utf8_mode = 1;
    if (const PyStatus status = Py_PreInitialize(&preConfig); PyStatus_Exception(status))
    {
      throw std::runtime_error(std::string("could not pre-initialize the embedded interpreter: ") + (status.err_msg != nullptr ? status.err_msg : "unknown"));
    }

    Py_Initialize();
    _impl->initialized = true;
  }

  if (!_config.repoRoot.empty())
  {
    const std::string code = "import sys; sys.path.insert(0, r'" + _config.repoRoot + "')";
    if (PyRun_SimpleString(code.c_str()) != 0) { throw std::runtime_error("could not extend sys.path"); }
  }

  _impl->bridge = PyImport_ImportModule("pypto_serving.bridge");
  if (_impl->bridge == nullptr) { throw std::runtime_error(takePythonError("importing pypto_serving.bridge")); }
}

BridgeExecutor::~BridgeExecutor()
{
  close();
  // Py_Finalize is not called: simpler's forked children and the device state
  // do not survive it.
}

int BridgeExecutor::registerModel()
{
  // Called from Engine::start() while the process is still single-threaded --
  // this is the call that forks simpler's per-chip children.
  //
  // Loading the model is the common failure, and every exit from here has to
  // hand the GIL back. Thrown out while still holding it, no other thread can
  // ever enter Python again -- including the close() this object's own
  // destructor runs on the way out.
  const auto releaseGil = [this] {
    if (_impl->mainState == nullptr) { _impl->mainState = PyEval_SaveThread(); }
  };
  const auto fail = [&](const char *what) {
    std::string message = takePythonError(what);
    if (message.empty()) { message = std::string{what} + ": failed"; }
    releaseGil();
    return std::runtime_error(message);
  };

  PyObject *pages = PyObject_CallMethod(
    _impl->bridge, "open_model", "sisiii", _config.modelDir.c_str(), _config.deviceId, _config.platform.c_str(), _config.maxModelLen, _config.blockSize, _config.maxNumSeqs);
  if (pages == nullptr) { throw fail("bridge.open_model"); }
  const long numPages = PyLong_AsLong(pages);
  Py_DECREF(pages);
  // -1 with an exception set is how PyLong_AsLong reports a non-integer. Left
  // unchecked it both feeds a bogus page count to the KV pool and leaves the
  // exception pending for whatever calls Python next.
  if (numPages == -1 && PyErr_Occurred() != nullptr) { throw fail("bridge.open_model returned a non-integer page count"); }

  PyObject *pageSize = PyObject_CallMethod(_impl->bridge, "page_size", nullptr);
  if (pageSize == nullptr) { throw fail("bridge.page_size"); }
  const long reportedPageSize = PyLong_AsLong(pageSize);
  Py_DECREF(pageSize);
  if (reportedPageSize == -1 && PyErr_Occurred() != nullptr) { throw fail("bridge.page_size returned a non-integer"); }
  _pageSize = static_cast<int>(reportedPageSize);

  // Hand the GIL back.
  //
  // Py_Initialize leaves the GIL with this thread. The engine thread takes it
  // per step, so it is released here once the model is up.
  releaseGil();

  return static_cast<int>(numPages);
}

namespace
{

/// The sampling block of a step payload. Optionals become JSON null, which is
/// the `None` the Python `SamplingParams` carries.
Json samplingJson(const SamplingParams &sampling)
{
  return Json{{"temperature", sampling.temperature},
              {"top_p", sampling.topP},
              {"top_k", sampling.topK.has_value() ? Json(*sampling.topK) : Json(nullptr)},
              {"seed", sampling.seed.has_value() ? Json(*sampling.seed) : Json(nullptr)}};
}

} // namespace

StepResult BridgeExecutor::executeStep(const StepCommand &command)
{
  StepResult result;

  Json payload;
  payload["prefill"] = Json::array();
  payload["decode"]  = Json::array();
  for (const PrefillItem &item : command.prefill)
  {
    payload["prefill"].push_back(Json{{"request_id", item.requestId},
                                      {"tokens", item.chunkTokens},
                                      {"num_computed", item.numComputedTokens},
                                      {"sample_at_length", item.sampleAtLength},
                                      {"block_ids", item.blockIds},
                                      {"sampling", samplingJson(item.sampling)}});
  }
  for (const DecodeItem &item : command.decode)
  {
    payload["decode"].push_back(
      Json{{"request_id", item.requestId}, {"last_token", item.lastToken}, {"seq_len", item.seqLen}, {"block_ids", item.blockIds}, {"sampling", samplingJson(item.sampling)}});
  }
  // Reported once, on the first step after the request ended, so the executor
  // can drop the sampler generator and any device state it was keeping.
  payload["finished"] = command.finished;

  const std::string request = payload.dump();

  // Take the GIL only around the call itself.
  const PyGILState_STATE gil      = PyGILState_Ensure();
  PyObject              *response = PyObject_CallMethod(_impl->bridge, "step", "s", request.c_str());
  if (response == nullptr)
  {
    result.error = takePythonError("bridge.step");
    PyGILState_Release(gil);
    return result;
  }
  // A bridge that returned something other than a str sets a TypeError here.
  // Clearing it before handing the GIL back matters as much as reporting it:
  // an exception left pending surfaces at the next unrelated Python call and
  // is attributed to the wrong request.
  const char *utf8 = PyUnicode_AsUTF8(response);
  std::string body;
  if (utf8 == nullptr) { result.error = takePythonError("bridge.step did not return a string"); }
  else { body = utf8; }
  Py_DECREF(response);
  PyGILState_Release(gil);
  if (!result.error.empty()) { return result; }

  try
  {
    const Json parsed = Json::parse(body);
    if (parsed.contains("error"))
    {
      result.error = parsed.at("error").get<std::string>();
      return result;
    }
    for (const auto &[requestId, tokens] : parsed.at("tokens").items()) { result.newTokens[requestId] = tokens.get<std::vector<int>>(); }
  }
  catch (const std::exception &e)
  {
    result.error = std::string{"bridge returned unparseable JSON: "} + e.what();
  }
  return result;
}

void BridgeExecutor::close()
{
  if (_impl == nullptr || _impl->bridge == nullptr) { return; }
  // PyGILState_Ensure works whether or not the GIL was handed back, so close()
  // is safe both before and after registerModel().
  const PyGILState_STATE gil    = PyGILState_Ensure();
  PyObject              *closed = PyObject_CallMethod(_impl->bridge, "close", nullptr);
  if (closed != nullptr) { Py_DECREF(closed); }
  else { PyErr_Clear(); }
  Py_DECREF(_impl->bridge);
  _impl->bridge = nullptr;
  PyGILState_Release(gil);
}

} // namespace serving::engine
