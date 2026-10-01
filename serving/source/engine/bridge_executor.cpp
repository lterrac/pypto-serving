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
  PyObject *pages = PyObject_CallMethod(
    _impl->bridge, "open_model", "sisiii", _config.modelDir.c_str(), _config.deviceId, _config.platform.c_str(), _config.maxModelLen, _config.blockSize, _config.maxNumSeqs);
  if (pages == nullptr) { throw std::runtime_error(takePythonError("bridge.open_model")); }
  const long numPages = PyLong_AsLong(pages);
  Py_DECREF(pages);

  PyObject *pageSize = PyObject_CallMethod(_impl->bridge, "page_size", nullptr);
  if (pageSize == nullptr) { throw std::runtime_error(takePythonError("bridge.page_size")); }
  _pageSize = static_cast<int>(PyLong_AsLong(pageSize));
  Py_DECREF(pageSize);

  // Hand the GIL back.
  //
  // Py_Initialize leaves the GIL with this thread. The engine thread takes it
  // per step, so it is released here once the model is up.
  if (_impl->mainState == nullptr) { _impl->mainState = PyEval_SaveThread(); }

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
  const char *utf8 = PyUnicode_AsUTF8(response);
  std::string body = utf8 == nullptr ? std::string{} : std::string{utf8};
  Py_DECREF(response);
  PyGILState_Release(gil);

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
