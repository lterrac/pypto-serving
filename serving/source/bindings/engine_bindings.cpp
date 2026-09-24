#include <serving/bindings/bindings.hpp>

#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <pybind11/stl.h>

#include <serving/engine/engine.hpp>
#include <serving/engine/executor.hpp>
#include <serving/model/tokenizer.hpp>

namespace py = pybind11;

namespace serving::bindings
{

namespace
{

/// ModelExecutor implementable in Python. An exception in execute_step becomes
/// StepResult.error: the step's requests fail and the engine keeps running.
class PyModelExecutor : public engine::ModelExecutor
{
  public:

  using ModelExecutor::ModelExecutor;

  int registerModel() override { PYBIND11_OVERRIDE_PURE_NAME(int, ModelExecutor, "register_model", registerModel, ); }

  engine::StepResult executeStep(const engine::StepCommand &command) override
  {
    try
    {
      PYBIND11_OVERRIDE_PURE_NAME(engine::StepResult, ModelExecutor, "execute_step", executeStep, command);
    }
    catch (const py::error_already_set &e)
    {
      engine::StepResult failed;
      failed.error = e.what();
      return failed;
    }
  }

  // close() may run from Engine's destructor during interpreter shutdown; an
  // exception out of a destructor would terminate the process.
  void close() override
  {
    try
    {
      PYBIND11_OVERRIDE(void, ModelExecutor, close, );
    }
    catch (const py::error_already_set &e)
    {
      std::fprintf(stderr, "[serving] executor close() raised: %s\n", e.what());
    }
  }
};

void bindExecutorTypes(py::module_ &m)
{
  py::class_<engine::PrefillItem>(m, "PrefillItem", "One request's prompt chunk in a step.")
    .def_readonly("request_id", &engine::PrefillItem::requestId)
    .def_readonly("chunk_tokens", &engine::PrefillItem::chunkTokens)
    .def_readonly("num_computed_tokens", &engine::PrefillItem::numComputedTokens)
    .def_readonly("sample_at_length", &engine::PrefillItem::sampleAtLength)
    .def_readonly("block_ids", &engine::PrefillItem::blockIds);

  py::class_<engine::DecodeItem>(m, "DecodeItem", "One request's decode slot in a step.")
    .def_readonly("request_id", &engine::DecodeItem::requestId)
    .def_readonly("last_token", &engine::DecodeItem::lastToken)
    .def_readonly("seq_len", &engine::DecodeItem::seqLen)
    .def_readonly("block_ids", &engine::DecodeItem::blockIds);

  py::class_<engine::StepCommand>(m, "StepCommand", "What the engine asks the executor to run: prefill chunks and decode slots for one step.")
    .def_readonly("prefill", &engine::StepCommand::prefill)
    .def_readonly("decode", &engine::StepCommand::decode)
    .def_property_readonly("empty", &engine::StepCommand::empty);

  py::class_<engine::StepResult>(m, "StepResult", "Tokens produced per request, or an error that fails every request in the step.")
    .def(py::init([](std::unordered_map<std::string, std::vector<int>> newTokens, std::string error) {
           engine::StepResult r;
           r.newTokens = std::move(newTokens);
           r.error     = std::move(error);
           return r;
         }),
         py::arg("new_tokens") = std::unordered_map<std::string, std::vector<int>>{},
         py::arg("error")      = "")
    .def_readwrite("new_tokens", &engine::StepResult::newTokens)
    .def_readwrite("error", &engine::StepResult::error);

  py::class_<engine::ModelExecutor, PyModelExecutor>(
    m, "ModelExecutor", "The model, as the engine drives it. Subclass in Python: register_model() -> KV pages, execute_step(command) -> StepResult, close().")
    .def(py::init<>())
    .def("register_model", &engine::ModelExecutor::registerModel)
    .def("execute_step", &engine::ModelExecutor::executeStep, py::arg("command"))
    .def("close", &engine::ModelExecutor::close);
}

} // namespace

void bindEngine(py::module_ &m)
{
  bindExecutorTypes(m);

  py::class_<engine::EngineConfig>(m, "EngineConfig")
    .def(py::init<>())
    .def_readwrite("scheduler", &engine::EngineConfig::scheduler)
    .def_readwrite("runtime", &engine::EngineConfig::runtime)
    .def_readwrite("defaults", &engine::EngineConfig::defaults)
    .def_readwrite("idle_poll_microseconds", &engine::EngineConfig::idlePollMicroseconds);

  py::class_<engine::TokenOutput>(m, "TokenOutput", "One streamed update for a request.")
    .def_readonly("request_id", &engine::TokenOutput::requestId)
    .def_readonly("delta", &engine::TokenOutput::delta)
    .def_readonly("text", &engine::TokenOutput::text)
    .def_readonly("token_id", &engine::TokenOutput::tokenId)
    .def_readonly("finished", &engine::TokenOutput::finished)
    .def_readonly("finish_reason", &engine::TokenOutput::finishReason)
    .def_readonly("error", &engine::TokenOutput::error);

  // pop() blocks on the engine thread, which may need the GIL: release it.
  py::class_<engine::RequestStream, std::shared_ptr<engine::RequestStream>>(m, "RequestStream", "A request's updates in order; pop() returns None once finished and drained.")
    .def("pop", &engine::RequestStream::pop, py::call_guard<py::gil_scoped_release>())
    .def_property_readonly("closed", &engine::RequestStream::closed)
    .def(
      "__iter__", [](const std::shared_ptr<engine::RequestStream> &self) { return self; }, py::keep_alive<0, 1>())
    .def("__next__", [](engine::RequestStream &self) {
      std::optional<engine::TokenOutput> next;
      {
        py::gil_scoped_release release;
        next = self.pop();
      }
      if (!next) { throw py::stop_iteration(); }
      return *next;
    });

  // keep_alive ties the tokenizer and executor to the engine. Calls that can
  // wait on the engine thread release the GIL.
  py::class_<engine::Engine, std::unique_ptr<engine::Engine, GilReleasingDelete>>(m, "Engine", "The serving engine: scheduler, KV cache and the step loop, on its own thread.")
    .def(py::init<engine::EngineConfig, const model::TokenizerAdapter &, engine::ModelExecutor &>(),
         py::arg("config"),
         py::arg("tokenizer"),
         py::arg("executor"),
         py::keep_alive<1, 3>(),
         py::keep_alive<1, 4>())
    .def(
      "start", &engine::Engine::start, py::call_guard<py::gil_scoped_release>(), "Load the model through the executor, then start the engine thread. Returns once the loop is live.")
    .def("stop", &engine::Engine::stop, py::call_guard<py::gil_scoped_release>())
    .def_property_readonly("is_ready", &engine::Engine::isReady)
    .def("add_request", &engine::Engine::addRequest, py::arg("request_id"), py::arg("prompt_token_ids"), py::arg("generate_config"), py::call_guard<py::gil_scoped_release>())
    .def("generate_request_id", &engine::Engine::generateRequestId)
    .def_property_readonly("pending_token_load",
                           [](engine::Engine &e) {
                             py::gil_scoped_release release;
                             return e.pendingTokenLoad();
                           })
    .def("abort_request", &engine::Engine::abortRequest, py::arg("request_id"), py::call_guard<py::gil_scoped_release>());
}

} // namespace serving::bindings
