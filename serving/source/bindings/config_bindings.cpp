#include <serving/bindings/bindings.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <serving/bindings/repr.hpp>
#include <serving/config/types.hpp>

namespace py = pybind11;

namespace serving::bindings
{

void bindConfigTypes(py::module_ &m)
{
  using namespace serving::config;

  py::class_<GenerateConfig>(m, "GenerateConfig", "User-facing options that control text generation.")
    .def(py::init(
           [](int maxNewTokens, double temperature, double topP, std::optional<int> topK, std::optional<uint64_t> seed, std::vector<std::string> stop, bool stream, bool ignoreEos) {
             return GenerateConfig{maxNewTokens, temperature, topP, topK, seed, std::move(stop), stream, ignoreEos};
           }),
         py::arg("max_new_tokens") = 256,
         py::arg("temperature")    = 0.0,
         py::arg("top_p")          = 1.0,
         py::arg("top_k")          = py::none(),
         py::arg("seed")           = py::none(),
         py::arg("stop")           = std::vector<std::string>{},
         py::arg("stream")         = false,
         py::arg("ignore_eos")     = false)
    .def_readonly("max_new_tokens", &GenerateConfig::maxNewTokens)
    .def_readonly("temperature", &GenerateConfig::temperature)
    .def_readonly("top_p", &GenerateConfig::topP)
    .def_readonly("top_k", &GenerateConfig::topK)
    .def_readonly("seed", &GenerateConfig::seed)
    .def_readonly("stop", &GenerateConfig::stop)
    .def_readonly("stream", &GenerateConfig::stream)
    .def_readonly("ignore_eos", &GenerateConfig::ignoreEos)
    .def(py::self == py::self)
    .def("__repr__", [](const GenerateConfig &c) {
      return Repr("GenerateConfig")
        .field("max_new_tokens", c.maxNewTokens)
        .field("temperature", c.temperature)
        .field("top_p", c.topP)
        .field("top_k", c.topK)
        .field("seed", c.seed)
        .field("stop", c.stop)
        .field("stream", c.stream)
        .field("ignore_eos", c.ignoreEos)
        .str();
    });

  py::class_<RuntimeConfig>(m, "RuntimeConfig", "The KV pool's geometry.")
    .def(py::init([](int pageSize, int maxBatchSize, int maxSeqLen) {
           return RuntimeConfig{pageSize, maxBatchSize, maxSeqLen};
         }),
         py::arg("page_size")      = 64,
         py::arg("max_batch_size") = 1,
         py::arg("max_seq_len")    = 4096)
    .def_readonly("page_size", &RuntimeConfig::pageSize)
    .def_readonly("max_batch_size", &RuntimeConfig::maxBatchSize)
    .def_readonly("max_seq_len", &RuntimeConfig::maxSeqLen)
    .def(py::self == py::self)
    .def("__repr__", [](const RuntimeConfig &r) {
      return Repr("RuntimeConfig").field("page_size", r.pageSize).field("max_batch_size", r.maxBatchSize).field("max_seq_len", r.maxSeqLen).str();
    });
}

} // namespace serving::bindings
