#include <serving/bindings/bindings.hpp>

#include <optional>
#include <string>
#include <vector>

#include <pybind11/stl.h>

#include <serving/model/tokenizer.hpp>
#if SERVING_HAVE_TOKENIZERS
  #include <serving/model/hf_tokenizer.hpp>
#endif

namespace py = pybind11;

namespace serving::bindings
{

namespace
{

/// TokenizerAdapter implementable in Python; pybind11 takes the GIL per call.
class PyTokenizerAdapter : public model::TokenizerAdapter
{
  public:

  using TokenizerAdapter::TokenizerAdapter;

  std::vector<int> encode(const std::string &text) const override { PYBIND11_OVERRIDE_PURE(std::vector<int>, TokenizerAdapter, encode, text); }

  std::string decode(const std::vector<int> &tokenIds, bool skipSpecialTokens) const override
  {
    PYBIND11_OVERRIDE_PURE(std::string, TokenizerAdapter, decode, tokenIds, skipSpecialTokens);
  }

  std::optional<int> bosTokenId() const override { PYBIND11_OVERRIDE_NAME(std::optional<int>, TokenizerAdapter, "bos_token_id", bosTokenId, ); }
  std::optional<int> eosTokenId() const override { PYBIND11_OVERRIDE_NAME(std::optional<int>, TokenizerAdapter, "eos_token_id", eosTokenId, ); }
};

} // namespace

void bindTokenizer(py::module_ &m)
{
  py::class_<model::TokenizerAdapter, PyTokenizerAdapter>(
    m, "TokenizerAdapter", "Text <-> token ids. Subclass in Python: encode(text), decode(token_ids, skip_special_tokens), and optionally bos_token_id() / eos_token_id().")
    .def(py::init<>())
    .def("encode", &model::TokenizerAdapter::encode, py::arg("text"))
    .def("decode", &model::TokenizerAdapter::decode, py::arg("token_ids"), py::arg("skip_special_tokens") = true)
    .def("bos_token_id", &model::TokenizerAdapter::bosTokenId)
    .def("eos_token_id", &model::TokenizerAdapter::eosTokenId);

#if SERVING_HAVE_TOKENIZERS
  py::class_<model::HfTokenizer, model::TokenizerAdapter>(m, "HfTokenizer", "tokenizers-cpp over a model directory's tokenizer.json.")
    .def_static("from_model_dir", &model::HfTokenizer::fromModelDir, py::arg("model_dir"));
  m.attr("HAS_TOKENIZERS") = true;
#else
  m.attr("HAS_TOKENIZERS") = false;
#endif
}

} // namespace serving::bindings
