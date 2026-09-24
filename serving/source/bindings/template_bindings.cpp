#include <serving/bindings/bindings.hpp>

#include <string>

#include <pybind11/stl.h>

#include <serving/model/chat_template.hpp>

namespace py = pybind11;

namespace serving::bindings
{

namespace
{

/// Python object -> ordered_json, through the json module.
nlohmann::ordered_json toJson(const py::handle &value)
{
  if (value.is_none()) { return nlohmann::ordered_json(); }
  const std::string text = py::module_::import("json").attr("dumps")(value).cast<std::string>();
  return nlohmann::ordered_json::parse(text);
}

} // namespace

void bindChatTemplate(py::module_ &m)
{
  py::class_<model::ChatTemplate>(m, "ChatTemplate", "A Jinja chat template rendered by minja, as transformers' apply_chat_template renders it.")
    .def(py::init<const std::string &, const std::string &, const std::string &>(), py::arg("source"), py::arg("bos_token") = "", py::arg("eos_token") = "")
    .def(
      "apply",
      [](const model::ChatTemplate &self, const py::object &messages, bool addGenerationPrompt, const py::object &extraContext, const py::object &tools) {
        return self.apply(toJson(messages), addGenerationPrompt, extraContext.is_none() ? nlohmann::ordered_json::object() : toJson(extraContext), toJson(tools));
      },
      py::arg("messages"),
      py::arg("add_generation_prompt") = true,
      py::arg("extra_context")         = py::none(),
      py::arg("tools")                 = py::none())
    .def_property_readonly("source", &model::ChatTemplate::source);

  m.def("load_chat_template", &model::loadChatTemplate, py::arg("model_dir"), "The chat template from <model_dir>/tokenizer_config.json, or None when there is none.");
}

} // namespace serving::bindings
