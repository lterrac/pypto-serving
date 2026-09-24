#include <serving/bindings/bindings.hpp>

#include <string>

#include <pybind11/stl.h>

#include <serving/engine/engine.hpp>
#include <serving/model/chat_template.hpp>
#include <serving/model/tokenizer.hpp>
#include <serving/server/http_server.hpp>

namespace py = pybind11;

namespace serving::bindings
{

void bindServer(py::module_ &m)
{
  py::class_<server::ServerConfig>(m, "ServerConfig")
    .def(py::init([](std::string host, int port, std::string modelId) {
           server::ServerConfig c;
           c.host    = std::move(host);
           c.port    = port;
           c.modelId = std::move(modelId);
           return c;
         }),
         py::arg("host")     = "0.0.0.0",
         py::arg("port")     = 8000,
         py::arg("model_id") = "model")
    .def_readwrite("host", &server::ServerConfig::host)
    .def_readwrite("port", &server::ServerConfig::port)
    .def_readwrite("model_id", &server::ServerConfig::modelId);

  py::class_<server::HttpServer>(m, "HttpServer", "OpenAI-compatible HTTP front end over an Engine, on its own threads.")
    .def(py::init<server::ServerConfig, engine::Engine &, const model::TokenizerAdapter &, const model::ChatTemplate *>(),
         py::arg("config"),
         py::arg("engine"),
         py::arg("tokenizer"),
         py::arg("chat_template") = nullptr,
         py::keep_alive<1, 3>(),
         py::keep_alive<1, 4>(),
         py::keep_alive<1, 5>())
    .def("start", &server::HttpServer::start, py::call_guard<py::gil_scoped_release>())
    .def("stop", &server::HttpServer::stop, py::call_guard<py::gil_scoped_release>())
    .def_property_readonly("bound_port", &server::HttpServer::boundPort);
}

} // namespace serving::bindings
