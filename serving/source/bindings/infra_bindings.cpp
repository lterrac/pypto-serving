#include <serving/bindings/bindings.hpp>

#include <string>
#include <vector>

#include <pybind11/stl.h>

#include <serving/bindings/repr.hpp>
#include <serving/infra/discovery.hpp>
#include <serving/infra/state.hpp>

namespace py = pybind11;

namespace serving::bindings
{

namespace
{

std::string show(infra::Phase phase) { return std::string("'") + infra::phaseName(phase) + "'"; }

} // namespace

void bindInfra(py::module_ &m)
{
  py::enum_<infra::Phase>(m, "Phase", "Where a replica is in its life.")
    .value("PENDING", infra::Phase::Pending, "wanted by the deployment, not started")
    .value("ACTIVE", infra::Phase::Active)
    .value("DRAINING", infra::Phase::Draining, "finishing what it has; still holds its devices")
    .value("DEAD", infra::Phase::Dead, "lost rather than drained; its devices are free again");

  py::enum_<infra::Action>(m, "Action", "What would close the gap between a replica's phase and the one wanted.")
    .value("START", infra::Action::Start)
    .value("DRAIN", infra::Action::Drain)
    .value("STOP", infra::Action::Stop);

  py::class_<infra::Node>(m, "Node", "A machine and the devices it owns.")
    .def(py::init([](std::string name, std::string host, std::vector<int> devices) {
           return infra::Node{std::move(name), std::move(host), std::move(devices)};
         }),
         py::arg("name"),
         py::arg("host")    = "",
         py::arg("devices") = std::vector<int>{})
    .def_readwrite("name", &infra::Node::name)
    .def_readwrite("host", &infra::Node::host)
    .def_readwrite("devices", &infra::Node::devices)
    .def("owns", &infra::Node::owns, py::arg("device"))
    .def("__repr__", [](const infra::Node &n) { return Repr("Node").field("name", n.name).field("host", n.host).field("devices", n.devices).str(); });

  py::class_<infra::Replica>(m, "Replica", "One replica, as the deployment describes it and as it is observed.")
    .def(py::init([](std::string name, int partition, std::string node, std::vector<int> devices, int port, std::string scheme, infra::Phase phase, infra::Phase desiredPhase) {
           infra::Replica r;
           r.name         = std::move(name);
           r.partition    = partition;
           r.node         = std::move(node);
           r.devices      = std::move(devices);
           r.port         = port;
           r.scheme       = std::move(scheme);
           r.phase        = phase;
           r.desiredPhase = desiredPhase;
           return r;
         }),
         py::arg("name"),
         py::arg("partition")     = 0,
         py::arg("node")          = "",
         py::arg("devices")       = std::vector<int>{},
         py::arg("port")          = 0,
         py::arg("scheme")        = "http",
         py::arg("phase")         = infra::Phase::Pending,
         py::arg("desired_phase") = infra::Phase::Active)
    .def_readwrite("name", &infra::Replica::name)
    .def_readwrite("partition", &infra::Replica::partition)
    .def_readwrite("node", &infra::Replica::node)
    .def_readwrite("devices", &infra::Replica::devices)
    .def_readwrite("port", &infra::Replica::port)
    .def_readwrite("scheme", &infra::Replica::scheme)
    .def_readwrite("instance", &infra::Replica::instance)
    .def_readwrite("phase", &infra::Replica::phase)
    .def_readwrite("desired_phase", &infra::Replica::desiredPhase)
    .def("__repr__", [](const infra::Replica &r) {
      return Repr("Replica").field("name", r.name).field("partition", r.partition).field("node", r.node).field("devices", r.devices).field("port", r.port).str();
    });

  py::class_<infra::Channel>(m, "Channel", "A control-plane channel between two replicas.")
    .def(py::init([](std::string from, std::string to) {
           return infra::Channel{std::move(from), std::move(to)};
         }),
         py::arg("from_replica"),
         py::arg("to_replica"))
    .def_readwrite("from_replica", &infra::Channel::from)
    .def_readwrite("to_replica", &infra::Channel::to);

  py::class_<infra::PendingAction>(m, "PendingAction")
    .def_readonly("replica", &infra::PendingAction::replica)
    .def_readonly("action", &infra::PendingAction::action)
    .def("__repr__", [](const infra::PendingAction &a) { return std::string("PendingAction(replica='") + a.replica + "', action='" + infra::actionName(a.action) + "')"; });

  py::class_<infra::StateMap>(m, "StateMap", "What runs where, holding which devices, in which state.")
    .def(py::init<>())
    .def_property_readonly("version", &infra::StateMap::version, "Moves on every edit, so a cached decision can tell the ground shifted.")
    .def_property_readonly("nodes", &infra::StateMap::nodes)
    .def_property_readonly("replicas", &infra::StateMap::replicas)
    .def_property_readonly("channels", &infra::StateMap::channels)
    .def("node", &infra::StateMap::node, py::arg("name"), py::return_value_policy::reference_internal)
    .def("replica", &infra::StateMap::replica, py::arg("name"), py::return_value_policy::reference_internal)
    .def("replicas_on", &infra::StateMap::replicasOn, py::arg("node"), py::return_value_policy::reference_internal)
    .def("replicas_in", &infra::StateMap::replicasIn, py::arg("partition"), py::return_value_policy::reference_internal)
    .def("partitions", &infra::StateMap::partitions)
    .def("devices_in_use", &infra::StateMap::devicesInUse, py::arg("node"))
    .def("devices_free", &infra::StateMap::devicesFree, py::arg("node"))
    .def("add_node", &infra::StateMap::addNode, py::arg("node"))
    .def("add_replica", &infra::StateMap::addReplica, py::arg("replica"))
    .def("add_channel", &infra::StateMap::addChannel, py::arg("channel"))
    .def("set_phase", &infra::StateMap::setPhase, py::arg("name"), py::arg("phase"))
    .def("set_desired_phase", &infra::StateMap::setDesiredPhase, py::arg("name"), py::arg("phase"))
    .def("place", &infra::StateMap::place, py::arg("name"), py::arg("node"), py::arg("devices"))
    .def("remove_replica", &infra::StateMap::removeReplica, py::arg("name"))
    .def("__repr__", [](const infra::StateMap &map) {
      return "StateMap(version=" + std::to_string(map.version()) + ", nodes=" + std::to_string(map.nodes().size()) + ", replicas=" + std::to_string(map.replicas().size()) + ")";
    });

  m.def("validate", &infra::validate, py::arg("map"), "Everything wrong with the deployment, one message per problem. Empty means coherent.");
  m.def("reconcile", &infra::reconcile, py::arg("map"), "The actions that would bring every replica to the phase it is wanted in.");
  m.def("discover_devices", &infra::discoverDevices, py::arg("device_dir") = "/dev", "Physical device ids this machine exposes.");
  m.def(
    "local_node", &infra::localNode, py::arg("name"), py::arg("host") = "", py::arg("device_dir") = "/dev", "This machine as a map entry, with the devices it was found to own.");
}

} // namespace serving::bindings
