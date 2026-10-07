#include <serving/bindings/bindings.hpp>

#include <string>
#include <vector>

#include <pybind11/stl.h>

#include <serving/bindings/repr.hpp>
#include <serving/infra/discovery.hpp>
#include <serving/infra/capacity.hpp>
#include <serving/infra/scaling.hpp>
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

  py::class_<infra::CapacityRequest>(m, "CapacityRequest", "What a scaling decision asks for.")
    .def(py::init([](infra::PartitionId partition, int devices, std::string name, int port) {
           return infra::CapacityRequest{partition, devices, std::move(name), port};
         }),
         py::arg("partition") = 0,
         py::arg("devices")   = 1,
         py::arg("name")      = "",
         py::arg("port")      = 0)
    .def_readwrite("partition", &infra::CapacityRequest::partition)
    .def_readwrite("devices", &infra::CapacityRequest::devices)
    .def_readwrite("name", &infra::CapacityRequest::name)
    .def_readwrite("port", &infra::CapacityRequest::port);

  py::class_<infra::Grant>(m, "Grant", "Where a request landed.")
    .def_readonly("replica", &infra::Grant::replica)
    .def_readonly("node", &infra::Grant::node)
    .def_readonly("devices", &infra::Grant::devices)
    .def_readonly("port", &infra::Grant::port)
    .def_readonly("instance", &infra::Grant::instance)
    .def("__repr__", [](const infra::Grant &g) { return Repr("Grant").field("replica", g.replica).field("node", g.node).field("devices", g.devices).str(); });

  py::class_<infra::CapacityProvider>(m, "CapacityProvider", "Somewhere capacity comes from.")
    .def("acquire", &infra::CapacityProvider::acquire, py::arg("request"))
    .def("release", &infra::CapacityProvider::release, py::arg("replica"))
    .def("available", &infra::CapacityProvider::available, py::arg("request"))
    .def_property_readonly("kind", &infra::CapacityProvider::kind);

  py::class_<infra::LocalCapacity, infra::CapacityProvider>(m, "LocalCapacity", "Capacity on nodes the map already knows about.")
    .def(py::init<const infra::StateMap &>(), py::arg("map"), py::keep_alive<1, 2>());

  py::class_<infra::ReplicaLoad>(m, "ReplicaLoad", "What one replica is carrying, sampled now.")
    .def(py::init([](std::string replica, int outstanding, int pendingTokens, int preemptions) {
           return infra::ReplicaLoad{std::move(replica), outstanding, pendingTokens, preemptions};
         }),
         py::arg("replica"),
         py::arg("outstanding")    = 0,
         py::arg("pending_tokens") = 0,
         py::arg("preemptions")    = 0)
    .def_readwrite("replica", &infra::ReplicaLoad::replica)
    .def_readwrite("outstanding", &infra::ReplicaLoad::outstanding)
    .def_readwrite("pending_tokens", &infra::ReplicaLoad::pendingTokens)
    .def_readwrite("preemptions", &infra::ReplicaLoad::preemptions);

  py::class_<infra::LoadSnapshot>(m, "LoadSnapshot").def(py::init<>()).def_readwrite("replicas", &infra::LoadSnapshot::replicas);

  py::class_<infra::ScalingConfig>(m, "ScalingConfig", "When to add or drain a replica. Serving's decision, not the control plane's.")
    .def(py::init<>())
    .def_readwrite("scale_up_above", &infra::ScalingConfig::scaleUpAbove)
    .def_readwrite("scale_down_below", &infra::ScalingConfig::scaleDownBelow)
    .def_readwrite("min_replicas", &infra::ScalingConfig::minReplicas)
    .def_readwrite("max_replicas", &infra::ScalingConfig::maxReplicas)
    .def_readwrite("preemptions_above", &infra::ScalingConfig::preemptionsAbove)
    .def_readwrite("devices_per_replica", &infra::ScalingConfig::devicesPerReplica)
    .def("validate", &infra::ScalingConfig::validate);

  py::enum_<infra::ScalingAction>(m, "ScalingAction").value("SCALE_UP", infra::ScalingAction::ScaleUp).value("SCALE_DOWN", infra::ScalingAction::ScaleDown);

  py::class_<infra::ScalingDecision>(m, "ScalingDecision")
    .def_readonly("partition", &infra::ScalingDecision::partition)
    .def_readonly("action", &infra::ScalingDecision::action)
    .def_readonly("replica", &infra::ScalingDecision::replica)
    .def_readonly("reason", &infra::ScalingDecision::reason);

  py::class_<infra::ScalingOutcome>(m, "ScalingOutcome")
    .def_readonly("decision", &infra::ScalingOutcome::decision)
    .def_readonly("applied", &infra::ScalingOutcome::applied)
    .def_readonly("replica", &infra::ScalingOutcome::replica)
    .def_readonly("declined", &infra::ScalingOutcome::declined);

  m.def("decide_scaling", &infra::decideScaling, py::arg("map"), py::arg("load"), py::arg("config"), "What the policy thinks, per partition. Reads only.");
  m.def("apply_scaling", &infra::applyScaling, py::arg("map"), py::arg("provider"), py::arg("decisions"), py::arg("config"));
  m.def("scaling_action_name", &infra::scalingActionName, py::arg("action"));

  m.def("write_document", &infra::writeDocument, py::arg("map"), "The map as the deployment document.");
  m.def("read_document", &infra::readDocument, py::arg("text"), py::arg("source") = "<document>", "Parse one. Strict: an unknown key is an error.");
  m.def("load_document", [](const std::string &path) { return infra::loadDocument(path); }, py::arg("path"));
  m.def(
    "save_document",
    [](const infra::StateMap &map, const std::string &path) { infra::saveDocument(map, path); },
    py::arg("map"),
    py::arg("path"),
    "Write it where a reader only ever sees it whole: a sibling temporary, renamed over the target.");

  m.def("validate", &infra::validate, py::arg("map"), "Everything wrong with the deployment, one message per problem. Empty means coherent.");
  m.def("reconcile", &infra::reconcile, py::arg("map"), "The actions that would bring every replica to the phase it is wanted in.");
  m.def("discover_devices", &infra::discoverDevices, py::arg("device_dir") = "/dev", "Physical device ids this machine exposes.");
  m.def(
    "local_node", &infra::localNode, py::arg("name"), py::arg("host") = "", py::arg("device_dir") = "/dev", "This machine as a map entry, with the devices it was found to own.");
}

} // namespace serving::bindings
