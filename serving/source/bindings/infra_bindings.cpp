#include <serving/bindings/bindings.hpp>

#include <optional>
#include <string>
#include <vector>

#include <pybind11/stl.h>

#include <serving/bindings/repr.hpp>
#include <serving/infra/discovery.hpp>
#include <serving/infra/document.hpp>
#include <serving/infra/capacity.hpp>
#include <serving/infra/launcher.hpp>
#include <serving/infra/monitor.hpp>
#include <serving/replica/rule_table.hpp>
#include <serving/infra/scaling.hpp>
#include <serving/infra/process_starter.hpp>
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
    .value("RESERVED", infra::Phase::Reserved, "up, loaded and healthy, deliberately not routed to: a warm spare")
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
    // Copies, not references into the map. These return pointers into
    // std::vector members that add_replica/remove_replica reallocate, and
    // reference_internal only keeps the StateMap alive -- it cannot keep the
    // vector's buffer alive, so a handle held across one add_replica reads and
    // writes freed memory. The KV pool solves the same problem by never
    // resizing; the map resizes by design, so Python gets a copy.
    .def(
      "node",
      [](const infra::StateMap &map, const std::string &name) -> std::optional<infra::Node> {
        const infra::Node *found = map.node(name);
        return found == nullptr ? std::nullopt : std::optional<infra::Node>{*found};
      },
      py::arg("name"))
    .def(
      "replica",
      [](const infra::StateMap &map, const std::string &name) -> std::optional<infra::Replica> {
        const infra::Replica *found = map.replica(name);
        return found == nullptr ? std::nullopt : std::optional<infra::Replica>{*found};
      },
      py::arg("name"))
    .def(
      "replicas_on",
      [](const infra::StateMap &map, const std::string &node) {
        std::vector<infra::Replica> out;
        for (const infra::Replica *replica : map.replicasOn(node)) { out.push_back(*replica); }
        return out;
      },
      py::arg("node"))
    .def(
      "replicas_in",
      [](const infra::StateMap &map, infra::PartitionId partition) {
        std::vector<infra::Replica> out;
        for (const infra::Replica *replica : map.replicasIn(partition)) { out.push_back(*replica); }
        return out;
      },
      py::arg("partition"))
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

  py::register_exception<infra::DocumentError>(m, "DocumentError", PyExc_ValueError);

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

  py::enum_<infra::Readiness>(m, "Readiness", "Where a started replica is between launched and serving.")
    .value("LOADING", infra::Readiness::Loading, "not serving yet, and might be: keep waiting")
    .value("READY", infra::Readiness::Ready)
    .value("FAILED", infra::Readiness::Failed, "will not become ready; the attempt has already failed");

  py::class_<infra::Backoff>(m, "Backoff", "Exponential backoff with a ceiling. The first attempt is immediate.")
    .def(py::init<double, double, int>(), py::arg("base_seconds") = 1.0, py::arg("max_seconds") = 60.0, py::arg("max_attempts") = 5)
    .def("delay_after", &infra::Backoff::delayAfter, py::arg("attempts"))
    .def("exhausted", &infra::Backoff::exhausted, py::arg("attempts"))
    .def_property_readonly("max_attempts", &infra::Backoff::maxAttempts);

  py::class_<infra::ReplicaStarter>(m, "ReplicaStarter", "How a replica is actually started.");
  py::class_<infra::HealthProbe>(m, "HealthProbe", "Whether a started replica is serving yet.");

  py::class_<infra::ProcessStarter, infra::ReplicaStarter>(m, "ProcessStarter", "Launches a command per replica. {name} {node} {device} {devices} {port} are substituted.")
    .def(py::init<std::string>(), py::arg("command_template"))
    .def("command_for", &infra::ProcessStarter::commandFor, py::arg("replica"), "What this replica would be started with, so a deployment can print it first.")
    .def("running", &infra::ProcessStarter::running, py::arg("replica"))
    .def("start", &infra::ProcessStarter::start, py::arg("replica"))
    .def("stop", &infra::ProcessStarter::stop, py::arg("replica"));

  py::class_<infra::HttpHealthProbe, infra::HealthProbe>(m, "HttpHealthProbe", "Asks the replica's own /health, which is 503 until its engine loop is live.")
    .def(py::init([](const infra::StateMap *map, double timeout) { return std::make_unique<infra::HttpHealthProbe>(map, timeout); }),
         py::arg("map")                     = nullptr,
         py::arg("connect_timeout_seconds") = 2.0,
         py::keep_alive<1, 2>())
    .def("address_of", &infra::HttpHealthProbe::addressOf, py::arg("replica"), "Where this replica would be probed.")
    .def("check", &infra::HttpHealthProbe::check, py::arg("replica"));

  py::class_<infra::LiveAndReadyProbe, infra::HealthProbe>(m, "LiveAndReadyProbe", "Health plus liveness: a replica whose process is gone is Failed, not Loading.")
    .def(py::init<infra::ProcessStarter &, infra::HealthProbe &>(), py::arg("starter"), py::arg("health"), py::keep_alive<1, 2>(), py::keep_alive<1, 3>())
    .def("check", &infra::LiveAndReadyProbe::check, py::arg("replica"));

  py::class_<infra::SupervisionReport>(m, "SupervisionReport")
    .def_readonly("started", &infra::SupervisionReport::started)
    .def_readonly("promoted", &infra::SupervisionReport::promoted)
    .def_readonly("retrying", &infra::SupervisionReport::retrying)
    .def_readonly("gave_up", &infra::SupervisionReport::gaveUp)
    .def_readonly("stopped", &infra::SupervisionReport::stopped)
    .def_readonly("waiting", &infra::SupervisionReport::waiting)
    .def_property_readonly("empty", &infra::SupervisionReport::empty)
    .def("__repr__", [](const infra::SupervisionReport &r) {
      auto names = [](const std::vector<std::string> &v) {
        std::string out;
        for (const auto &n : v) { out += (out.empty() ? "" : ",") + n; }
        return out.empty() ? "-" : out;
      };
      return "SupervisionReport(started=" + names(r.started) + ", promoted=" + names(r.promoted) + ", retrying=" + names(r.retrying) + ", gave_up=" + names(r.gaveUp) +
             ", stopped=" + names(r.stopped) + ", waiting=" + std::to_string(r.waiting) + ")";
    });

  py::class_<infra::Supervisor>(m, "Supervisor", "Drives the map towards what it says it wants. One service() per tick.")
    .def(py::init([](infra::StateMap &map, infra::ReplicaStarter &starter, infra::HealthProbe &probe, infra::Backoff backoff) {
           return std::make_unique<infra::Supervisor>(map, starter, probe, backoff);
         }),
         py::arg("map"),
         py::arg("starter"),
         py::arg("probe"),
         py::arg("backoff") = infra::Backoff{},
         py::keep_alive<1, 2>(),
         py::keep_alive<1, 3>(),
         py::keep_alive<1, 4>())
    .def("set_ready_timeout",
         &infra::Supervisor::setReadyTimeout,
         py::arg("seconds"),
         "How long a replica may take to answer its health probe. Longer than a model load, or every launch gives up.")
    // The engine's model load holds the GIL through a Python executor; a tick
    // that blocked on a health probe without releasing it would deadlock.
    .def("service", &infra::Supervisor::service, py::call_guard<py::gil_scoped_release>())
    .def("attempts", &infra::Supervisor::attempts, py::arg("replica"));

  py::class_<infra::MonitorConfig>(m, "MonitorConfig")
    .def(py::init([](int failures) { return infra::MonitorConfig{failures}; }), py::arg("failures_before_dead") = 3)
    .def_readwrite("failures_before_dead", &infra::MonitorConfig::failuresBeforeDead)
    .def("validate", &infra::MonitorConfig::validate);

  py::class_<infra::MonitorReport>(m, "MonitorReport")
    .def_readonly("lost", &infra::MonitorReport::lost)
    .def_readonly("recovered", &infra::MonitorReport::recovered)
    .def_readonly("probed", &infra::MonitorReport::probed)
    .def_property_readonly("empty", &infra::MonitorReport::empty)
    .def("__repr__", [](const infra::MonitorReport &r) {
      auto names = [](const std::vector<std::string> &v) {
        std::string out;
        for (const auto &n : v) { out += (out.empty() ? "" : ",") + n; }
        return out.empty() ? "-" : out;
      };
      return "MonitorReport(lost=" + names(r.lost) + ", recovered=" + names(r.recovered) + ", probed=" + std::to_string(r.probed) + ")";
    });

  py::class_<infra::HealthMonitor>(m, "HealthMonitor", "Notices that a replica has gone, and writes it in the map.")
    .def(py::init([](infra::StateMap &map, infra::HealthProbe &probe, infra::MonitorConfig config) { return std::make_unique<infra::HealthMonitor>(map, probe, config); }),
         py::arg("map"),
         py::arg("probe"),
         py::arg("config") = infra::MonitorConfig{},
         py::keep_alive<1, 2>(),
         py::keep_alive<1, 3>())
    // Probing blocks on the network; a tick holding the GIL would stall a
    // Python executor loading its model on the same process.
    .def("service", &infra::HealthMonitor::service, py::call_guard<py::gil_scoped_release>())
    .def("failures", &infra::HealthMonitor::failures, py::arg("replica"));

  py::class_<replica::NextHop>(m, "NextHop", "Where a session goes after this replica.")
    .def_readonly("name", &replica::NextHop::name)
    .def_readonly("instance", &replica::NextHop::instance)
    .def_readonly("generation", &replica::NextHop::generation)
    .def_readonly("endpoint", &replica::NextHop::endpoint);

  py::class_<replica::RuleTable>(m, "RuleTable", "A replica's forwarding rules, newest-used kept.")
    .def(py::init<size_t>(), py::arg("max_sessions") = 65536)
    .def("lookup", &replica::RuleTable::lookup, py::arg("session_id"))
    .def("peek", &replica::RuleTable::peek, py::arg("session_id"))
    .def_property_readonly("size", &replica::RuleTable::size)
    .def_property_readonly("evicted", &replica::RuleTable::evicted)
    .def("clear", &replica::RuleTable::clear);

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
