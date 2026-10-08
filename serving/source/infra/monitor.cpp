#include <serving/infra/monitor.hpp>

#include <stdexcept>

namespace serving::infra
{

void MonitorConfig::validate() const
{
  if (failuresBeforeDead < 1) { throw std::invalid_argument("failures_before_dead must be at least 1"); }
}

HealthMonitor::HealthMonitor(StateMap &map, HealthProbe &probe, MonitorConfig config)
  : _map(map),
    _probe(probe),
    _config(config)
{
  _config.validate();
}

int HealthMonitor::failures(const std::string &replica) const
{
  const auto it = _failures.find(replica);
  return it == _failures.end() ? 0 : it->second;
}

MonitorReport HealthMonitor::service()
{
  MonitorReport report;

  // Copied, because marking one dead edits the map and would invalidate an
  // iterator over it.
  std::vector<std::string> serving;
  for (const Replica &replica : _map.replicas())
  {
    if (replica.phase == Phase::Active) { serving.push_back(replica.name); }
  }

  for (const std::string &name : serving)
  {
    const Replica *replica = _map.replica(name);
    if (replica == nullptr) { continue; }

    report.probed += 1;
    const Readiness readiness = _probe.check(*replica);

    if (readiness == Readiness::Ready)
    {
      // Consecutive, so a replica has to fail every probe in a row to be
      // declared gone; one blip in the middle resets it.
      if (_failures.erase(name) > 0) { report.recovered.push_back(name); }
      continue;
    }

    const int failures = ++_failures[name];
    if (failures < _config.failuresBeforeDead) { continue; }

    // Dead rather than draining: draining is an orderly exit that keeps its
    // devices, and this replica is not exiting, it is gone. Dead releases the
    // devices and runs the loss path in routing.
    _map.setPhase(name, Phase::Dead);
    _failures.erase(name);
    report.lost.push_back(name);
  }

  // A replica that left for any other reason stops being watched.
  for (auto it = _failures.begin(); it != _failures.end();)
  {
    const Replica *replica = _map.replica(it->first);
    if (replica == nullptr || replica->phase != Phase::Active) { it = _failures.erase(it); }
    else { ++it; }
  }

  return report;
}

} // namespace serving::infra
