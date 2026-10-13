#include <serving/router/http_rule_transport.hpp>

#include <utility>

#include <httplib.h>
#include <nlohmann/json.hpp>

namespace serving::router
{

using nlohmann::json;

HttpRuleTransport::HttpRuleTransport(std::vector<ReplicaSpec> replicas, HttpRuleTransportConfig config)
  : _config(config)
{
  setReplicas(std::move(replicas));
}

void HttpRuleTransport::setReplicas(std::vector<ReplicaSpec> replicas)
{
  const std::lock_guard<std::mutex> lock(_mutex);

  _replicas.clear();
  for (auto &replica : replicas) { _replicas.emplace(replica.name, std::move(replica)); }

  // A replica that left the deployment is not an unreachable one; keeping it
  // marked would silence a name the deployment may later reuse.
  for (auto it = _unreachable.begin(); it != _unreachable.end();) { it = _replicas.count(*it) == 0 ? _unreachable.erase(it) : std::next(it); }
}

void HttpRuleTransport::onReplicaUnreachable(const std::string &replicaName)
{
  const std::lock_guard<std::mutex> lock(_mutex);
  _unreachable.insert(replicaName);
}

void HttpRuleTransport::markReachable(const std::string &replicaName)
{
  const std::lock_guard<std::mutex> lock(_mutex);
  _unreachable.erase(replicaName);
}

bool HttpRuleTransport::post(const ReplicaSpec &target, const std::string &path, const std::string &body)
{
  httplib::Client client(target.host, target.port);
  client.set_connection_timeout(static_cast<time_t>(_config.connectTimeoutSeconds), 0);
  client.set_read_timeout(static_cast<time_t>(_config.requestTimeoutSeconds), 0);
  client.set_write_timeout(static_cast<time_t>(_config.requestTimeoutSeconds), 0);

  const auto response = client.Post(path, body, "application/json");
  // A 409 is a stale generation, which is the replica doing its job: the rule
  // did reach it. Only a transport failure or a refusal counts as failed.
  return response != nullptr && response->status < 500;
}

void HttpRuleTransport::publish(const ReplicaSpec &holder, const std::string &sessionId, const ReplicaSpec &nextHop, uint64_t generation)
{
  {
    const std::lock_guard<std::mutex> lock(_mutex);
    if (_unreachable.count(holder.name) != 0) { return; }
  }

  const json body{{"session_id", sessionId}, {"next_hop", nextHop.name}, {"next_hop_url", nextHop.baseUrl()}, {"generation", generation}};

  const bool                        delivered = post(holder, "/control/rules", body.dump());
  const std::lock_guard<std::mutex> lock(_mutex);
  if (delivered) { _published += 1; }
  else { _failed += 1; }
}

void HttpRuleTransport::revoke(const std::string &sessionId, uint64_t generation)
{
  const json body{{"session_id", sessionId}, {"generation", generation}};

  // Snapshot rather than hold the lock across the posts: a revoke goes to every
  // replica, and holding the lock would serialise routing behind the network.
  std::vector<ReplicaSpec> targets;
  {
    const std::lock_guard<std::mutex> lock(_mutex);
    for (const auto &[name, replica] : _replicas)
    {
      if (_unreachable.count(name) == 0) { targets.push_back(replica); }
    }
  }

  int delivered = 0;
  int lost      = 0;
  for (const auto &target : targets) { (post(target, "/control/rules/revoke", body.dump()) ? delivered : lost) += 1; }

  const std::lock_guard<std::mutex> lock(_mutex);
  _revoked += delivered;
  _failed += lost;
}

int HttpRuleTransport::published() const
{
  const std::lock_guard<std::mutex> lock(_mutex);
  return _published;
}

int HttpRuleTransport::revoked() const
{
  const std::lock_guard<std::mutex> lock(_mutex);
  return _revoked;
}

int HttpRuleTransport::failed() const
{
  const std::lock_guard<std::mutex> lock(_mutex);
  return _failed;
}

} // namespace serving::router
