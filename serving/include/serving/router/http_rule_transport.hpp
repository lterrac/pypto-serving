#pragma once

/**
 * Publishing Option B's forwarding rules over HTTP.
 *
 * `coordinator/channel_transport.hpp` is the transport a platform deployment
 * uses; this is the one a deployment of plain HTTP replicas uses, and it is
 * what makes the coordinator usable before the channel fabric exists. It POSTs
 * to the control routes `server/http_server.cpp` registers:
 *
 *   POST /control/rules         {"session_id", "next_hop", "next_hop_url", "generation"}
 *   POST /control/rules/revoke  {"session_id", "generation"}
 *
 * A rule carries `next_hop_url` because the binary wire format addresses the
 * next hop by instance, and an HTTP forward cannot use one.
 *
 * **This runs on the routing thread**, once per session rather than per token,
 * which is where the deck puts it. It is still a network call under the
 * coordinator's routing lock, so the timeouts are short and a replica known to
 * be unreachable is not dialled at all: a path is replanned around a dead
 * replica, and waiting to tell it so would hold up every other session.
 */

#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include <serving/router/strategy.hpp>

namespace serving::router
{

struct HttpRuleTransportConfig
{
  /// Deliberately short: this is on the routing path. A replica that cannot
  /// accept a rule this fast is one the deployment should be replanning around.
  double connectTimeoutSeconds = 2.0;
  double requestTimeoutSeconds = 5.0;
};

class HttpRuleTransport final : public RuleTransport
{
  public:

  /// `replicas` is every replica in the pipeline: a revoke names no holder, so
  /// it goes to all of them -- which replicas were on the path is exactly what
  /// the coordinator no longer wants to depend on.
  HttpRuleTransport(std::vector<ReplicaSpec> replicas, HttpRuleTransportConfig config = {});

  void publish(const ReplicaSpec &holder, const std::string &sessionId, const ReplicaSpec &nextHop, uint64_t generation) override;

  void revoke(const std::string &sessionId, uint64_t generation) override;

  /// Stop dialling a replica the deployment has given up on. Reversible: a
  /// restarted replica is reachable again, and `markReachable` says so.
  void onReplicaUnreachable(const std::string &replicaName) override;

  void markReachable(const std::string &replicaName);

  /// Replicas this transport will address, refreshed when the deployment
  /// changes shape. Anything no longer listed is also no longer unreachable.
  void setReplicas(std::vector<ReplicaSpec> replicas);

  [[nodiscard]] int published() const;
  [[nodiscard]] int revoked() const;
  /// Posts that never got an answer. A rule that did not land is a session the
  /// next request replans, not a lost request, but it is worth counting.
  [[nodiscard]] int failed() const;

  private:

  /// True when the replica answered. Caller must not hold `_mutex`.
  bool post(const ReplicaSpec &target, const std::string &path, const std::string &body);

  HttpRuleTransportConfig _config;

  mutable std::mutex                 _mutex;
  std::map<std::string, ReplicaSpec> _replicas;
  std::set<std::string>              _unreachable;
  int                                _published = 0;
  int                                _revoked   = 0;
  int                                _failed    = 0;
};

} // namespace serving::router
