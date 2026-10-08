#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <serving/coordinator/module.hpp>
#include <serving/coordinator/rule_message.hpp>
#include <serving/replica/rule_table.hpp>

using serving::coordinator::Coordinator;
using serving::coordinator::CoordinatorConfig;
using serving::coordinator::decodeRulePublish;
using serving::coordinator::decodeRuleRevoke;
using serving::coordinator::encodeRulePublish;
using serving::coordinator::encodeRuleRevoke;
using serving::coordinator::ReplicaBinding;
using serving::coordinator::RulePublish;
using serving::coordinator::RuleRevoke;
using serving::replica::NextHop;
using serving::replica::RuleTable;
using serving::router::ReplicaSpec;
using serving::router::RoutingMode;
using serving::router::RuleTransport;

namespace
{

ReplicaBinding binding(const std::string &name, int partition, uint64_t instanceId)
{
  ReplicaBinding b;
  b.spec.name      = name;
  b.spec.partition = partition;
  b.spec.host      = "10.0.0." + std::to_string(partition);
  b.spec.port      = 8000;
  b.instanceId     = instanceId;
  return b;
}

/**
 * Carries the coordinator's messages into the replicas' tables, through the
 * same encode/decode a channel would use.
 *
 * This is the pairing `rules` mode never had: until now the publisher was only
 * ever exercised against a transport that recorded strings.
 */
class LoopbackTransport final : public RuleTransport
{
  public:

  explicit LoopbackTransport(std::map<std::string, RuleTable> &tables)
    : _tables(tables)
  {}

  void publish(const ReplicaSpec &holder, const std::string &sessionId, const ReplicaSpec &nextHop, uint64_t generation) override
  {
    if (_unreachable.count(holder.name) > 0) { return; }

    RulePublish message;
    message.sessionId       = sessionId;
    message.nextHopName     = nextHop.name;
    message.nextHopInstance = 0;
    message.generation      = generation;

    const auto frame   = encodeRulePublish(message);
    const auto decoded = decodeRulePublish(frame.data(), frame.size());
    if (!decoded.has_value()) { return; }
    _tables[holder.name].apply(*decoded);
    published += 1;
  }

  void revoke(const std::string &sessionId, uint64_t generation) override
  {
    const auto frame   = encodeRuleRevoke(RuleRevoke{sessionId, generation});
    const auto decoded = decodeRuleRevoke(frame.data(), frame.size());
    if (!decoded.has_value()) { return; }
    // A revoke has no single holder, so every replica hears it.
    for (auto &[name, table] : _tables)
    {
      if (_unreachable.count(name) == 0) { table.revoke(*decoded); }
    }
    revoked += 1;
  }

  void onReplicaUnreachable(const std::string &replicaName) override { _unreachable.insert(replicaName); }

  int published = 0;
  int revoked   = 0;

  private:

  std::map<std::string, RuleTable> &_tables;
  std::set<std::string>             _unreachable;
};

/// Walk the pipeline from the tables alone, the way a replica forwards without
/// asking anyone.
std::vector<std::string> walk(std::map<std::string, RuleTable> &tables, const std::string &from, const std::string &sessionId)
{
  std::vector<std::string> path{from};
  std::string              at = from;
  for (int hop = 0; hop < 8; ++hop)
  {
    const auto next = tables[at].lookup(sessionId);
    if (!next.has_value()) { break; }
    at = next->name;
    path.push_back(at);
  }
  return path;
}

CoordinatorConfig pipeline()
{
  CoordinatorConfig config;
  config.partition = 0;
  config.mode      = RoutingMode::DistributedRules;
  config.replicas  = {binding("p0a", 0, 10), binding("p0b", 0, 11), binding("p1a", 1, 20), binding("p1b", 1, 21), binding("p2a", 2, 30)};
  return config;
}

} // namespace

TEST(PipelineTest, AReplicaCanForwardWithoutAskingTheCoordinator)
{
  std::map<std::string, RuleTable> tables;
  LoopbackTransport                transport(tables);
  Coordinator                      coordinator(pipeline(), &transport);
  coordinator.initialize();

  const auto planned = coordinator.routeFor("chat-1");
  ASSERT_EQ(planned.hops.size(), 3U);

  // The coordinator decided once. Everything after this is the replicas.
  const auto               walked = walk(tables, planned.hops.front().name, "chat-1");
  std::vector<std::string> expected;
  for (const auto &hop : planned.hops) { expected.push_back(hop.name); }
  EXPECT_EQ(walked, expected);

  // The last hop holds nothing: there is nothing after it.
  EXPECT_FALSE(tables[planned.hops.back().name].peek("chat-1").has_value());
}

TEST(PipelineTest, TheWholeSessionFollowsOnePathAcrossTurns)
{
  std::map<std::string, RuleTable> tables;
  LoopbackTransport                transport(tables);
  Coordinator                      coordinator(pipeline(), &transport);
  coordinator.initialize();

  const auto first = coordinator.routeFor("chat-1");
  const int  after = transport.published;

  // The second turn of the same conversation. Option B's whole premise is that
  // routing happens once per session, so the cached data is reused.
  const auto second = coordinator.routeFor("chat-1");
  EXPECT_EQ(first.hops.front().name, second.hops.front().name);
  EXPECT_EQ(transport.published, after) << "a second turn should not republish the path";

  EXPECT_EQ(walk(tables, first.hops.front().name, "chat-1").size(), 3U);
}

TEST(PipelineTest, LosingAReplicaTakesThePathsThroughItOutOfTheTables)
{
  std::map<std::string, RuleTable> tables;
  LoopbackTransport                transport(tables);
  Coordinator                      coordinator(pipeline(), &transport);
  coordinator.initialize();

  const auto path = coordinator.routeFor("chat-1");
  ASSERT_EQ(walk(tables, path.hops.front().name, "chat-1").size(), 3U);

  // The middle of the pipeline goes.
  coordinator.reportReplicaLost(path.hops[1].name);
  coordinator.service();

  // A replica still holding the rule would forward into a replica that is
  // gone, which is what the revoke exists to prevent.
  EXPECT_GT(transport.revoked, 0);
  EXPECT_FALSE(tables[path.hops.front().name].peek("chat-1").has_value());
}

TEST(PipelineTest, AReplannedSessionIsRepublishedAndFollowsTheNewPath)
{
  std::map<std::string, RuleTable> tables;
  LoopbackTransport                transport(tables);
  Coordinator                      coordinator(pipeline(), &transport);
  coordinator.initialize();

  const auto before = coordinator.routeFor("chat-1");
  coordinator.reportReplicaLost(before.hops[1].name);
  coordinator.service();

  // The next turn gets a fresh path, at a higher generation, and the tables
  // agree with it rather than with the one that was revoked.
  const auto after = coordinator.routeFor("chat-1");
  EXPECT_GT(after.generation, before.generation);

  const auto               walked = walk(tables, after.hops.front().name, "chat-1");
  std::vector<std::string> expected;
  for (const auto &hop : after.hops) { expected.push_back(hop.name); }
  EXPECT_EQ(walked, expected);
  EXPECT_EQ(std::find(walked.begin(), walked.end(), before.hops[1].name), walked.end()) << "the lost replica is still on the path";
}
