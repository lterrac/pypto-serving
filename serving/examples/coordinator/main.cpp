/**
 * Runs CoordinatorModule through initialize, routing, a replica loss, a
 * service tick and finalize, with each published rule carried through the
 * wire format into the receiving replica's own table -- the loop Option B
 * closes: the coordinator plans once, the replicas forward without it.
 *
 * No HiCR instances are stood up, so the frames are handed to the replica
 * handlers directly instead of travelling a channel. What a live deployment
 * adds is the channel itself; the bytes and the decisions are these.
 *
 *   coordinator-module --config FILE     (coordinator.json is one)
 */

#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <serving/coordinator/channel_transport.hpp>
#include <serving/coordinator/config_file.hpp>
#include <serving/coordinator/platform_module.hpp>
#include <serving/replica/channel_rules.hpp>

namespace
{

using namespace serving;

/**
 * Publishes as a channel deployment does -- encode, then deliver -- except
 * that delivery is a direct call rather than a channel send. Each replica gets
 * the table it would keep, fed by the handlers `RuleModule` subscribes with.
 */
class ReplicaFacingTransport : public router::RuleTransport
{
  public:

  void publish(const router::ReplicaSpec &holder, const std::string &sessionId, const router::ReplicaSpec &nextHop, uint64_t generation) override
  {
    std::printf("    [rule gen=%lu] %s -> %s for session %s\n", static_cast<unsigned long>(generation), holder.name.c_str(), nextHop.name.c_str(), sessionId.c_str());

    coordinator::RulePublish rule;
    rule.sessionId       = sessionId;
    rule.nextHopName     = nextHop.name;
    rule.nextHopInstance = instanceOf(nextHop.name);
    rule.generation      = generation;

    const auto frame = coordinator::encodeRulePublish(rule);
    deliver(holder.name, frame, true);
  }

  void revoke(const std::string &sessionId, uint64_t generation) override
  {
    std::printf("    [revoke gen=%lu] session %s\n", static_cast<unsigned long>(generation), sessionId.c_str());

    coordinator::RuleRevoke message;
    message.sessionId  = sessionId;
    message.generation = generation;

    // A revoke goes to everyone: the coordinator withdraws a path, and which
    // replicas were on it is exactly what it no longer wants to depend on.
    const auto frame = coordinator::encodeRuleRevoke(message);
    for (const auto &[name, replica] : _replicas) { (void)name, replica->onRevoke(nullptr, system::channels::Message(frame.data(), frame.size(), _revokeMetadata)); }
  }

  /// What `name` would do with the next message of `sessionId`.
  [[nodiscard]] std::optional<replica::NextHop> heldBy(const std::string &name, const std::string &sessionId)
  {
    const auto it = _replicas.find(name);
    return it == _replicas.end() ? std::nullopt : it->second->table.peek(sessionId);
  }

  [[nodiscard]] const replica::ChannelDrain &tally(const std::string &name) { return replicaFor(name).tally; }

  [[nodiscard]] std::vector<std::string> replicaNames() const
  {
    std::vector<std::string> names;
    for (const auto &[name, replica] : _replicas) { (void)replica, names.push_back(name); }
    return names;
  }

  private:

  /// One replica's receiving end: its table, its tally, and the two handlers.
  struct Receiver
  {
    replica::RuleTable        table;
    replica::ChannelDrain     tally;
    modules::messageHandler_t onPublish = replica::rulePublishHandler(table, &tally);
    modules::messageHandler_t onRevoke  = replica::ruleRevokeHandler(table, &tally);
  };

  /// A channel deployment is handed the name -> instance table by the engine;
  /// here the ids are made up, consistently, so the frames carry a real one.
  uint64_t instanceOf(const std::string &name)
  {
    const auto [it, inserted] = _instances.emplace(name, _instances.size() + 1);
    (void)inserted;
    return it->second;
  }

  Receiver &replicaFor(const std::string &name)
  {
    auto it = _replicas.find(name);
    if (it == _replicas.end()) { it = _replicas.emplace(name, std::make_unique<Receiver>()).first; }
    return *it->second;
  }

  void deliver(const std::string &name, const std::vector<uint8_t> &frame, bool publish)
  {
    auto                           &replica = replicaFor(name);
    const system::channels::Message message(frame.data(), frame.size(), publish ? _publishMetadata : _revokeMetadata);
    publish ? replica.onPublish(nullptr, message) : replica.onRevoke(nullptr, message);
  }

  std::map<std::string, uint64_t> _instances;

  system::channels::MessageTypeRegistry _registry;
  system::channels::Message::metadata_t _publishMetadata{_registry.registerType(coordinator::RULE_PUBLISH_MESSAGE), 0, 0};
  system::channels::Message::metadata_t _revokeMetadata{_registry.registerType(coordinator::RULE_REVOKE_MESSAGE), 0, 0};

  // Held by pointer: a Receiver owns handlers that capture its own members, so
  // it must not move when the map grows.
  std::map<std::string, std::unique_ptr<Receiver>> _replicas;
};

void printPath(const router::RoutingPath &path)
{
  std::printf("    path: ");
  for (size_t i = 0; i < path.hops.size(); ++i) { std::printf("%s%s", i ? " -> " : "", path.hops[i].name.c_str()); }
  std::printf("   (generation %lu)\n", static_cast<unsigned long>(path.generation));
}

} // namespace

int main(int argc, char **argv)
{
  std::string configPath;
  for (int i = 1; i < argc; ++i)
  {
    const std::string arg = argv[i];
    if (arg == "--config" && i + 1 < argc) { configPath = argv[++i]; }
    else if (arg == "--help" || arg == "-h")
    {
      std::printf("usage: %s --config FILE\n", argv[0]);
      return 0;
    }
  }
  if (configPath.empty())
  {
    std::fprintf(stderr, "usage: %s --config FILE\n", argv[0]);
    return 2;
  }

  coordinator::CoordinatorConfig config;
  try
  {
    config = coordinator::loadCoordinatorConfig(configPath);
  }
  catch (const std::exception &e)
  {
    std::fprintf(stderr, "fatal: %s\n", e.what());
    return 1;
  }

  ReplicaFacingTransport transport;
  // A real deployment registers this with serving::system::Engine::addModule,
  // which then drives the lifecycle below.
  coordinator::CoordinatorModule module(config, &transport);

  std::printf("coordinator for partition %d, mode %s (from %s)\n\n", config.partition, router::routingModeName(config.mode), configPath.c_str());

  std::printf("  initialize()\n");
  module.initialize();
  std::printf("    has a service: %s\n\n", module.hasService() ? "yes" : "no");

  std::printf("  two turns of a conversation\n");
  printPath(module.coordinator().routeFor("chat-1"));
  const auto second = module.coordinator().routeFor("chat-1");
  printPath(second);

  std::printf("\n  a heartbeat reports %s gone\n", second.hops[1].name.c_str());
  module.coordinator().reportReplicaLost(second.hops[1].name);
  std::printf("    before the tick, losses applied: %d\n", module.coordinator().lossesApplied());

  // The platform engine drives this on its timer; stepped by hand here.
  std::printf("  service() tick\n");
  module.coordinator().service();
  std::printf("    after the tick, losses applied: %d\n", module.coordinator().lossesApplied());

  std::printf("\n  the next turn re-routes around it\n");
  printPath(module.coordinator().routeFor("chat-1"));

  std::printf("\n  what each replica now holds for chat-1 -- this is what it forwards on, without asking\n");
  for (const auto &name : transport.replicaNames())
  {
    const auto  hop   = transport.heldBy(name, "chat-1");
    const auto &tally = transport.tally(name);
    std::printf("    %-8s -> %-8s  (applied=%d revoked=%d stale=%d unknown=%d rejected=%d)\n",
                name.c_str(),
                hop.has_value() ? hop->name.c_str() : "(nothing)",
                tally.applied,
                tally.revoked,
                tally.stale,
                tally.unknown,
                tally.rejected);
  }

  const auto stats = module.coordinator().stats();
  std::printf("\n  plans: %d   rules published: %d   paths invalidated: %d\n", stats.plansComputed, stats.rulesPublished, stats.pathsInvalidated);

  std::printf("\n  finalize()\n");
  module.finalize();
  std::printf("    initialized: %s\n", module.coordinator().initialized() ? "yes" : "no");
  return 0;
}
