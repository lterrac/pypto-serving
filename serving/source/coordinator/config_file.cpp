#include <serving/coordinator/config_file.hpp>

#include <fstream>
#include <limits>
#include <set>
#include <sstream>

#include <nlohmann/json.hpp>

namespace serving::coordinator
{

namespace
{

using nlohmann::json;

/// Error messages of the form "<source>: <where>: <what>".
class Context
{
  public:

  Context(const std::string &source, std::string where)
    : _source(source),
      _where(std::move(where))
  {}

  [[nodiscard]] ConfigError error(const std::string &what) const { return ConfigError(_source + ": " + (_where.empty() ? "" : _where + ": ") + what); }

  [[nodiscard]] Context at(const std::string &where) const { return Context(_source, _where.empty() ? where : _where + "." + where); }

  private:

  const std::string &_source;
  std::string        _where;
};

/// Unknown keys are errors, so a typo cannot pass as a setting.
void rejectUnknownKeys(const json &object, const std::set<std::string> &allowed, const Context &ctx)
{
  for (const auto &[key, value] : object.items())
  {
    if (allowed.count(key) == 0) { throw ctx.error("unknown key '" + key + "'"); }
  }
}

const json &require(const json &object, const char *key, const Context &ctx)
{
  const auto it = object.find(key);
  if (it == object.end()) { throw ctx.error(std::string("'") + key + "' is required"); }
  return *it;
}

std::string requireString(const json &object, const char *key, const Context &ctx)
{
  const json &value = require(object, key, ctx);
  if (!value.is_string()) { throw ctx.error(std::string("'") + key + "' must be a string"); }
  return value.get<std::string>();
}

/// An integer in [lo, hi]. Booleans and floats are numbers to JSON but never
/// what a count or a port means, so they are refused rather than coerced.
long long integerIn(const json &value, const char *key, long long lo, long long hi, const Context &ctx)
{
  if (!value.is_number_integer())
  {
    throw ctx.error(std::string("'") + key + "' must be an integer in " + std::to_string(lo) + ".." + std::to_string(hi) + ", got " + value.dump());
  }
  const auto n = value.get<long long>();
  if (n < lo || n > hi) { throw ctx.error(std::string("'") + key + "' must be in " + std::to_string(lo) + ".." + std::to_string(hi) + ", got " + std::to_string(n)); }
  return n;
}

ReplicaBinding parseReplica(const json &entry, size_t index, const Context &parent)
{
  const Context ctx = parent.at("replica " + std::to_string(index));
  if (!entry.is_object()) { throw ctx.error("must be an object"); }
  rejectUnknownKeys(entry, {"name", "host", "port", "scheme", "partition", "instance"}, ctx);

  ReplicaBinding binding;
  binding.spec.name = requireString(entry, "name", ctx);
  if (binding.spec.name.empty()) { throw ctx.error("'name' must not be empty"); }
  binding.spec.host = requireString(entry, "host", ctx);
  if (binding.spec.host.empty()) { throw ctx.error("'host' must not be empty"); }
  binding.spec.port = static_cast<int>(integerIn(require(entry, "port", ctx), "port", 1, 65535, ctx));

  if (const auto it = entry.find("scheme"); it != entry.end())
  {
    if (!it->is_string()) { throw ctx.error("'scheme' must be a string"); }
    const auto scheme = it->get<std::string>();
    if (scheme != "http" && scheme != "https") { throw ctx.error("'scheme' must be 'http' or 'https', got '" + scheme + "'"); }
    binding.spec.scheme = scheme;
  }
  if (const auto it = entry.find("partition"); it != entry.end())
  {
    binding.spec.partition = static_cast<router::PartitionId>(integerIn(*it, "partition", 0, std::numeric_limits<int>::max(), ctx));
  }
  if (const auto it = entry.find("instance"); it != entry.end())
  {
    binding.instanceId = static_cast<uint64_t>(integerIn(*it, "instance", 0, std::numeric_limits<long long>::max(), ctx));
  }
  return binding;
}

router::RoutingConfig parseRouting(const json &object, const Context &parent)
{
  const Context ctx = parent.at("routing");
  if (!object.is_object()) { throw ctx.error("must be an object"); }
  rejectUnknownKeys(object, {"session_ttl_seconds", "affinity_slack"}, ctx);

  router::RoutingConfig routing;
  if (const auto it = object.find("session_ttl_seconds"); it != object.end())
  {
    if (!it->is_number() || it->is_boolean()) { throw ctx.error("'session_ttl_seconds' must be a number"); }
    routing.sessionTtlSeconds = it->get<double>();
    if (routing.sessionTtlSeconds <= 0) { throw ctx.error("'session_ttl_seconds' must be positive"); }
  }
  if (const auto it = object.find("affinity_slack"); it != object.end())
  {
    routing.affinitySlack = static_cast<int>(integerIn(*it, "affinity_slack", 0, std::numeric_limits<int>::max(), ctx));
  }
  return routing;
}

} // namespace

CoordinatorConfig parseCoordinatorConfig(const std::string &text, const std::string &source)
{
  const Context ctx(source, "");

  json document;
  try
  {
    document = json::parse(text);
  }
  catch (const json::parse_error &e)
  {
    throw ctx.error(std::string("not valid JSON: ") + e.what());
  }
  if (!document.is_object()) { throw ctx.error("must be a JSON object"); }

  // `hosts` is the router launcher's key; the coordinator does not launch.
  if (document.contains("hosts"))
  {
    throw ctx.error("'hosts' declares launchable capacity for the router's launcher; the coordinator routes to replicas that already exist, so list them under 'replicas'");
  }
  rejectUnknownKeys(document, {"routing_mode", "partition", "routing", "replicas"}, ctx);

  CoordinatorConfig config;

  // Required: the file exists to choose it.
  const std::string modeText = requireString(document, "routing_mode", ctx);
  try
  {
    config.mode = router::parseRoutingMode(modeText);
  }
  catch (const std::invalid_argument &e)
  {
    throw ctx.error(std::string("'routing_mode': ") + e.what());
  }

  if (const auto it = document.find("partition"); it != document.end())
  {
    config.partition = static_cast<router::PartitionId>(integerIn(*it, "partition", 0, std::numeric_limits<int>::max(), ctx));
  }

  if (const auto it = document.find("routing"); it != document.end()) { config.routing = parseRouting(*it, ctx); }

  const json &replicas = require(document, "replicas", ctx);
  if (!replicas.is_array()) { throw ctx.error("'replicas' must be a list"); }
  if (replicas.empty()) { throw ctx.error("'replicas' must list at least one replica"); }

  std::set<std::string>         names;
  std::set<std::string>         endpoints;
  std::set<uint64_t>            instances;
  std::set<router::PartitionId> partitions;
  bool                          ownsSomething = false;
  for (size_t i = 0; i < replicas.size(); ++i)
  {
    const std::string where   = "replica " + std::to_string(i);
    ReplicaBinding    binding = parseReplica(replicas[i], i, ctx);
    if (!names.insert(binding.spec.name).second) { throw ctx.error(where + ": duplicate name '" + binding.spec.name + "'"); }
    if (!endpoints.insert(binding.spec.baseUrl()).second) { throw ctx.error(where + ": duplicate endpoint " + binding.spec.baseUrl()); }

    // In rules mode the instance is what a published rule names. Left out it
    // would default to 0 and address whichever replica happens to be instance
    // 0, which looks like a working deployment.
    if (config.mode == router::RoutingMode::DistributedRules)
    {
      if (!replicas[i].contains("instance")) { throw ctx.error(where + ": 'instance' is required when routing_mode is 'rules'"); }
      if (!instances.insert(binding.instanceId).second) { throw ctx.error(where + ": duplicate instance " + std::to_string(binding.instanceId)); }
    }

    partitions.insert(binding.spec.partition);
    ownsSomething = ownsSomething || binding.spec.partition == config.partition;
    config.replicas.push_back(std::move(binding));
  }

  // A pipeline is a chain: a gap means plan() produces a path that skips a
  // stage, which routes around a partition rather than through it.
  router::PartitionId expected = 0;
  for (const router::PartitionId partition : partitions)
  {
    if (partition != expected)
    {
      throw ctx.error("partitions must be 0.." + std::to_string(partitions.size() - 1) + " with no gaps; found " + std::to_string(partition) + " where " +
                      std::to_string(expected) + " was expected");
    }
    expected += 1;
  }

  // A partition no replica serves is a misconfiguration.
  if (!ownsSomething) { throw ctx.error("'partition' is " + std::to_string(config.partition) + " but no replica belongs to that partition"); }

  return config;
}

CoordinatorConfig loadCoordinatorConfig(const std::filesystem::path &path)
{
  std::ifstream in(path);
  if (!in) { throw ConfigError(path.string() + ": cannot read"); }
  std::stringstream buffer;
  buffer << in.rdbuf();
  return parseCoordinatorConfig(buffer.str(), path.string());
}

} // namespace serving::coordinator
