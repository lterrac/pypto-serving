#include <serving/infra/document.hpp>

#include <fstream>
#include <set>
#include <sstream>
#include <system_error>

#include <nlohmann/json.hpp>

namespace serving::infra
{

namespace
{

using json = nlohmann::ordered_json;

/// Where in the document we are, so a message names its subject.
struct Context
{
  std::string where;

  [[nodiscard]] Context       in(const std::string &child) const { return Context{where + ": " + child}; }
  [[nodiscard]] DocumentError error(const std::string &what) const { return DocumentError(where + ": " + what); }
};

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

/// A name that will be interpolated into shell commands, paths and routing
/// keys. Rejected here rather than carried into the map: `isUsableName` states
/// why, and this is the edge a document from another process crosses.
std::string requireName(const json &object, const char *key, const Context &ctx)
{
  const std::string value = requireString(object, key, ctx);
  if (!isUsableName(value))
  {
    throw ctx.error(std::string("'") + key + "' must be 1-64 characters of letters, digits, '.', '_' or '-'; got '" + value + "'");
  }
  return value;
}

std::string optionalString(const json &object, const char *key, const std::string &fallback, const Context &ctx)
{
  const auto it = object.find(key);
  if (it == object.end()) { return fallback; }
  if (!it->is_string()) { throw ctx.error(std::string("'") + key + "' must be a string"); }
  return it->get<std::string>();
}

long long integerIn(const json &value, const char *key, long long lo, long long hi, const Context &ctx)
{
  if (!value.is_number_integer()) { throw ctx.error(std::string("'") + key + "' must be an integer"); }
  const auto number = value.get<long long>();
  if (number < lo || number > hi) { throw ctx.error(std::string("'") + key + "' must be between " + std::to_string(lo) + " and " + std::to_string(hi)); }
  return number;
}

long long optionalInteger(const json &object, const char *key, long long fallback, long long lo, long long hi, const Context &ctx)
{
  const auto it = object.find(key);
  if (it == object.end()) { return fallback; }
  return integerIn(*it, key, lo, hi, ctx);
}

std::vector<int> deviceList(const json &object, const char *key, const Context &ctx)
{
  const auto it = object.find(key);
  if (it == object.end()) { return {}; }
  if (!it->is_array()) { throw ctx.error(std::string("'") + key + "' must be an array of integers"); }

  std::vector<int> devices;
  devices.reserve(it->size());
  for (const json &entry : *it) { devices.push_back(static_cast<int>(integerIn(entry, key, 0, 1024, ctx))); }
  return devices;
}

Phase requirePhase(const json &object, const char *key, Phase fallback, const Context &ctx)
{
  const auto it = object.find(key);
  if (it == object.end()) { return fallback; }
  if (!it->is_string()) { throw ctx.error(std::string("'") + key + "' must be a string"); }

  const auto phase = parsePhase(it->get<std::string>());
  if (!phase.has_value()) { throw ctx.error(std::string("'") + key + "': unknown phase '" + it->get<std::string>() + "'"); }
  return *phase;
}

} // namespace

std::string writeDocument(const StateMap &map)
{
  json document;
  document["format"]  = DOCUMENT_FORMAT;
  document["version"] = map.version();

  json nodes = json::array();
  for (const Node &node : map.nodes()) { nodes.push_back(json{{"name", node.name}, {"host", node.host}, {"devices", node.devices}}); }
  document["nodes"] = std::move(nodes);

  json replicas = json::array();
  for (const Replica &replica : map.replicas())
  {
    replicas.push_back(json{{"name", replica.name},
                            {"partition", replica.partition},
                            {"node", replica.node},
                            {"devices", replica.devices},
                            {"port", replica.port},
                            {"scheme", replica.scheme},
                            {"instance", replica.instance},
                            {"phase", phaseName(replica.phase)},
                            {"desired_phase", phaseName(replica.desiredPhase)}});
  }
  document["replicas"] = std::move(replicas);

  json channels = json::array();
  for (const Channel &channel : map.channels()) { channels.push_back(json{{"from", channel.from}, {"to", channel.to}}); }
  document["channels"] = std::move(channels);

  return document.dump(2) + "\n";
}

StateMap readDocument(const std::string &text, const std::string &source)
{
  const Context root{source};

  json document;
  try
  {
    document = json::parse(text);
  }
  catch (const json::exception &e)
  {
    throw root.error(std::string("not valid JSON: ") + e.what());
  }
  if (!document.is_object()) { throw root.error("the document must be a JSON object"); }
  rejectUnknownKeys(document, {"format", "version", "nodes", "replicas", "channels"}, root);

  const auto format = optionalInteger(document, "format", DOCUMENT_FORMAT, 1, 1 << 20, root);
  if (format != DOCUMENT_FORMAT) { throw root.error("format " + std::to_string(format) + " is not one this build reads (expected " + std::to_string(DOCUMENT_FORMAT) + ")"); }

  StateMap map;

  if (const auto it = document.find("nodes"); it != document.end())
  {
    if (!it->is_array()) { throw root.error("'nodes' must be an array"); }
    for (size_t i = 0; i < it->size(); ++i)
    {
      const Context ctx   = root.in("node " + std::to_string(i));
      const json   &entry = (*it)[i];
      if (!entry.is_object()) { throw ctx.error("must be an object"); }
      rejectUnknownKeys(entry, {"name", "host", "devices"}, ctx);

      Node node;
      node.name    = requireName(entry, "name", ctx);
      node.host    = optionalString(entry, "host", "", ctx);
      node.devices = deviceList(entry, "devices", ctx);
      map.addNode(std::move(node));
    }
  }

  if (const auto it = document.find("replicas"); it != document.end())
  {
    if (!it->is_array()) { throw root.error("'replicas' must be an array"); }
    for (size_t i = 0; i < it->size(); ++i)
    {
      const Context ctx   = root.in("replica " + std::to_string(i));
      const json   &entry = (*it)[i];
      if (!entry.is_object()) { throw ctx.error("must be an object"); }
      rejectUnknownKeys(entry, {"name", "partition", "node", "devices", "port", "scheme", "instance", "phase", "desired_phase"}, ctx);

      Replica replica;
      replica.name      = requireName(entry, "name", ctx);
      replica.partition = static_cast<PartitionId>(optionalInteger(entry, "partition", 0, 0, 1 << 20, ctx));
      replica.node      = optionalString(entry, "node", "", ctx);
      replica.devices   = deviceList(entry, "devices", ctx);
      replica.port      = static_cast<int>(optionalInteger(entry, "port", 0, 0, 65535, ctx));
      replica.scheme    = optionalString(entry, "scheme", "http", ctx);
      replica.instance  = static_cast<uint64_t>(optionalInteger(entry, "instance", 0, 0, (1LL << 62), ctx));
      // A replica with no phase has not been observed yet; one with no desired
      // phase is wanted serving, which is why it is in the document at all.
      replica.phase        = requirePhase(entry, "phase", Phase::Pending, ctx);
      replica.desiredPhase = requirePhase(entry, "desired_phase", Phase::Active, ctx);
      map.addReplica(std::move(replica));
    }
  }

  if (const auto it = document.find("channels"); it != document.end())
  {
    if (!it->is_array()) { throw root.error("'channels' must be an array"); }
    for (size_t i = 0; i < it->size(); ++i)
    {
      const Context ctx   = root.in("channel " + std::to_string(i));
      const json   &entry = (*it)[i];
      if (!entry.is_object()) { throw ctx.error("must be an object"); }
      rejectUnknownKeys(entry, {"from", "to"}, ctx);
      map.addChannel(Channel{requireString(entry, "from", ctx), requireString(entry, "to", ctx)});
    }
  }

  // Last, so building the map does not leave the version counting the load.
  map.restoreVersion(static_cast<uint64_t>(optionalInteger(document, "version", static_cast<long long>(map.version()), 0, (1LL << 62), root)));
  return map;
}

StateMap loadDocument(const std::filesystem::path &path)
{
  std::ifstream input(path);
  if (!input) { throw DocumentError(path.string() + ": cannot be read"); }

  std::ostringstream buffer;
  buffer << input.rdbuf();
  return readDocument(buffer.str(), path.string());
}

void saveDocument(const StateMap &map, const std::filesystem::path &path)
{
  // Same directory, so the rename below stays within one filesystem.
  const std::filesystem::path temporary = std::filesystem::path(path).replace_extension(path.extension().string() + ".tmp");

  {
    std::ofstream output(temporary, std::ios::trunc);
    if (!output) { throw DocumentError(temporary.string() + ": cannot be written"); }
    output << writeDocument(map);
    if (!output) { throw DocumentError(temporary.string() + ": write failed"); }
  }

  std::error_code ec;
  std::filesystem::rename(temporary, path, ec);
  if (ec)
  {
    std::filesystem::remove(temporary, ec);
    throw DocumentError(path.string() + ": could not be replaced");
  }
}

} // namespace serving::infra
