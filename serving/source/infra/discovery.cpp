#include <serving/infra/discovery.hpp>

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <system_error>

namespace serving::infra
{

namespace
{

constexpr const char *DEVICE_PREFIX = "davinci";

/// The index in `davinci<N>`, or nullopt for anything else in the directory --
/// `davinci_manager` above all, which is a control node and not a card.
std::optional<int> deviceIndex(const std::string &filename)
{
  const std::string prefix = DEVICE_PREFIX;
  if (filename.size() <= prefix.size() || filename.compare(0, prefix.size(), prefix) != 0) { return std::nullopt; }

  const std::string suffix = filename.substr(prefix.size());
  if (suffix.empty() || !std::all_of(suffix.begin(), suffix.end(), [](unsigned char c) { return std::isdigit(c) != 0; })) { return std::nullopt; }

  int        value  = 0;
  const auto result = std::from_chars(suffix.data(), suffix.data() + suffix.size(), value);
  if (result.ec != std::errc{}) { return std::nullopt; }
  return value;
}

} // namespace

std::vector<int> discoverDevices(const std::string &deviceDir)
{
  std::vector<int>                    devices;
  std::error_code                     ec;
  std::filesystem::directory_iterator it(deviceDir, ec);
  // A missing or unreadable directory is a machine with no accelerators, not an
  // error: a CPU-only build walks this path every time.
  if (ec) { return devices; }

  for (const auto &entry : it)
  {
    if (const auto index = deviceIndex(entry.path().filename().string()); index.has_value()) { devices.push_back(*index); }
  }
  std::sort(devices.begin(), devices.end());
  return devices;
}

Node localNode(std::string name, std::string host, const std::string &deviceDir)
{
  Node node;
  node.name    = std::move(name);
  node.host    = std::move(host);
  node.devices = discoverDevices(deviceDir);
  return node;
}

} // namespace serving::infra
