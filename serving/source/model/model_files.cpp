#include <serving/model/model_files.hpp>

#include <fstream>
#include <sstream>

namespace serving::model
{

std::optional<std::string> readFile(const std::string &path)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) { return std::nullopt; }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

std::string specialTokenContent(const nlohmann::json &config, const std::string &key)
{
  if (!config.contains(key) || config.at(key).is_null()) { return {}; }
  const auto &entry = config.at(key);
  if (entry.is_string()) { return entry.get<std::string>(); }
  if (entry.is_object() && entry.contains("content")) { return entry.at("content").get<std::string>(); }
  return {};
}

} // namespace serving::model
