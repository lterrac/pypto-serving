#include <serving/model/chat_template.hpp>

#include <fstream>
#include <sstream>

namespace serving::model
{

std::unique_ptr<ChatTemplate> loadChatTemplate(const std::string &modelDir)
{
  std::ifstream in(modelDir + "/tokenizer_config.json", std::ios::binary);
  if (!in) { return nullptr; }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  const auto config = nlohmann::json::parse(buffer.str());
  if (!config.contains("chat_template") || !config.at("chat_template").is_string()) { return nullptr; }

  // bos/eos are either a string or {"content": ...} in tokenizer_config.json.
  const auto content = [&](const char *key) -> std::string {
    if (!config.contains(key) || config.at(key).is_null()) { return {}; }
    const auto &entry = config.at(key);
    if (entry.is_string()) { return entry.get<std::string>(); }
    if (entry.is_object() && entry.contains("content")) { return entry.at("content").get<std::string>(); }
    return {};
  };
  return std::make_unique<ChatTemplate>(config.at("chat_template").get<std::string>(), content("bos_token"), content("eos_token"));
}

} // namespace serving::model
