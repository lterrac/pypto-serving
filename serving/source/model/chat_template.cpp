#include <serving/model/chat_template.hpp>

#include <serving/model/model_files.hpp>

namespace serving::model
{

std::unique_ptr<ChatTemplate> loadChatTemplate(const std::string &modelDir)
{
  const auto text = readFile(modelDir + "/tokenizer_config.json");
  if (!text.has_value()) { return nullptr; }
  const auto config = nlohmann::json::parse(*text);
  if (!config.contains("chat_template") || !config.at("chat_template").is_string()) { return nullptr; }

  return std::make_unique<ChatTemplate>(config.at("chat_template").get<std::string>(), specialTokenContent(config, "bos_token"), specialTokenContent(config, "eos_token"));
}

} // namespace serving::model
