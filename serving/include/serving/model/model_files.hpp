#pragma once

/// Reading the JSON a model directory ships.

#include <optional>
#include <string>

#include <nlohmann/json.hpp>

namespace serving::model
{

/// The file's contents, or nullopt if it cannot be opened.
[[nodiscard]] std::optional<std::string> readFile(const std::string &path);

/// A special token's text from `tokenizer_config.json`, where an entry is
/// either the string itself or an object carrying "content". Empty if absent.
[[nodiscard]] std::string specialTokenContent(const nlohmann::json &config, const std::string &key);

} // namespace serving::model
