#pragma once

/**
 * Chat-template rendering with minja, matching transformers'
 * apply_chat_template. Polyfills stay off: transformers renders the template
 * as it is.
 */

#include <chrono>
#include <memory>
#include <string>
#include <utility>

#include <minja/chat-template.hpp>
#include <nlohmann/json.hpp>

namespace serving::model
{

/// One model's chat template, rendered the way transformers renders it.
class ChatTemplate
{
  public:

  ChatTemplate(const std::string &source, const std::string &bosToken, const std::string &eosToken)
    : _template(source, bosToken, eosToken)
  {}

  /**
   * Render `messages` into the model's generation prompt.
   *
   * `extraContext` carries the template-specific keyword arguments callers pass
   * to `apply_chat_template` -- for Qwen3 that is `enable_thinking`.
   */
  [[nodiscard]] std::string apply(const nlohmann::ordered_json &messages,
                                  bool                          addGenerationPrompt,
                                  const nlohmann::ordered_json &extraContext = nlohmann::ordered_json::object(),
                                  const nlohmann::ordered_json &tools        = nlohmann::ordered_json()) const
  {
    minja::chat_template_inputs inputs;
    inputs.messages              = messages;
    inputs.tools                 = tools;
    inputs.add_generation_prompt = addGenerationPrompt;
    inputs.extra_context         = extraContext;
    inputs.now                   = std::chrono::system_clock::now();

    minja::chat_template_options options;
    // See the parity note above: transformers does not polyfill, so neither do we.
    options.apply_polyfills = false;

    return _template.apply(inputs, options);
  }

  [[nodiscard]] const std::string &source() const { return _template.source(); }

  private:

  // minja::chat_template::apply is non-const.
  mutable minja::chat_template _template;
};

/// The template from `<modelDir>/tokenizer_config.json`, or nullptr when the
/// file has none. A malformed file throws.
std::unique_ptr<ChatTemplate> loadChatTemplate(const std::string &modelDir);

} // namespace serving::model
