#include <serving/server/http_server.hpp>

#include <chrono>
#include <map>
#include <stdexcept>

#include <httplib.h>
#include <nlohmann/json.hpp>

namespace serving::server
{

namespace
{

using nlohmann::json;

long nowSeconds() { return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count(); }

/// Pull a GenerateConfig out of a request body, falling back to the defaults.
config::GenerateConfig parseGenerateConfig(const json &body)
{
  config::GenerateConfig generate;
  if (body.contains("max_tokens") && body.at("max_tokens").is_number()) { generate.maxNewTokens = body.at("max_tokens").get<int>(); }
  if (body.contains("temperature") && body.at("temperature").is_number()) { generate.temperature = body.at("temperature").get<double>(); }
  if (body.contains("top_p") && body.at("top_p").is_number()) { generate.topP = body.at("top_p").get<double>(); }
  if (body.contains("top_k") && body.at("top_k").is_number()) { generate.topK = body.at("top_k").get<int>(); }
  if (body.contains("seed") && body.at("seed").is_number()) { generate.seed = body.at("seed").get<uint64_t>(); }
  if (body.contains("stream") && body.at("stream").is_boolean()) { generate.stream = body.at("stream").get<bool>(); }
  if (body.contains("ignore_eos") && body.at("ignore_eos").is_boolean()) { generate.ignoreEos = body.at("ignore_eos").get<bool>(); }
  if (body.contains("stop"))
  {
    const auto &stop = body.at("stop");
    if (stop.is_string()) { generate.stop.push_back(stop.get<std::string>()); }
    else if (stop.is_array())
    {
      for (const auto &entry : stop)
      {
        if (entry.is_string()) { generate.stop.push_back(entry.get<std::string>()); }
      }
    }
  }
  return generate;
}

void sendJson(httplib::Response &response, const json &payload, int status = 200)
{
  response.status = status;
  response.set_content(payload.dump(), "application/json");
}

void sendError(httplib::Response &response, const std::string &message, int status)
{
  sendJson(response, json{{"error", {{"message", message}, {"type", "invalid_request_error"}}}}, status);
}

} // namespace

std::string mapFinishReason(const std::string &reason)
{
  static const std::map<std::string, std::string> mapping{{"FINISHED_EOS", "eos"}, {"FINISHED_LENGTH", "length"}, {"FINISHED_STOP", "stop"}, {"FINISHED_ABORTED", "aborted"}};
  const auto                                      it = mapping.find(reason);
  return it == mapping.end() ? "stop" : it->second;
}

HttpServer::HttpServer(ServerConfig config, engine::Engine &engine, const model::TokenizerAdapter &tokenizer, const model::ChatTemplate *chatTemplate)
  : _config(std::move(config)),
    _engine(engine),
    _tokenizer(tokenizer),
    _chatTemplate(chatTemplate),
    _server(std::make_unique<httplib::Server>())
{
  registerRoutes();
}

HttpServer::~HttpServer() { stop(); }

void HttpServer::start()
{
  if (_config.port == 0)
  {
    // bind_to_port(host, 0) reports success without telling us which port the
    // kernel chose; bind_to_any_port returns it.
    const int port = _server->bind_to_any_port(_config.host.c_str());
    if (port <= 0) { throw std::runtime_error("could not bind " + _config.host + " to an ephemeral port"); }
    _boundPort = port;
  }
  else
  {
    if (!_server->bind_to_port(_config.host.c_str(), _config.port)) { throw std::runtime_error("could not bind " + _config.host + ":" + std::to_string(_config.port)); }
    _boundPort = _config.port;
  }

  _thread = std::thread([this] { _server->listen_after_bind(); });
  // listen_after_bind returns only when the server stops, so wait for it to be
  // accepting rather than racing the first request against the accept loop.
  _server->wait_until_ready();
}

void HttpServer::stop()
{
  if (_server) { _server->stop(); }
  if (_thread.joinable()) { _thread.join(); }
}

void HttpServer::registerRoutes()
{
  _server->Get("/health", [this](const httplib::Request &, httplib::Response &response) {
    // 503 until the engine is genuinely up -- see the header note.
    if (!_engine.isReady())
    {
      sendJson(response, json{{"status", "not_ready"}}, 503);
      return;
    }
    sendJson(response, json{{"status", "ok"}});
  });

  _server->Get("/v1/models", [this](const httplib::Request &, httplib::Response &response) {
    sendJson(response, json{{"object", "list"}, {"data", json::array({json{{"id", _config.modelId}, {"object", "model"}, {"owned_by", "pypto"}}})}});
  });

  const auto completions = [this](const httplib::Request &request, httplib::Response &response, bool chat) {
    json body;
    try
    {
      body = json::parse(request.body);
    }
    catch (const std::exception &)
    {
      sendError(response, "request body is not valid JSON", 400);
      return;
    }

    // Templating and tokenization happen here, on the handler thread, so they
    // never block the engine loop.
    std::string prompt;
    if (chat)
    {
      if (!body.contains("messages") || !body.at("messages").is_array())
      {
        sendError(response, "chat completions require a 'messages' array", 400);
        return;
      }
      if (_chatTemplate == nullptr)
      {
        sendError(response, "this model has no chat template", 400);
        return;
      }
      try
      {
        prompt = _chatTemplate->apply(body.at("messages"), true);
      }
      catch (const std::exception &e)
      {
        sendError(response, std::string{"chat template failed: "} + e.what(), 400);
        return;
      }
    }
    else
    {
      if (!body.contains("prompt") || !body.at("prompt").is_string())
      {
        sendError(response, "completions require a 'prompt' string", 400);
        return;
      }
      prompt = body.at("prompt").get<std::string>();
    }

    const auto             generate     = parseGenerateConfig(body);
    const std::string      requestId    = _engine.generateRequestId();
    const std::vector<int> promptIds    = _tokenizer.encode(prompt);
    const std::string      model        = _config.modelId;
    const int              promptTokens = static_cast<int>(promptIds.size());

    std::shared_ptr<engine::RequestStream> stream;
    try
    {
      stream = _engine.addRequest(requestId, promptIds, generate);
    }
    catch (const std::exception &e)
    {
      // A prompt the scheduler will never admit is a client error, as it is in
      // the Python server.
      sendError(response, e.what(), 400);
      return;
    }

    const char *objectName = chat ? "chat.completion" : "text_completion";

    if (!generate.stream)
    {
      std::string text;
      std::string finishReason     = "stop";
      int         completionTokens = 0;
      while (auto update = stream->pop())
      {
        if (update->tokenId.has_value()) { completionTokens += 1; }
        text = update->text;
        if (update->finished) { finishReason = mapFinishReason(update->finishReason); }
      }

      json choice;
      if (chat) { choice = json{{"index", 0}, {"message", {{"role", "assistant"}, {"content", text}}}, {"finish_reason", finishReason}}; }
      else { choice = json{{"index", 0}, {"text", text}, {"finish_reason", finishReason}}; }

      sendJson(response,
               json{{"id", requestId},
                    {"object", objectName},
                    {"created", nowSeconds()},
                    {"model", model},
                    {"choices", json::array({choice})},
                    {"usage", {{"prompt_tokens", promptTokens}, {"completion_tokens", completionTokens}, {"total_tokens", promptTokens + completionTokens}}}});
      return;
    }

    // SSE. The chunk provider runs on this handler thread and drains the
    // request's own queue, so streaming one response never touches the engine.
    const std::string objectNameStr    = objectName;
    auto              completionTokens = std::make_shared<int>(0);
    auto              done             = std::make_shared<bool>(false);

    response.set_chunked_content_provider(
      "text/event-stream", [stream, requestId, model, objectNameStr, chat, promptTokens, completionTokens, done](size_t, httplib::DataSink &sink) {
        if (*done)
        {
          sink.done();
          return false;
        }

        auto update = stream->pop();
        if (!update.has_value())
        {
          const std::string tail = "data: [DONE]\n\n";
          sink.write(tail.data(), tail.size());
          *done = true;
          sink.done();
          return true;
        }

        if (update->tokenId.has_value()) { *completionTokens += 1; }

        json       choice;
        const json finishReason = update->finished ? json(mapFinishReason(update->finishReason)) : json(nullptr);
        if (chat) { choice = json{{"index", 0}, {"delta", {{"content", update->delta}}}, {"finish_reason", finishReason}}; }
        else { choice = json{{"index", 0}, {"text", update->delta}, {"finish_reason", finishReason}}; }

        const json        chunk{{"id", requestId}, {"object", objectNameStr}, {"created", nowSeconds()}, {"model", model}, {"choices", json::array({choice})}};
        const std::string payload = "data: " + chunk.dump() + "\n\n";
        sink.write(payload.data(), payload.size());

        if (update->finished)
        {
          // Terminal usage chunk: empty choices, authoritative counts. Same
          // shape as OpenAI's stream_options.include_usage.
          const json        usageChunk{{"id", requestId},
                                       {"object", objectNameStr},
                                       {"created", nowSeconds()},
                                       {"model", model},
                                       {"choices", json::array()},
                                       {"usage", {{"prompt_tokens", promptTokens}, {"completion_tokens", *completionTokens}, {"total_tokens", promptTokens + *completionTokens}}}};
          const std::string usagePayload = "data: " + usageChunk.dump() + "\n\n";
          sink.write(usagePayload.data(), usagePayload.size());
        }
        return true;
      });
  };

  _server->Post("/v1/completions", [completions](const httplib::Request &request, httplib::Response &response) { completions(request, response, false); });
  _server->Post("/v1/chat/completions", [completions](const httplib::Request &request, httplib::Response &response) { completions(request, response, true); });
}

} // namespace serving::server
