#pragma once

/**
 * OpenAI-compatible HTTP: /v1/completions, /v1/chat/completions, /v1/models,
 * /health (503 until the engine is ready).
 *
 * Parsing, templating and tokenization run on the handler thread; SSE drains
 * the request's stream.
 */

#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include <serving/engine/engine.hpp>
#include <serving/model/chat_template.hpp>
#include <serving/model/tokenizer.hpp>

namespace httplib
{
class Server;
}

namespace serving::server
{

struct ServerConfig
{
  std::string host = "0.0.0.0";
  /// 0 binds an ephemeral port; read it back with `boundPort()`.
  int         port    = 8000;
  std::string modelId = "model";
};

/// Map a scheduler finish reason onto the OpenAI vocabulary.
[[nodiscard]] std::string mapFinishReason(const std::string &reason);

class HttpServer
{
  public:

  HttpServer(ServerConfig config, engine::Engine &engine, const model::TokenizerAdapter &tokenizer, const model::ChatTemplate *chatTemplate = nullptr);
  ~HttpServer();

  HttpServer(const HttpServer &)            = delete;
  HttpServer &operator=(const HttpServer &) = delete;

  /// Bind, then serve on a background thread. Returns once the port is listening.
  void start();

  void stop();

  /// The port actually bound, which matters when config.port was 0.
  [[nodiscard]] int boundPort() const { return _boundPort.load(); }

  private:

  void registerRoutes();

  ServerConfig                   _config;
  engine::Engine                &_engine;
  const model::TokenizerAdapter &_tokenizer;
  const model::ChatTemplate     *_chatTemplate;

  std::unique_ptr<httplib::Server> _server;
  std::thread                      _thread;
  std::atomic<int>                 _boundPort{0};
};

} // namespace serving::server
