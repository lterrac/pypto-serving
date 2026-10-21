#pragma once

/**
 * OpenAI-compatible HTTP: /v1/completions, /v1/chat/completions, /v1/models,
 * /health (503 until the engine is ready).
 *
 * Parsing, templating and tokenization run on the handler thread; SSE drains
 * the request's stream.
 *
 * Given a `replica::RuleTable` it also speaks the deck's Option B: a
 * coordinator publishes to `/control/rules` where this session goes next, and
 * a request carrying that session is forwarded there rather than served here
 * -- without asking the coordinator, which is what keeps it off the per-token
 * path. Without a table the control routes are not registered at all and the
 * replica behaves exactly as before.
 */

#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include <serving/engine/engine.hpp>
#include <serving/model/chat_template.hpp>
#include <serving/model/tokenizer.hpp>
#include <serving/replica/rule_table.hpp>

namespace httplib
{
class Server;
struct Request;
struct Response;
} // namespace httplib

namespace serving::server
{

struct ServerConfig
{
  std::string host = "0.0.0.0";
  /// 0 binds an ephemeral port; read it back with `boundPort()`.
  int         port    = 8000;
  std::string modelId = "model";

  /// What a request gets for every field it does not set itself -- the
  /// operator's `--generate-config`. Without this the flag configured the
  /// one-shot `--prompt` path and nothing served over HTTP, so the same
  /// deployment answered differently depending on how it was asked.
  config::GenerateConfig defaults;
};

/// Map a scheduler finish reason onto the OpenAI vocabulary.
[[nodiscard]] std::string mapFinishReason(const std::string &reason);

class HttpServer
{
  public:

  /// `rules` is optional; without one this replica never forwards, and the
  /// control endpoints do not exist.
  HttpServer(ServerConfig                   config,
             engine::Engine                &engine,
             const model::TokenizerAdapter &tokenizer,
             const model::ChatTemplate     *chatTemplate = nullptr,
             replica::RuleTable            *rules        = nullptr);
  ~HttpServer();

  HttpServer(const HttpServer &)            = delete;
  HttpServer &operator=(const HttpServer &) = delete;

  /// Bind, then serve on a background thread. Returns once the port is listening.
  void start();

  void stop();

  /// The port actually bound, which matters when config.port was 0.
  [[nodiscard]] int boundPort() const { return _boundPort.load(); }

  /// Requests this replica forwarded to a next hop rather than serving.
  [[nodiscard]] int forwarded() const { return _forwarded.load(); }

  private:

  void registerRoutes();
  void registerControlRoutes();
  /// Relay this request to `hop`. True when it was handled here.
  bool forwardTo(const replica::NextHop &hop, const httplib::Request &request, httplib::Response &response, const std::string &path, const std::string &sessionId);

  ServerConfig                   _config;
  engine::Engine                &_engine;
  const model::TokenizerAdapter &_tokenizer;
  const model::ChatTemplate     *_chatTemplate;
  /// Locks itself: handler threads read it while a control route -- or, in a
  /// channel deployment, the coordinator's rule module -- writes it.
  replica::RuleTable *_rules;
  std::atomic<int>    _forwarded{0};

  std::unique_ptr<httplib::Server> _server;
  std::thread                      _thread;
  /// Exchanged by stop(), so a destructor and an explicit stop cannot both
  /// join the same thread.
  std::atomic<bool>                _serving{false};
  std::atomic<int>                 _boundPort{0};
};

} // namespace serving::server
