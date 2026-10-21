#include <serving/server/http_server.hpp>

#include <serving/router/proxy.hpp>
#include <serving/util/http_relay.hpp>
#include <serving/util/http_serving.hpp>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <thread>
#include <map>
#include <stdexcept>

#include <httplib.h>
#include <nlohmann/json.hpp>

namespace serving::server
{

namespace
{

using nlohmann::json;

/// Hops a request may be forwarded through before the chain is called a cycle.
/// A pipeline is a handful of stages deep; anything past that is a loop.
constexpr int MAX_FORWARD_DEPTH = 8;
/// Chunks held between the hop and the client before the reader is made to
/// wait. Bounded so a client slower than the hop cannot grow the queue without
/// limit; the same bound the router's proxy uses.
constexpr size_t MAX_CHUNKS_IN_FLIGHT = 256;
/// Silence from the next hop that means it is gone. Between chunks, not across
/// the answer: a generation longer than this is ordinary.
constexpr double NEXT_HOP_IDLE_SECONDS = 300.0;
/// Carries the count above between replicas.
constexpr const char *FORWARD_DEPTH_HEADER = "x-serving-forward-depth";

long nowSeconds() { return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count(); }

/// Pull a GenerateConfig out of a request body, falling back to the server's
/// own defaults for every field the request did not set.
config::GenerateConfig parseGenerateConfig(const json &body, const config::GenerateConfig &defaults)
{
  config::GenerateConfig generate = defaults;
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

/**
 * The template-specific keyword arguments a chat request carries.
 *
 * Applied in the Python server's order (`server.py:_apply_chat_template`):
 * `chat_template_kwargs` first, then `reasoning_effort`, which only supplies
 * `enable_thinking` when the caller did not set it explicitly. A request
 * carrying neither renders exactly as one that omits both.
 *
 * Returns an error message, empty when `context` was filled.
 */
std::string buildChatTemplateContext(const json &body, nlohmann::ordered_json &context)
{
  context = nlohmann::ordered_json::object();

  if (body.contains("chat_template_kwargs") && !body.at("chat_template_kwargs").is_null())
  {
    const auto &kwargs = body.at("chat_template_kwargs");
    if (!kwargs.is_object()) { return "chat_template_kwargs must be an object"; }
    for (const auto &entry : kwargs.items()) { context[entry.key()] = nlohmann::ordered_json(entry.value()); }
  }

  if (body.contains("reasoning_effort") && !body.at("reasoning_effort").is_null())
  {
    const auto &effort = body.at("reasoning_effort");
    if (!effort.is_string()) { return "reasoning_effort must be a string"; }
    const auto value = effort.get<std::string>();
    if (value == "none")
    {
      // A hard override, not a default. Asking for no reasoning and getting a
      // <think> block because chat_template_kwargs also mentioned thinking is
      // the opposite of what was asked for. `thinking` is the second key the
      // DeepSeek template reads, and it has to be written too.
      context["enable_thinking"] = false;
      context["thinking"]        = false;
    }
    // setdefault otherwise: an explicit enable_thinking in chat_template_kwargs
    // wins over an effort level that merely implies one.
    else if (!context.contains("enable_thinking")) { context["enable_thinking"] = true; }
    context["reasoning_effort"] = value;
  }

  return {};
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
  static const std::map<std::string, std::string> mapping{
    {"FINISHED_EOS", "eos"}, {"FINISHED_LENGTH", "length"}, {"FINISHED_STOP", "stop"}, {"FINISHED_ABORTED", "aborted"}, {"FINISHED_ERROR", "error"}};
  const auto it = mapping.find(reason);
  return it == mapping.end() ? "stop" : it->second;
}

HttpServer::HttpServer(ServerConfig config, engine::Engine &engine, const model::TokenizerAdapter &tokenizer, const model::ChatTemplate *chatTemplate, replica::RuleTable *rules)
  : _config(std::move(config)),
    _engine(engine),
    _tokenizer(tokenizer),
    _chatTemplate(chatTemplate),
    _rules(rules),
    _server(std::make_unique<httplib::Server>())
{
  registerRoutes();
}

HttpServer::~HttpServer() { stop(); }

void HttpServer::start()
{
  _boundPort = util::bindAndServe(*_server, _config.host, _config.port, _thread, "serving");
  _serving   = true;
}

void HttpServer::stop() { util::stopServing(_server.get(), _thread, _serving); }

/// The coordinator's half of Option B, over HTTP: it publishes where a session
/// goes next and withdraws it when the path is replanned. A deployment with a
/// control plane puts the same messages on a channel; the table is the same
/// either way.
void HttpServer::registerControlRoutes()
{
  _server->Post("/control/rules", [this](const httplib::Request &request, httplib::Response &response) {
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
    if (!body.contains("session_id") || !body.at("session_id").is_string() || !body.contains("next_hop") || !body.at("next_hop").is_string())
    {
      sendError(response, "a rule needs 'session_id' and 'next_hop'", 400);
      return;
    }

    coordinator::RulePublish rule;
    rule.sessionId       = body.at("session_id").get<std::string>();
    rule.nextHopName     = body.at("next_hop").get<std::string>();
    rule.nextHopInstance = body.value("next_hop_instance", 0ULL);
    rule.generation      = body.value("generation", 0ULL);
    const auto endpoint  = body.value("next_hop_url", std::string{});

    const replica::RuleOutcome outcome = _rules->apply(rule, endpoint);
    // Reported rather than swallowed: a coordinator whose rule was ignored as
    // stale needs to know it did not take.
    sendJson(response, json{{"outcome", replica::ruleOutcomeName(outcome)}, {"session_id", rule.sessionId}});
  });

  _server->Post("/control/rules/revoke", [this](const httplib::Request &request, httplib::Response &response) {
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
    coordinator::RuleRevoke revoke;
    revoke.sessionId  = body.value("session_id", std::string{});
    revoke.generation = body.value("generation", 0ULL);

    const replica::RuleOutcome outcome = _rules->revoke(revoke);
    sendJson(response, json{{"outcome", replica::ruleOutcomeName(outcome)}, {"session_id", revoke.sessionId}});
  });

  // What this replica believes: a forwarding decision nobody can inspect is
  // one nobody can debug.
  _server->Get("/control/rules", [this](const httplib::Request &request, httplib::Response &response) {
    const std::string               session = request.get_param_value("session");
    const std::optional<replica::NextHop> hop = _rules->peek(session);
    if (!hop.has_value())
    {
      sendJson(response, json{{"session_id", session}, {"next_hop", nullptr}});
      return;
    }
    sendJson(response,
             json{{"session_id", session}, {"next_hop", hop->name}, {"next_hop_url", hop->endpoint}, {"next_hop_instance", hop->instance}, {"generation", hop->generation}});
  });
}

bool HttpServer::forwardTo(const replica::NextHop &hop, const httplib::Request &request, httplib::Response &response, const std::string &path, const std::string &sessionId)
{
  // Addressed by instance with no HTTP endpoint: a control plane would route
  // this and this replica cannot. Serving it here is the honest fallback.
  if (hop.endpoint.empty()) { return false; }

  // A rule naming this replica, or a cycle through a sibling, would otherwise
  // recurse over HTTP until the thread and socket pools are gone. Revokes are
  // best-effort, so a replica that missed one can point back at a replica whose
  // replanned rule points here -- reachable without anyone meaning it.
  const int depth = request.has_header(FORWARD_DEPTH_HEADER) ? std::atoi(request.get_header_value(FORWARD_DEPTH_HEADER).c_str()) : 0;
  if (depth >= MAX_FORWARD_DEPTH)
  {
    sendError(response, "forwarding for this session exceeded " + std::to_string(MAX_FORWARD_DEPTH) + " hops; the routing rules form a cycle", 508);
    return true;
  }

  std::string host   = hop.endpoint;
  std::string scheme = "http";
  int         port   = 80;
  if (const auto mark = host.find("://"); mark != std::string::npos)
  {
    scheme = host.substr(0, mark);
    host   = host.substr(mark + 3);
  }
  // Refused rather than downgraded. This build links no TLS, so dialling an
  // https next hop over a plain client would put prompts and generated text on
  // the wire in clear -- the exact thing ReplicaSpec::scheme exists to prevent.
  // Saying so beats doing it quietly.
  if (scheme != "http")
  {
    sendError(response, "next hop " + hop.name + " is published as '" + scheme + "', which this build cannot dial; it speaks http only", 502);
    return true;
  }
  if (const auto colon = host.rfind(':'); colon != std::string::npos)
  {
    try
    {
      port = std::stoi(host.substr(colon + 1));
      host = host.substr(0, colon);
    }
    catch (const std::exception &)
    {
      return false;
    }
  }

  // Relayed as it arrives, not buffered. The forward decision is taken before
  // the stream flag is ever read, so a `"stream": true` request whose session
  // has a rule came through here: a buffering client accumulates the whole SSE
  // body and replays it in one write, which turns time-to-first-token into
  // time-to-last-token, and its read timeout is wall-clock over the whole body
  // so a long generation returns 502 with the answer already discarded.
  //
  // The same relay the router's proxy uses. What is this side's own is the one
  // fixed next hop and the forwarding-depth header; there is no retry, because
  // a rule names one replica and there is no sibling to fall back to.
  httplib::Headers headers;
  headers.emplace(std::string(router::SESSION_HEADER), sessionId);
  headers.emplace(std::string(FORWARD_DEPTH_HEADER), std::to_string(depth + 1));

  const util::Relay relay = util::startRelay(host, port, path, request.body, headers, 5.0, NEXT_HOP_IDLE_SECONDS, MAX_CHUNKS_IN_FLIGHT);

  auto queue           = relay.queue;
  auto upstreamStatus  = relay.status;
  auto transportFailed = relay.transportFailed;
  std::string contentType = "application/json";
  if (const auto it = relay.headers->find("Content-Type"); it != relay.headers->end()) { contentType = it->second; }

  if (upstreamStatus->load() == 0)
  {
    // Nothing answered. Falling back to serving locally would answer from the
    // wrong stage of the pipeline, so the caller has to see this.
    queue->close();
    sendError(response, "next hop " + hop.name + " is unreachable", 502);
    return true;
  }

  _forwarded.fetch_add(1);
  response.status = upstreamStatus->load();
  response.set_header(std::string(router::SESSION_HEADER).c_str(), sessionId.c_str());
  response.set_header("x-forwarded-by", _config.modelId.c_str());

  // Destroying the provider closes the queue however the relay ends, so a
  // client that disappears between chunks cannot leave the upstream thread
  // blocked on a full queue for ever.
  std::shared_ptr<void> closeOnDestroy(nullptr, [queue](void *) { queue->close(); });

  response.set_chunked_content_provider(contentType, [queue, transportFailed, closeOnDestroy](size_t, httplib::DataSink &sink) {
    auto chunk = queue->pop();
    if (!chunk.has_value())
    {
      // The hop died part-way through its answer. Ending the chunked body here
      // would frame a short answer as a complete one, so the connection is
      // dropped instead and every client sees a truncated transfer.
      if (transportFailed->load()) { return false; }
      sink.done();
      return true;
    }
    if (!sink.write(chunk->data(), chunk->size()))
    {
      queue->close();
      return false;
    }
    return true;
  });
  return true;
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
    // Up is not the same as working. A hung kernel leaves the loop thread
    // blocked without it exiting, so a check that asks only whether the engine
    // exists answers 200 for ever and the monitor never marks the replica
    // dead -- the recovery path then never runs, and every request pinned here
    // waits out its full timeout instead.
    if (!_engine.isHealthy())
    {
      sendJson(response, json{{"status", "stalled"}, {"step_seconds", _engine.stepElapsedSeconds()}}, 503);
      return;
    }
    sendJson(response, json{{"status", "ok"}});
  });

  // What this replica is carrying, for whoever decides about scaling. Not
  // /metrics: upstream serves Prometheus there, and this is three numbers for
  // a control plane rather than an observability surface.
  _server->Get("/control/load", [this](const httplib::Request &, httplib::Response &response) {
    sendJson(response,
             json{{"replica", _config.modelId},
                  {"ready", _engine.isReady()},
                  // Admitted and not yet generated: the queue a scaling
                  // decision actually cares about.
                  // Requests in flight: what the scaling thresholds are
                  // expressed in. Without it the control plane reads every
                  // replica as idle.
                  {"outstanding", _engine.outstandingRequests()},
                  {"pending_tokens", _engine.pendingTokenLoad()},
                  // Rising means the KV pool cannot hold what was admitted, so
                  // this replica is already failing to serve what it took.
                  {"preemptions", _engine.preemptions()},
                  {"forwarded", _forwarded.load()}});
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

    // Option B: if a coordinator has said where this session goes next, it
    // goes there. No round trip to the coordinator -- that is the whole point
    // of publishing the rule ahead of the request.
    if (_rules != nullptr)
    {
      const std::string               sessionId = router::resolveSessionId(request, request.body);
      const std::optional<replica::NextHop> hop = _rules->lookup(sessionId);
      if (hop.has_value() && forwardTo(*hop, request, response, chat ? "/v1/chat/completions" : "/v1/completions", sessionId)) { return; }
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
      nlohmann::ordered_json extraContext;
      if (const auto error = buildChatTemplateContext(body, extraContext); !error.empty())
      {
        sendError(response, error, 400);
        return;
      }
      try
      {
        prompt = _chatTemplate->apply(body.at("messages"), true, extraContext);
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

    // Both endpoints honour EOS unless the caller or the operator says
    // otherwise. The Python forces ignore_eos on /v1/completions
    // (server.py:312) and calls it a historic override in its own comment; it
    // is not reproduced here. Ignoring the model's end-of-sequence token pads
    // every completion out to max_tokens and reports finish_reason "length"
    // where "stop" is the truth, which is wrong for an OpenAI-compatible
    // endpoint whatever its history. A caller who wants the old behaviour asks
    // for it with ignore_eos, and an operator can set it for the whole server
    // through --generate-config.
    const auto generate = parseGenerateConfig(body, _config.defaults);
    if (generate.topK.has_value() && *generate.topK <= 0)
    {
      sendError(response, "top_k must be positive", 400);
      return;
    }
    if (generate.topP <= 0.0 || generate.topP > 1.0)
    {
      sendError(response, "top_p must be in (0, 1]", 400);
      return;
    }
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
      std::string finishReason = "stop";
      std::string error;
      bool        finishedCleanly  = false;
      int         completionTokens = 0;
      while (auto update = stream->pop())
      {
        if (update->tokenId.has_value()) { completionTokens += 1; }
        if (update->finished)
        {
          text            = update->text;
          finishReason    = mapFinishReason(update->finishReason);
          error           = update->error;
          finishedCleanly = update->error.empty();
        }
      }
      // A step that failed, or a stream closed without finishing, is not a
      // successful completion: reporting it as one hides the failure entirely.
      if (!error.empty())
      {
        sendError(response, error, 500);
        return;
      }
      if (!finishedCleanly)
      {
        sendError(response, "request did not complete", 500);
        return;
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
    // The same four fields on every chunk; only choices and usage differ.
    const json envelope{{"id", requestId}, {"object", std::string(objectName)}, {"created", nowSeconds()}, {"model", model}};
    auto       completionTokens = std::make_shared<int>(0);
    auto       done             = std::make_shared<bool>(false);

    // httplib stops calling the provider once the peer is gone, so the only
    // reliable hook is the destruction of this closure: without it a client
    // that hangs up leaves the request decoding to max_tokens, holding a decode
    // slot and its KV blocks.
    engine::Engine       *engine = &_engine;
    std::shared_ptr<void> abortIfUnfinished(nullptr, [engine, requestId, done](void *) {
      if (!*done) { engine->abortRequest(requestId); }
    });

    response.set_chunked_content_provider(
      "text/event-stream", [stream, requestId, envelope, chat, promptTokens, completionTokens, done, engine, abortIfUnfinished](size_t, httplib::DataSink &sink) {
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

        if (!update->error.empty())
        {
          const std::string payload = "data: " + json{{"error", {{"message", update->error}, {"type", "server_error"}}}}.dump() + "\n\n";
          sink.write(payload.data(), payload.size());
          const std::string tail = "data: [DONE]\n\n";
          sink.write(tail.data(), tail.size());
          *done = true;
          sink.done();
          return true;
        }

        json       choice;
        const json finishReason = update->finished ? json(mapFinishReason(update->finishReason)) : json(nullptr);
        if (chat) { choice = json{{"index", 0}, {"delta", {{"content", update->delta}}}, {"finish_reason", finishReason}}; }
        else { choice = json{{"index", 0}, {"text", update->delta}, {"finish_reason", finishReason}}; }

        json chunk                = envelope;
        chunk["choices"]          = json::array({choice});
        const std::string payload = "data: " + chunk.dump() + "\n\n";
        if (!sink.write(payload.data(), payload.size()))
        {
          // The client is gone; stop generating for it.
          engine->abortRequest(requestId);
          *done = true;
          return false;
        }

        if (update->finished)
        {
          // Terminal usage chunk: empty choices, authoritative counts. Same
          // shape as OpenAI's stream_options.include_usage.
          json usageChunk                = envelope;
          usageChunk["choices"]          = json::array();
          usageChunk["usage"]            = {{"prompt_tokens", promptTokens}, {"completion_tokens", *completionTokens}, {"total_tokens", promptTokens + *completionTokens}};
          const std::string usagePayload = "data: " + usageChunk.dump() + "\n\n";
          sink.write(usagePayload.data(), usagePayload.size());
        }
        return true;
      });
  };

  _server->Post("/v1/completions", [completions](const httplib::Request &request, httplib::Response &response) { completions(request, response, false); });
  _server->Post("/v1/chat/completions", [completions](const httplib::Request &request, httplib::Response &response) { completions(request, response, true); });

  if (_rules != nullptr) { registerControlRoutes(); }
}

} // namespace serving::server
