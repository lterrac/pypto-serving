/**
 * pypto-serving-cpp: the serving binary.
 *
 * Flags follow pypto_serving/cli/main.py. --prompt generates offline;
 * otherwise it serves HTTP until SIGINT/SIGTERM. The model is loaded before
 * any thread starts.
 */

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <serving/engine/bridge_executor.hpp>
#include <serving/engine/engine.hpp>
#include <serving/model/chat_template.hpp>
#include <serving/model/hf_tokenizer.hpp>
#include <serving/server/http_server.hpp>
#include <serving/util/signals.hpp>

namespace
{

using Json = nlohmann::json;

struct Options
{
  std::string              modelDir;
  std::string              servedModelName;
  std::string              platform                  = "a2a3";
  int                      device                    = 0;
  int                      maxModelLen               = 1024;
  int                      blockSize                 = 128;
  int                      maxNumSeqs                = 16;
  int                      maxNumBatchedTokens       = 4096;
  int                      longPrefillTokenThreshold = 2048;
  bool                     enablePrefixCaching       = true;
  bool                     enableChunkedPrefill      = true;
  std::string              host                      = "0.0.0.0";
  int                      port                      = 8000;
  std::vector<std::string> prompts;
  std::string              generateConfigJson;
  std::string              repoRoot;
};

[[noreturn]] void usage(const char *argv0, int code)
{
  std::fprintf(code == 0 ? stdout : stderr,
               "usage: %s --model DIR [options]\n"
               "\n"
               "  --model DIR                  model directory (required)\n"
               "  --served-model-name NAME     name reported by the API (default: the dir name)\n"
               "  --platform NAME              NPU platform (default: a2a3)\n"
               "  --device N                   NPU device id (default: 0)\n"
               "  --max-model-len N            prompt + generated cap (default: 1024)\n"
               "  --block-size N               KV block size (default: 128)\n"
               "  --max-num-seqs N             concurrent requests (default: 16)\n"
               "  --max-num-batched-tokens N   tokens per step (default: 4096)\n"
               "  --long-prefill-token-threshold N   prefill chunk size (default: 2048)\n"
               "  --no-enable-prefix-caching   disable the prefix cache\n"
               "  --no-enable-chunked-prefill  disable chunked prefill\n"
               "  --host HOST                  bind address (default: 0.0.0.0)\n"
               "  --port N                     bind port (default: 8000)\n"
               "  --prompt TEXT                offline: generate for TEXT and exit (repeatable)\n"
               "  --generate-config JSON       {\"max_new_tokens\":..,\"temperature\":..}\n"
               "  --repo-root DIR              prepended to sys.path for the Python bridge\n",
               argv0);
  std::exit(code);
}

std::string basename(const std::string &path)
{
  auto trimmed = path;
  while (!trimmed.empty() && trimmed.back() == '/') { trimmed.pop_back(); }
  const auto slash = trimmed.find_last_of('/');
  return slash == std::string::npos ? trimmed : trimmed.substr(slash + 1);
}

Options parseArgs(int argc, char **argv)
{
  Options options;
  for (int i = 1; i < argc; ++i)
  {
    const std::string arg  = argv[i];
    const auto        next = [&](const char *name) -> std::string {
      if (i + 1 >= argc) { usage(argv[0], 2); }
      (void)name;
      return argv[++i];
    };

    if (arg == "--help" || arg == "-h") { usage(argv[0], 0); }
    else if (arg == "--model") { options.modelDir = next("model"); }
    else if (arg == "--served-model-name") { options.servedModelName = next("served-model-name"); }
    else if (arg == "--platform") { options.platform = next("platform"); }
    else if (arg == "--device") { options.device = std::stoi(next("device")); }
    else if (arg == "--max-model-len") { options.maxModelLen = std::stoi(next("max-model-len")); }
    else if (arg == "--block-size") { options.blockSize = std::stoi(next("block-size")); }
    else if (arg == "--max-num-seqs") { options.maxNumSeqs = std::stoi(next("max-num-seqs")); }
    else if (arg == "--max-num-batched-tokens") { options.maxNumBatchedTokens = std::stoi(next("max-num-batched-tokens")); }
    else if (arg == "--long-prefill-token-threshold") { options.longPrefillTokenThreshold = std::stoi(next("lp")); }
    else if (arg == "--no-enable-prefix-caching") { options.enablePrefixCaching = false; }
    else if (arg == "--no-enable-chunked-prefill") { options.enableChunkedPrefill = false; }
    else if (arg == "--host") { options.host = next("host"); }
    else if (arg == "--port") { options.port = std::stoi(next("port")); }
    else if (arg == "--prompt") { options.prompts.push_back(next("prompt")); }
    else if (arg == "--generate-config") { options.generateConfigJson = next("generate-config"); }
    else if (arg == "--repo-root") { options.repoRoot = next("repo-root"); }
    else
    {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      usage(argv[0], 2);
    }
  }
  if (options.modelDir.empty())
  {
    std::fprintf(stderr, "--model is required\n");
    usage(argv[0], 2);
  }
  if (options.servedModelName.empty()) { options.servedModelName = basename(options.modelDir); }
  return options;
}

serving::config::GenerateConfig parseGenerateConfig(const std::string &text)
{
  serving::config::GenerateConfig generate;
  if (text.empty()) { return generate; }
  const Json body = Json::parse(text);
  if (body.contains("max_new_tokens")) { generate.maxNewTokens = body.at("max_new_tokens").get<int>(); }
  if (body.contains("temperature")) { generate.temperature = body.at("temperature").get<double>(); }
  if (body.contains("top_p")) { generate.topP = body.at("top_p").get<double>(); }
  if (body.contains("top_k")) { generate.topK = body.at("top_k").get<int>(); }
  if (body.contains("ignore_eos")) { generate.ignoreEos = body.at("ignore_eos").get<bool>(); }
  return generate;
}

/// The model's chat template, if it ships one.
std::unique_ptr<serving::model::ChatTemplate> loadChatTemplate(const std::string &modelDir)
{
  try
  {
    return serving::model::loadChatTemplate(modelDir);
  }
  catch (const std::exception &e)
  {
    std::fprintf(stderr, "[warn] no usable chat template (%s); /v1/chat/completions will refuse\n", e.what());
    return nullptr;
  }
}

} // namespace

int runServing(const Options &options);

int main(int argc, char **argv)
{
  // Before any thread exists: threads inherit the mask, and a shutdown signal
  // delivered to one that is not waiting would terminate the process.
  serving::util::blockShutdownSignals();

  const Options options = parseArgs(argc, argv);
  try
  {
    return runServing(options);
  }
  catch (const std::exception &e)
  {
    // A model whose kernels disagree with the requested configuration is an
    // ordinary misconfiguration -- report it and exit, rather than letting an
    // uncaught exception core-dump the server.
    std::fprintf(stderr, "\n[serving] fatal: %s\n", e.what());
    return 1;
  }
}

int runServing(const Options &options)
{
  std::printf("[serving] model=%s device=%d platform=%s\n", options.modelDir.c_str(), options.device, options.platform.c_str());

  const auto tokenizer    = serving::model::HfTokenizer::fromModelDir(options.modelDir);
  const auto chatTemplate = loadChatTemplate(options.modelDir);

  serving::engine::BridgeConfig bridgeConfig;
  bridgeConfig.modelDir    = options.modelDir;
  bridgeConfig.deviceId    = options.device;
  bridgeConfig.platform    = options.platform;
  bridgeConfig.maxModelLen = options.maxModelLen;
  bridgeConfig.blockSize   = options.blockSize;
  bridgeConfig.maxNumSeqs  = options.maxNumSeqs;
  bridgeConfig.repoRoot    = options.repoRoot;
  serving::engine::BridgeExecutor executor(bridgeConfig);

  serving::engine::EngineConfig engineConfig;
  engineConfig.runtime.pageSize                    = options.blockSize;
  engineConfig.runtime.maxBatchSize                = options.maxNumSeqs;
  engineConfig.runtime.maxSeqLen                   = options.maxModelLen;
  engineConfig.scheduler.maxSeqLen                 = options.maxModelLen;
  engineConfig.scheduler.maxNumRunningReqs         = options.maxNumSeqs;
  engineConfig.scheduler.maxNumScheduledTokens     = options.maxNumBatchedTokens;
  engineConfig.scheduler.longPrefillTokenThreshold = options.longPrefillTokenThreshold;
  engineConfig.scheduler.enablePrefixCache         = options.enablePrefixCaching;
  engineConfig.scheduler.enableChunkPrefill        = options.enableChunkedPrefill;

  serving::engine::Engine engine(engineConfig, *tokenizer, executor);

  const auto startedAt = std::chrono::steady_clock::now();
  std::printf("[serving] loading the model (kernel compile + weight staging) ...\n");
  std::fflush(stdout);
  engine.start();
  const double startupSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - startedAt).count();
  std::printf("[serving] ready in %.1fs\n", startupSeconds);

  if (!options.prompts.empty())
  {
    const auto generate = parseGenerateConfig(options.generateConfigJson);
    int        status   = 0;
    for (size_t i = 0; i < options.prompts.size(); ++i)
    {
      const auto promptIds = tokenizer->encode(options.prompts[i]);
      const auto began     = std::chrono::steady_clock::now();
      const auto stream    = engine.addRequest(engine.generateRequestId(), promptIds, generate);

      std::string      text;
      std::vector<int> tokens;
      std::string      finishReason;
      while (auto update = stream->pop())
      {
        text = update->text;
        if (update->tokenId.has_value()) { tokens.push_back(*update->tokenId); }
        if (update->finished) { finishReason = update->finishReason; }
      }
      const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();

      std::printf("\n-- prompt %zu/%zu --\n", i + 1, options.prompts.size());
      std::printf("text: %s\n", text.c_str());
      std::printf("token_ids: [");
      for (size_t t = 0; t < tokens.size(); ++t) { std::printf("%s%d", t ? ", " : "", tokens[t]); }
      std::printf("]\n");
      std::printf("finish_reason: %s\n", finishReason.c_str());
      std::printf("[generate] %zu tokens in %.2fs (%.1f tok/s)\n", tokens.size(), elapsed, tokens.empty() ? 0.0 : static_cast<double>(tokens.size()) / elapsed);
      if (tokens.empty()) { status = 1; }
    }
    engine.stop();
    return status;
  }

  serving::server::ServerConfig serverConfig;
  serverConfig.host    = options.host;
  serverConfig.port    = options.port;
  serverConfig.modelId = options.servedModelName;
  serving::server::HttpServer server(serverConfig, engine, *tokenizer, chatTemplate.get());
  server.start();
  std::printf("[serving] listening on %s:%d\n", options.host.c_str(), server.boundPort());
  std::fflush(stdout);

  // Not stdin: a server launched by a job runner usually has none, and reading
  // it would exit the moment the pipe closed.
  serving::util::awaitShutdownSignal();
  std::printf("[serving] shutting down\n");
  std::fflush(stdout);

  // Engine first: it closes the request streams the HTTP workers are parked on.
  engine.stop();
  server.stop();
  return 0;
}
