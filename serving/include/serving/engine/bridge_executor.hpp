#pragma once

/**
 * A ModelExecutor that embeds CPython and calls pypto_serving.bridge; a step
 * crosses as JSON.
 *
 * registerModel() loads the model, during which simpler forks one child per
 * chip, so it must run while the process is single-threaded; Engine::start()
 * guarantees that. Compiled only with the pythonBridge feature.
 */

#include <memory>
#include <string>

#include <serving/engine/executor.hpp>

namespace serving::engine
{

struct BridgeConfig
{
  std::string modelDir;
  int         deviceId      = 0;
  std::string platform      = "a2a3";
  int         maxModelLen   = 512;
  int         blockSize     = 128;
  int         maxNumSeqs    = 16;
  std::string pyptoBuildDir = "build_output/bridge";
  /// Prepended to sys.path so `import pypto_serving.bridge` resolves.
  std::string repoRoot;
};

class BridgeExecutor : public ModelExecutor
{
  public:

  explicit BridgeExecutor(BridgeConfig config);
  ~BridgeExecutor() override;

  BridgeExecutor(const BridgeExecutor &)            = delete;
  BridgeExecutor &operator=(const BridgeExecutor &) = delete;

  int        registerModel() override;
  StepResult executeStep(const StepCommand &command) override;
  void       close() override;

  /// KV tokens per block, as the model layer reports it.
  [[nodiscard]] int pageSize() const { return _pageSize; }

  private:

  struct Impl;
  std::unique_ptr<Impl> _impl;
  BridgeConfig          _config;
  int                   _pageSize = 0;
};

} // namespace serving::engine
