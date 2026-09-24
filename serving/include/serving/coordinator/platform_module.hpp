#pragma once

/**
 * Coordinator bound to serving::modules::Module. Compiled only with the
 * platform feature; Coordinator itself is HiCR-free.
 */

#include <memory>
#include <utility>

#include <modules/module.hpp>

#include <serving/coordinator/module.hpp>

namespace serving::coordinator
{

/// Default service period. The coordinator's tick only drains replica-loss
/// reports, so it wants to be responsive without busy-waiting.
inline constexpr size_t DEFAULT_SERVICE_INTERVAL_MS = 100;

class CoordinatorModule final : public modules::Module
{
  public:

  CoordinatorModule(CoordinatorConfig config, router::RuleTransport *transport, const size_t intervalMs = DEFAULT_SERVICE_INTERVAL_MS)
    : modules::Module(intervalMs),
      _coordinator(std::make_unique<Coordinator>(std::move(config), transport))
  {}

  void initialize() override { _coordinator->initialize(); }

  /// Nothing to start: the coordinator is driven by requests and by its service
  /// tick, both of which the engine already owns.
  void run() override {}

  void terminate() override {}

  void await() override {}

  void finalize() override { _coordinator->finalize(); }

  [[nodiscard]] Coordinator       &coordinator() { return *_coordinator; }
  [[nodiscard]] const Coordinator &coordinator() const { return *_coordinator; }

  protected:

  void service() override { _coordinator->service(); }

  private:

  std::unique_ptr<Coordinator> _coordinator;
};

} // namespace serving::coordinator
