#pragma once

/**
 * Capacity from CloudR, the HiCR cloud emulator.
 *
 * The deck's second source: *beyond the local pool, ask the provider for
 * additional nodes*. CloudR emulates instances with arbitrary topologies on
 * one machine, so a deployment can be scaled onto "new nodes" without a second
 * machine to put them on -- which is the only way this path is testable at all
 * on a single box.
 *
 * A grant here is a **node**, not a card on an existing one, and it carries
 * the HiCR instance id so the control plane can address what it was given.
 *
 * Compiled only when the `cloudr` feature is on; `infra/capacity.hpp` stays
 * HiCR-free, the way `coordinator/module.hpp` does, so everything that merely
 * scales keeps building without MPI.
 */

#include <map>
#include <memory>
#include <string>

#include <hicr/backends/cloudr/instanceManager.hpp>

#include <serving/infra/capacity.hpp>

namespace serving::infra
{

/**
 * Asks CloudR for an instance per grant.
 *
 * The emulated instance stands in for a machine: `devicesPerNode` says how many
 * accelerators that machine is to be described as having, because CloudR models
 * host topology and knows nothing about Ascend cards. That is a deliberate
 * stand-in and not a measurement -- what is being exercised here is the control
 * plane's ability to take delivery of a node it did not have, not the node.
 */
class CloudrCapacity final : public CapacityProvider
{
  public:

  CloudrCapacity(HiCR::backend::cloudr::InstanceManager &instances, int devicesPerNode = 1, std::string namePrefix = "cloud");

  [[nodiscard]] std::optional<Grant> acquire(const CapacityRequest &request) override;

  /// Terminates the instance the replica was placed on. A node with nothing on
  /// it is given back: an emulated machine still costs a real process.
  void release(const std::string &replica) override;

  /// How many more instances the emulator will create. CloudR has no quota of
  /// its own, so this is the deployment's own ceiling.
  [[nodiscard]] int available(const CapacityRequest &request) const override;

  [[nodiscard]] const char *kind() const override { return "cloudr"; }

  /// Instances created and not yet released.
  [[nodiscard]] int outstanding() const { return static_cast<int>(_held.size()); }

  /// Caps how many instances will be asked for, so a runaway policy cannot
  /// fork the emulator until the machine falls over.
  void setLimit(int limit) { _limit = limit; }

  private:

  HiCR::backend::cloudr::InstanceManager &_instances;
  int                                     _devicesPerNode;
  std::string                             _prefix;
  int                                     _limit    = 16;
  int                                     _sequence = 0;

  /// Replica name to the instance serving it, so release knows what to kill.
  std::map<std::string, std::shared_ptr<HiCR::Instance>> _held;
};

} // namespace serving::infra
