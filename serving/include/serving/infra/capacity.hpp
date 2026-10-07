#pragma once

/**
 * Where new capacity comes from (`scaling-pypto-serving.pptx`, M2).
 *
 * The deck names two sources: spare workers started at launch that sit idle
 * until a single message activates one, and a provider asked for nodes beyond
 * the local pool. Both answer the same question — *give me somewhere to put a
 * replica of this partition* — so both are a `CapacityProvider`, and a scaling
 * policy does not care which one it is talking to.
 *
 * The policy is serving's, not the control plane's: the deck is explicit that
 * the control plane executes actions and decides nothing. Nothing here decides
 * when to scale; it only answers what is available and hands it over.
 *
 * HiCR-free, like the rest of `infra/`. `LocalCapacity` needs nothing but the
 * map, so a single-node deployment and a test both scale without a cloud
 * underneath.
 */

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <serving/infra/state.hpp>

namespace serving::infra
{

/// What a scaling decision asks for.
struct CapacityRequest
{
  /// The partition the new replica would serve. Partitions scale on their own.
  PartitionId partition = 0;
  /// How many devices it needs wherever it lands.
  int devices = 1;
  /// The identity the replica will carry, so the map and the provider agree.
  std::string name;
  /// The port it should listen on. 0 leaves it to the provider.
  int port = 0;
};

/// Where a request landed.
struct Grant
{
  std::string      replica;
  std::string      node;
  std::vector<int> devices;
  int              port = 0;
  /// The provider's own handle, when it has one -- a HiCR instance id for a
  /// cloud-backed provider, 0 for one that only places on nodes already known.
  uint64_t instance = 0;

  /**
   * Set when the provider obtained a machine that was not in the map.
   *
   * A cloud grant is a node, not a card on one: that is the whole of the
   * deck's "wire new nodes into the application without a system restart".
   * The provider still does not edit the map -- it says what it got, and the
   * caller registers it, so one component owns the document.
   */
  std::optional<Node> newNode;

  bool operator==(const Grant &) const = default;
};

/**
 * Somewhere capacity comes from.
 *
 * `acquire` returns nullopt when there is nothing to give, which is a normal
 * answer: a policy that asks for more than exists should back off, not fail.
 */
class CapacityProvider
{
  public:

  virtual ~CapacityProvider() = default;

  [[nodiscard]] virtual std::optional<Grant> acquire(const CapacityRequest &request) = 0;

  /// Give it back. Releasing something never granted is a no-op, so a caller
  /// recovering from a half-finished scale-up does not have to track what stuck.
  virtual void release(const std::string &replica) = 0;

  /// How many more replicas of this shape could be acquired right now, for a
  /// policy that would rather look than fail.
  [[nodiscard]] virtual int available(const CapacityRequest &request) const = 0;

  /// For logs and tests.
  [[nodiscard]] virtual const char *kind() const = 0;
};

/**
 * Capacity on nodes the map already knows about.
 *
 * Places a replica on the first node with enough free devices. This is the
 * whole of elastic capacity for a single machine: eight cards is eight
 * replicas' worth, and nothing has to be asked for.
 *
 * It does not edit the map. A grant describes where a replica *would* go;
 * putting it there is the caller's, because the caller is what knows whether
 * the replica actually came up.
 */
class LocalCapacity final : public CapacityProvider
{
  public:

  explicit LocalCapacity(const StateMap &map)
    : _map(map)
  {}

  [[nodiscard]] std::optional<Grant> acquire(const CapacityRequest &request) override;
  void                               release(const std::string &replica) override;
  [[nodiscard]] int                  available(const CapacityRequest &request) const override;
  [[nodiscard]] const char          *kind() const override { return "local"; }

  private:

  const StateMap &_map;
  /// Devices handed out but not yet in the map, so two acquires in a row do not
  /// both get the same card. Cleared as the caller places them.
  std::vector<Grant> _outstanding;
};

} // namespace serving::infra
