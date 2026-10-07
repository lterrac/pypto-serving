#pragma once

/**
 * Sharing the deployment document from the node that owns it.
 *
 * There is one copy, on the deployer -- the node that started the application
 * and is known to everyone. Coordinators on other nodes do not keep their own
 * and do not replicate it; they ask for a copy. One writer, one truth, and no
 * consistency problem to solve.
 *
 * The request goes over HiCR's RPC engine, which is the control plane's own
 * fabric, so this stays off the per-token path like everything else here.
 *
 * Compiled only when HiCR is available (it arrives with the `cloudr` feature).
 * `infra/document.hpp` itself stays HiCR-free: the document is JSON either
 * way, and a single-node deployment reads it off disk.
 */

#include <string>

#include <hicr/core/instance.hpp>
#include <hicr/frontends/RPCEngine/RPCEngine.hpp>

#include <serving/infra/document.hpp>
#include <serving/infra/state.hpp>

namespace serving::infra
{

/// The RPC the deployer answers.
inline constexpr const char *DOCUMENT_RPC = "serving.deployment.document";

/**
 * Serves the deployer's document.
 *
 * Holds a reference, not a copy, so a caller always gets the map as it is now
 * rather than as it was when the server was built -- the document moves, and a
 * stale answer would be worse than none.
 */
class DocumentServer
{
  public:

  /// Registers the RPC target. `map` must outlive this.
  DocumentServer(HiCR::frontend::RPCEngine &rpc, HiCR::ComputeManager &compute, const StateMap &map);

  /// Documents handed out, so a deployer can say whether anyone has asked.
  [[nodiscard]] int served() const { return _served; }

  private:

  const StateMap &_map;
  /// The serialised copy, kept alive until the RPC engine has sent it.
  std::string _document;
  int         _served = 0;
};

/**
 * Ask the deployer for the document.
 *
 * Throws `DocumentError` if the answer is missing or unparseable: a
 * coordinator that cannot read the deployment must not carry on with an empty
 * map, because an empty map routes nowhere and says so only much later.
 */
[[nodiscard]] StateMap fetchDocument(HiCR::frontend::RPCEngine &rpc, HiCR::Instance::instanceId_t deployer);

} // namespace serving::infra
