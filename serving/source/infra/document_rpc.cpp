#include <serving/infra/document_rpc.hpp>

#include <cstring>

namespace serving::infra
{

DocumentServer::DocumentServer(HiCR::frontend::RPCEngine &rpc, HiCR::ComputeManager &compute, const StateMap &map)
  : _map(map)
{
  // Serialised inside the handler, not here: the document changes, and a
  // snapshot taken at registration would be wrong by the time anyone asked.
  auto unit = compute.createExecutionUnit([this, &rpc](void *) {
    _document = writeDocument(_map);
    _served += 1;
    rpc.submitReturnValue(const_cast<char *>(_document.data()), _document.size());
  });
  rpc.addRPCTarget(DOCUMENT_RPC, unit);
}

StateMap fetchDocument(HiCR::frontend::RPCEngine &rpc, HiCR::Instance::instanceId_t deployer)
{
  rpc.requestRPC(deployer, DOCUMENT_RPC);

  const auto returned = rpc.getReturnValue();
  if (returned == nullptr || returned->getPointer() == nullptr) { throw DocumentError("the deployer returned no document"); }

  const std::string text(static_cast<const char *>(returned->getPointer()), returned->getSize());
  return readDocument(text, "deployer");
}

} // namespace serving::infra
