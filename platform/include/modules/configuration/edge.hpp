#pragma once

/**
 * An edge of the deployment graph: one channel between two modules.
 *
 * `system/channels/{base,input,output}.hpp` take one of these and read the
 * managers and memory spaces a HiCR channel needs out of it. It is
 * configuration only -- it owns nothing and allocates nothing, so an edge can
 * be described before the instances it joins exist.
 *
 * Two sets of managers rather than one: a variable-size channel keeps its
 * payload and its coordination metadata in separate buffers, and a deployment
 * may want them in different memory. Passing the same manager twice is the
 * ordinary case.
 */

#include <memory>
#include <string>
#include <utility>

#include <hicr/core/communicationManager.hpp>
#include <hicr/core/memoryManager.hpp>
#include <hicr/core/memorySpace.hpp>

namespace serving::configuration
{

class Edge final
{
  public:

  Edge() = default;

  /// `bufferSize` is the payload buffer in bytes; `bufferCapacity` the number
  /// of messages that may be in flight before a producer blocks.
  Edge(std::string                        name,
       size_t                             bufferCapacity,
       size_t                             bufferSize,
       HiCR::CommunicationManager        *payloadCommunicationManager,
       HiCR::MemoryManager               *payloadMemoryManager,
       std::shared_ptr<HiCR::MemorySpace> payloadMemorySpace,
       HiCR::CommunicationManager        *coordinationCommunicationManager,
       HiCR::MemoryManager               *coordinationMemoryManager,
       std::shared_ptr<HiCR::MemorySpace> coordinationMemorySpace)
    : _name(std::move(name)),
      _bufferCapacity(bufferCapacity),
      _bufferSize(bufferSize),
      _payloadCommunicationManager(payloadCommunicationManager),
      _payloadMemoryManager(payloadMemoryManager),
      _payloadMemorySpace(std::move(payloadMemorySpace)),
      _coordinationCommunicationManager(coordinationCommunicationManager),
      _coordinationMemoryManager(coordinationMemoryManager),
      _coordinationMemorySpace(std::move(coordinationMemorySpace))
  {}

  /// One manager and one memory space for both halves, which is what a
  /// single-host deployment wants.
  Edge(std::string                        name,
       size_t                             bufferCapacity,
       size_t                             bufferSize,
       HiCR::CommunicationManager        *communicationManager,
       HiCR::MemoryManager               *memoryManager,
       std::shared_ptr<HiCR::MemorySpace> memorySpace)
    : Edge(std::move(name), bufferCapacity, bufferSize, communicationManager, memoryManager, memorySpace, communicationManager, memoryManager, std::move(memorySpace))
  {}

  [[nodiscard]] const std::string &getName() const { return _name; }
  [[nodiscard]] size_t             getBufferCapacity() const { return _bufferCapacity; }
  [[nodiscard]] size_t             getBufferSize() const { return _bufferSize; }

  [[nodiscard]] HiCR::CommunicationManager        *getPayloadCommunicationManager() const { return _payloadCommunicationManager; }
  [[nodiscard]] HiCR::MemoryManager               *getPayloadMemoryManager() const { return _payloadMemoryManager; }
  [[nodiscard]] std::shared_ptr<HiCR::MemorySpace> getPayloadMemorySpace() const { return _payloadMemorySpace; }

  [[nodiscard]] HiCR::CommunicationManager        *getCoordinationCommunicationManager() const { return _coordinationCommunicationManager; }
  [[nodiscard]] HiCR::MemoryManager               *getCoordinationMemoryManager() const { return _coordinationMemoryManager; }
  [[nodiscard]] std::shared_ptr<HiCR::MemorySpace> getCoordinationMemorySpace() const { return _coordinationMemorySpace; }

  private:

  std::string _name;
  size_t      _bufferCapacity = 1;
  size_t      _bufferSize     = 0;

  HiCR::CommunicationManager        *_payloadCommunicationManager = nullptr;
  HiCR::MemoryManager               *_payloadMemoryManager        = nullptr;
  std::shared_ptr<HiCR::MemorySpace> _payloadMemorySpace;

  HiCR::CommunicationManager        *_coordinationCommunicationManager = nullptr;
  HiCR::MemoryManager               *_coordinationMemoryManager        = nullptr;
  std::shared_ptr<HiCR::MemorySpace> _coordinationMemorySpace;
};

} // namespace serving::configuration
