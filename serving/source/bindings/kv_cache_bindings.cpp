#include <serving/bindings/bindings.hpp>

#include <optional>
#include <string>
#include <vector>

#include <pybind11/stl.h>

#include <serving/bindings/repr.hpp>
#include <serving/memory/kv_cache.hpp>

namespace py = pybind11;

namespace serving::bindings
{

using namespace serving::memory;

void bindKvCache(py::module_ &m)
{

  // Blocks are owned by the manager and handed out by reference; Python never
  // creates one. reference_internal keeps the manager alive while a block is held.
  py::class_<KVCacheBlock>(m, "KVCacheBlock", "Metadata for one physical KV cache page/block.")
    .def_readonly("block_id", &KVCacheBlock::blockId)
    .def_readonly("ref_cnt", &KVCacheBlock::refCnt)
    .def_readonly("block_hash", &KVCacheBlock::blockHash)
    .def("__repr__", [](const KVCacheBlock &b) { return Repr("KVCacheBlock").field("block_id", b.blockId).field("ref_cnt", b.refCnt).field("block_hash", b.blockHash).str(); });

  py::class_<KvCacheManager>(m, "KvCacheManager", "KV block metadata and the prefix cache for one model.")
    .def(py::init<std::optional<int>, int, bool>(), py::kw_only(), py::arg("num_blocks") = py::none(), py::arg("block_size") = 64, py::arg("enable_prefix_cache") = true)
    .def_property_readonly("num_free_blocks", &KvCacheManager::numFreeBlocks)
    .def_property_readonly("num_blocks", &KvCacheManager::numBlocks)
    .def_property_readonly("block_size", &KvCacheManager::blockSize)
    .def_property_readonly("enable_prefix_cache", &KvCacheManager::enablePrefixCache)
    .def("initialize", &KvCacheManager::initialize, py::arg("runtime"), py::kw_only(), py::arg("num_blocks"))
    .def("allocate_block_ids", &KvCacheManager::allocateBlockIds, py::arg("num_blocks"))
    // Python takes *block_id_groups; each group is released in turn.
    .def("release_blocks_by_ids",
         [](KvCacheManager &self, const py::args &groups) {
           for (const auto &group : groups) { self.releaseBlocksByIds(group.cast<std::vector<int>>()); }
         })
    .def("release_cached_blocks", &KvCacheManager::releaseCachedBlocks, py::arg("blocks"))
    .def("compute_block_hashes", &KvCacheManager::computeBlockHashes, py::arg("token_ids"))
    .def("get_computed_blocks",
         static_cast<std::vector<KVCacheBlock *> (KvCacheManager::*)(const std::vector<int> &, std::optional<int>)>(&KvCacheManager::getComputedBlocks),
         py::arg("token_ids"),
         py::arg("max_cache_hit_tokens") = py::none(),
         py::return_value_policy::reference_internal)
    .def("cache_block_ids", &KvCacheManager::cacheBlockIds, py::arg("block_ids"), py::arg("block_hashes"), py::arg("start"), py::arg("end"))
    .def("has_groups", &KvCacheManager::hasGroups)
    .def("block_at", &KvCacheManager::blockAt, py::arg("block_id"), py::return_value_policy::reference_internal)
    .def("__repr__", [](const KvCacheManager &k) {
      return Repr("KvCacheManager")
        .field("num_blocks", k.numBlocks())
        .field("num_free_blocks", k.numFreeBlocks())
        .field("block_size", k.blockSize())
        .field("enable_prefix_cache", k.enablePrefixCache())
        .str();
    });

  m.attr("NONE_HASH") = py::int_(NONE_HASH);
}

} // namespace serving::bindings
