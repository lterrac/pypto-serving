/**
 * Starts an interpreter and imports _pypto_serving from the inittab.
 */

#include <cstdio>
#include <exception>

#include <pybind11/embed.h>

namespace py = pybind11;

int main()
{
  try
  {
    py::scoped_interpreter guard{};
    py::exec(R"(
import _pypto_serving as s

cfg = s.RuntimeConfig(max_seq_len=8192)
kv = s.KvCacheManager(num_blocks=8, block_size=cfg.page_size)
ids = kv.allocate_block_ids(3)
print(f"embedded import ok: page_size={cfg.page_size} allocated={ids} free={kv.num_free_blocks}")

try:
    s.KVCacheSpec(block_size=0, page_size_bytes=1)
except ValueError as e:
    print(f"validation crosses the boundary as ValueError: {e}")
)");
    return 0;
  }
  catch (const std::exception &e)
  {
    std::fprintf(stderr, "fatal: %s\n", e.what());
    return 1;
  }
}
