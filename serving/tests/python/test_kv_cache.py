"""KvCacheManager through the bindings, against the semantics the C++ tests pin."""

from __future__ import annotations

import importlib

import pytest

s = importlib.import_module("_pypto_serving")


def test_kv_cache_manager_allocates_and_releases_like_the_python_one():
    m = s.KvCacheManager(num_blocks=4, block_size=2)
    assert (m.num_blocks, m.num_free_blocks, m.block_size, m.enable_prefix_cache) == (4, 4, 2, True)

    ids = m.allocate_block_ids(3)
    assert sorted(ids) == [0, 1, 2]
    assert m.num_free_blocks == 1
    assert m.allocate_block_ids(2) is None, "all or nothing"
    assert m.num_free_blocks == 1

    m.release_blocks_by_ids(ids[:2], ids[2:])  # *block_id_groups, as in Python
    assert m.num_free_blocks == 4
    assert m.block_at(0).ref_cnt == 0


def test_kv_cache_manager_prefix_cache_round_trip():
    m = s.KvCacheManager(num_blocks=8, block_size=2)
    tokens = [1, 2, 3, 4]
    hashes = m.compute_block_hashes(tokens)
    assert len(hashes) == 2, "two full blocks"
    assert all(isinstance(h, int) and h >= 0 for h in hashes)

    ids = m.allocate_block_ids(2)
    m.cache_block_ids(ids, hashes, 0, 2)

    # No limit: one token is held back so logits can be recomputed without
    # writing into the last shared block -- so 3 of 4 tokens, one full block.
    hit = m.get_computed_blocks(tokens)
    assert [b.block_id for b in hit] == ids[:1]
    assert m.block_at(ids[0]).ref_cnt == 2, "the request and the hit"
    m.release_cached_blocks(hit)
    assert m.block_at(ids[0]).ref_cnt == 1

    full = m.get_computed_blocks(tokens, max_cache_hit_tokens=4)
    assert [b.block_id for b in full] == ids
    m.release_cached_blocks(full)

    assert m.get_computed_blocks([9, 9, 9, 9]) == []
    assert not m.has_groups() and not m.has_eagle_groups()


def test_kv_cache_manager_initialize_from_runtime_config():
    m = s.KvCacheManager()
    assert m.num_blocks == 0
    m.initialize(s.RuntimeConfig(page_size=16), num_blocks=32)
    assert (m.num_blocks, m.block_size) == (32, 16)
    with pytest.raises(RuntimeError, match="invalid KV cache block count"):
        s.KvCacheManager().initialize(s.RuntimeConfig(), num_blocks=0)


def test_kv_cache_manager_keyword_only_as_in_python():
    with pytest.raises(TypeError):
        s.KvCacheManager(4)  # Python's signature is keyword-only
