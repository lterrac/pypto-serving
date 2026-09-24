"""The config types through the bindings: names, defaults, keyword construction
and ``__post_init__`` messages match the dataclasses.

Defaults are written down rather than imported, because ``config.types`` imports
torch; the parity test compares against the live dataclasses where torch is
present.
"""

from __future__ import annotations

import importlib

import pytest

s = importlib.import_module("_pypto_serving")


# ---------------------------------------------------------------------------
# construction and defaults
# ---------------------------------------------------------------------------


def test_defaults_are_the_dataclass_defaults():
    g = s.GenerateConfig()
    assert (g.max_new_tokens, g.temperature, g.top_p, g.top_k, g.seed) == (256, 0.0, 1.0, None, None)
    assert (g.stop, g.stream, g.ignore_eos) == ([], False, False)

    r = s.RuntimeConfig()
    assert (r.page_size, r.max_batch_size, r.max_seq_len) == (64, 1, 4096)
    assert (r.device, r.kv_dtype, r.weight_dtype) == ("cpu", "bfloat16", "bfloat16")
    assert r.total_kv_pages is None
    assert r.npu_memory_utilization == pytest.approx(0.9)
    assert r.max_num_batched_tokens == 4096
    assert r.max_prefill_tokens_per_request is None
    assert r.prefill_chunk_size_choices == []
    assert r.supports_chunked_prefill_with_speculation is True
    assert r.requires_homogeneous_prefill_decode is False
    assert (r.max_new_tokens, r.num_speculative_tokens) == (256, 0)
    assert r.kv_cache_groups == []
    assert (r.ring_dep_pool, r.ring_task_window, r.ring_heap) == (None, None, None)

    spec = s.KVCacheSpec(block_size=16, page_size_bytes=1024)
    assert spec.compress_ratio == 1
    assert (spec.token_capacity, spec.storage_block_size) == (16, 16)

    group = s.KVCacheGroupSpec(name="full", layer_indices=[0, 1], spec=spec, max_blocks_per_seq=8)
    assert (group.num_blocks, group.num_partitions, group.sliding_window, group.is_eagle_group) == (None, 1, None, False)

    p = s.SamplingParams(temperature=0.0, top_p=1.0)
    assert (p.top_k, p.seed) == (None, None)

    a = s.KvAllocation(request_id="r", model_id="m", page_ids=[1, 2], tokens_capacity=128)
    assert a.tokens_used == 0

    st = s.RequestState(request_id="r", model_id="m", prompt="hi", prompt_token_ids=[1, 2])
    assert st.generated_token_ids == []
    assert st.sampling_params is None
    assert st.status == "waiting"
    assert (st.max_new_tokens, st.stop_strings, st.eos_token_id, st.seq_len) == (0, [], None, 0)
    assert (st.num_prompt_tokens, st.kv_allocation, st.output_text) == (0, None, "")

    assert s.PREFILL_CHUNK_SIZE_CHOICES == [1024, 2048, 4096, 8192]


def test_keyword_construction_as_the_model_writes_it():
    # These are the exact spellings found in pypto_serving/model/.
    r = s.RuntimeConfig(max_seq_len=8192)
    assert r.max_seq_len == 8192 and r.page_size == 64

    p = s.SamplingParams(temperature=0.0, top_p=1.0, top_k=None)
    assert p == s.SamplingParams(temperature=0.0, top_p=1.0)

    c = s.ModelConfig(
        model_id="qwen3-14b", architecture="Qwen3ForCausalLM", vocab_size=151936, hidden_size=5120,
        intermediate_size=17408, num_hidden_layers=40, num_attention_heads=40, num_key_value_heads=8,
        head_dim=128, max_position_embeddings=40960, rms_norm_eps=1e-6, rope_theta=1000000.0,
        bos_token_id=None, eos_token_id=151645, pad_token_id=None, torch_dtype="bfloat16",
    )
    assert (c.num_key_value_heads, c.eos_token_id, c.bos_token_id) == (8, 151645, None)

    layer = s.LayerSpec(layer_idx=3, hidden_size=5120, intermediate_size=17408, num_attention_heads=40, num_key_value_heads=8, head_dim=128)
    assert layer.layer_idx == 3


def test_required_fields_are_required():
    for make in (s.ModelConfig, s.KVCacheSpec, s.LayerSpec, s.KvAllocation, s.GenerateResult, s.SamplingParams):
        with pytest.raises(TypeError):
            make()
    with pytest.raises(TypeError):
        s.KVCacheGroupSpec(name="g")  # layer_indices, spec, max_blocks_per_seq missing


def test_sequences_accept_tuples_as_the_frozen_dataclasses_use_them():
    g = s.GenerateConfig(stop=("</s>", "\n\n"))
    assert g.stop == ["</s>", "\n\n"]
    r = s.RuntimeConfig(prefill_chunk_size_choices=(1024, 2048))
    assert r.prefill_chunk_size_choices == [1024, 2048]
    grp = s.KVCacheGroupSpec(name="g", layer_indices=(0, 1, 2), spec=s.KVCacheSpec(block_size=8, page_size_bytes=64), max_blocks_per_seq=4)
    assert grp.layer_indices == [0, 1, 2]


def test_ring_knobs_take_int_list_or_none():
    assert s.RuntimeConfig(ring_heap=None).ring_heap is None
    assert s.RuntimeConfig(ring_heap=536870912).ring_heap == [536870912]
    assert s.RuntimeConfig(ring_heap=[1, 2, 3]).ring_heap == [1, 2, 3]
    assert s.RuntimeConfig(ring_dep_pool=(4, 5)).ring_dep_pool == [4, 5]


# ---------------------------------------------------------------------------
# validation: the __post_init__ messages, verbatim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(block_size=0, page_size_bytes=1), "KV cache block_size must be positive"),
        (dict(block_size=8, page_size_bytes=0), "KV cache page_size_bytes must be positive"),
        (dict(block_size=8, page_size_bytes=1, compress_ratio=0), "KV cache compress_ratio must be positive"),
        (dict(block_size=8, page_size_bytes=1, compress_ratio=3), "KV cache block_size must be divisible by compress_ratio"),
    ],
)
def test_kv_cache_spec_raises_what_post_init_raises(kwargs, message):
    with pytest.raises(ValueError, match=f"^{message}$"):
        s.KVCacheSpec(**kwargs)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(name=""), "KV cache group name must not be empty"),
        (dict(max_blocks_per_seq=0), "KV cache max_blocks_per_seq must be positive"),
        (dict(num_blocks=0), "KV cache num_blocks must be positive when specified"),
        (dict(num_partitions=0), "KV cache num_partitions must be positive"),
        (dict(sliding_window=0), "KV cache sliding_window must be positive"),
        (dict(sliding_window=24), "KV cache sliding_window must be a multiple of the block token capacity"),
        (dict(sliding_window=16 * 9), "KV cache sliding_window requires more blocks than max_blocks_per_seq"),
    ],
)
def test_kv_cache_group_spec_raises_what_post_init_raises(kwargs, message):
    base = dict(name="g", layer_indices=[0], spec=s.KVCacheSpec(block_size=16, page_size_bytes=1024), max_blocks_per_seq=8)
    with pytest.raises(ValueError, match=f"^{message}$"):
        s.KVCacheGroupSpec(**{**base, **kwargs})


def test_request_status_is_one_of_the_six_names():
    st = s.RequestState(request_id="r", model_id="m", prompt="", prompt_token_ids=[])
    for name in ("waiting", "prefill", "decode", "finished", "aborted", "error"):
        st.status = name
        assert st.status == name
    with pytest.raises(ValueError, match="invalid request status 'done'"):
        st.status = "done"
    with pytest.raises(ValueError):
        s.RequestState(request_id="r", model_id="m", prompt="", prompt_token_ids=[], status="done")


# ---------------------------------------------------------------------------
# dataclass semantics: frozen, equality, repr
# ---------------------------------------------------------------------------


def test_frozen_types_are_read_only_and_mutable_ones_are_not():
    for frozen in (s.GenerateConfig(), s.RuntimeConfig(), s.KVCacheSpec(block_size=8, page_size_bytes=1)):
        with pytest.raises(AttributeError):
            frozen.max_new_tokens = 1 if hasattr(frozen, "max_new_tokens") else None
            frozen.block_size = 1

    p = s.SamplingParams(temperature=0.0, top_p=1.0)
    p.temperature = 0.7
    assert p.temperature == 0.7

    st = s.RequestState(request_id="r", model_id="m", prompt="", prompt_token_ids=[1])
    st.generated_token_ids = [5, 6]
    st.kv_allocation = s.KvAllocation(request_id="r", model_id="m", page_ids=[0], tokens_capacity=64)
    st.sampling_params = p
    assert st.generated_token_ids == [5, 6]
    assert st.kv_allocation.page_ids == [0]
    assert st.sampling_params.temperature == 0.7


def test_equality_is_by_value():
    assert s.RuntimeConfig(max_seq_len=8) == s.RuntimeConfig(max_seq_len=8)
    assert s.RuntimeConfig(max_seq_len=8) != s.RuntimeConfig(max_seq_len=9)
    spec = s.KVCacheSpec(block_size=8, page_size_bytes=1)
    assert s.KVCacheGroupSpec(name="g", layer_indices=[0], spec=spec, max_blocks_per_seq=1) == s.KVCacheGroupSpec(name="g", layer_indices=[0], spec=spec, max_blocks_per_seq=1)
    assert s.GenerateResult(text="a", token_ids=[1], finish_reason="stop") == s.GenerateResult(text="a", token_ids=[1], finish_reason="stop")


def test_repr_reads_like_a_dataclass():
    assert repr(s.KVCacheSpec(block_size=8, page_size_bytes=64)) == "KVCacheSpec(block_size=8, page_size_bytes=64, compress_ratio=1)"
    assert repr(s.SamplingParams(temperature=0.0, top_p=1.0)) == "SamplingParams(temperature=0.0, top_p=1.0, top_k=None, seed=None)"
    assert repr(s.GenerateConfig()).startswith("GenerateConfig(max_new_tokens=256, temperature=0.0, top_p=1.0, top_k=None, seed=None, stop=[], ")


# ---------------------------------------------------------------------------
# live parity, where the Python dataclasses can be imported at all
# ---------------------------------------------------------------------------



def test_parity_with_the_live_dataclasses():
    # Inside the test, not at module level: a module-level importorskip skips
    # the whole file at collection, and this is the only test that needs torch.
    py_types = pytest.importorskip("pypto_serving.config.types", reason="pypto_serving.config.types needs torch")
    import dataclasses

    def same(cpp, py):
        for f in dataclasses.fields(py):
            got, want = getattr(cpp, f.name), getattr(py, f.name)
            if isinstance(want, tuple):
                want = list(want)
            assert got == want, f"{type(py).__name__}.{f.name}: {got!r} != {want!r}"

    same(s.RuntimeConfig(), py_types.RuntimeConfig())
    same(s.GenerateConfig(), py_types.GenerateConfig())
    spec_kw = dict(block_size=16, page_size_bytes=1024)
    same(s.KVCacheSpec(**spec_kw), py_types.KVCacheSpec(**spec_kw))
    same(s.SamplingParams(temperature=0.5, top_p=0.9, top_k=40, seed=7), py_types.SamplingParams(temperature=0.5, top_p=0.9, top_k=40, seed=7))
    st_kw = dict(request_id="r", model_id="m", prompt="p", prompt_token_ids=[1, 2, 3])
    same(s.RequestState(**st_kw), py_types.RequestState(**st_kw))

    for kwargs in (dict(block_size=0, page_size_bytes=1), dict(block_size=8, page_size_bytes=1, compress_ratio=3)):
        with pytest.raises(ValueError) as cpp_err:
            s.KVCacheSpec(**kwargs)
        with pytest.raises(ValueError) as py_err:
            py_types.KVCacheSpec(**kwargs)
        assert str(cpp_err.value) == str(py_err.value)
