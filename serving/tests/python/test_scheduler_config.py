"""SchedulerConfig through the bindings."""

from __future__ import annotations

import importlib

s = importlib.import_module("_pypto_serving")


def test_defaults_and_keywords():
    cfg = s.SchedulerConfig()
    assert (cfg.max_num_running_reqs, cfg.max_num_scheduled_tokens, cfg.max_seq_len) == (32, 4096, 4096)
    assert (cfg.enable_prefix_cache, cfg.enable_chunk_prefill, cfg.num_speculative_tokens) == (True, True, 0)
    assert cfg.max_prefill_tokens_per_request is None
    cfg = s.SchedulerConfig(max_seq_len=8, enable_prefix_cache=False)
    assert (cfg.max_seq_len, cfg.enable_prefix_cache) == (8, False)
    cfg.max_num_running_reqs = 4
    assert cfg.max_num_running_reqs == 4
