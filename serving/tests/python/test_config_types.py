"""GenerateConfig and RuntimeConfig through the bindings.

The engine reads only these two out of ``pypto_serving.config.types``. They keep
that module's field names, defaults and keyword construction, so a caller can
move from the dataclass to the bound type without changing the call.

Defaults are written down rather than imported, because ``config.types`` imports
torch; the parity test compares against the live dataclass where torch is
present.
"""

from __future__ import annotations

import importlib

import pytest

s = importlib.import_module("_pypto_serving")


def test_generate_config_defaults():
    g = s.GenerateConfig()
    assert (g.max_new_tokens, g.temperature, g.top_p, g.top_k, g.seed) == (256, 0.0, 1.0, None, None)
    assert (g.stop, g.stream, g.ignore_eos) == ([], False, False)


def test_runtime_config_defaults():
    r = s.RuntimeConfig()
    assert (r.page_size, r.max_batch_size, r.max_seq_len) == (64, 1, 4096)


def test_keyword_construction_as_a_caller_writes_it():
    g = s.GenerateConfig(max_new_tokens=8, stop=("</s>", "\n\n"), ignore_eos=True)
    assert g.max_new_tokens == 8
    assert g.stop == ["</s>", "\n\n"], "a tuple is accepted, as the frozen dataclass uses one"
    assert g.ignore_eos is True

    r = s.RuntimeConfig(page_size=128, max_batch_size=16, max_seq_len=1024)
    assert (r.page_size, r.max_batch_size, r.max_seq_len) == (128, 16, 1024)


def test_both_are_read_only():
    for frozen, field in ((s.GenerateConfig(), "max_new_tokens"), (s.RuntimeConfig(), "page_size")):
        with pytest.raises(AttributeError):
            setattr(frozen, field, 1)


def test_equality_is_by_value():
    assert s.RuntimeConfig(max_seq_len=8) == s.RuntimeConfig(max_seq_len=8)
    assert s.RuntimeConfig(max_seq_len=8) != s.RuntimeConfig(max_seq_len=9)
    assert s.GenerateConfig(max_new_tokens=4) == s.GenerateConfig(max_new_tokens=4)


def test_repr_reads_like_a_dataclass():
    assert repr(s.RuntimeConfig()) == "RuntimeConfig(page_size=64, max_batch_size=1, max_seq_len=4096)"
    assert repr(s.GenerateConfig()).startswith("GenerateConfig(max_new_tokens=256, temperature=0.0, top_p=1.0, top_k=None, seed=None, stop=[], ")


def test_parity_with_the_live_dataclass():
    # Inside the test: a module-level importorskip would skip the whole file.
    py_types = pytest.importorskip("pypto_serving.config.types", reason="pypto_serving.config.types needs torch")

    cpp, dataclass = s.GenerateConfig(), py_types.GenerateConfig()
    for field in ("max_new_tokens", "temperature", "top_p", "top_k", "seed", "stream", "ignore_eos"):
        assert getattr(cpp, field) == getattr(dataclass, field), field
    assert cpp.stop == list(dataclass.stop)

    # RuntimeConfig keeps only the pool geometry; the scheduling limits it used
    # to mirror live in SchedulerConfig, so only these three are compared.
    cpp_rt, py_rt = s.RuntimeConfig(), py_types.RuntimeConfig()
    for field in ("page_size", "max_batch_size", "max_seq_len"):
        assert getattr(cpp_rt, field) == getattr(py_rt, field), field
