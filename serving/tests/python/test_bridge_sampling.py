# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Which sampling path a batch takes.

``pypto_serving.bridge.sampling`` is deliberately torch-free, but importing it
by name pulls ``pypto_serving/__init__.py``, which is not. Loading it by path
keeps these rules checkable on a machine with no torch and no device -- which
is the whole reason they live in their own module.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest


def _load():
    path = pathlib.Path(__file__).resolve().parents[3] / "pypto_serving" / "bridge" / "sampling.py"
    spec = importlib.util.spec_from_file_location("bridge_sampling", path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses looks the module up in sys.modules while building the class.
    sys.modules["bridge_sampling"] = module
    spec.loader.exec_module(module)
    return module


s = _load()


def test_an_absent_block_is_greedy():
    # An older caller that sends no sampling block keeps the behaviour it had.
    assert s.Sampling.from_wire(None) == s.Sampling()
    assert s.Sampling.from_wire({}) == s.Sampling()
    assert s.Sampling.from_wire(None).is_greedy


def test_a_block_reads_the_wire_types():
    params = s.Sampling.from_wire({"temperature": 0.7, "top_p": 0.95, "top_k": 40, "seed": 1234})
    assert (params.temperature, params.top_p, params.top_k, params.seed) == (0.7, 0.95, 40, 1234)
    assert not params.is_greedy
    # None stays None rather than becoming 0, which would read as "top_k 0".
    assert s.Sampling.from_wire({"temperature": 0.7, "top_k": None, "seed": None}).top_k is None


def test_device_sampling_needs_every_request_to_be_greedy():
    greedy, sampled = s.Sampling(), s.Sampling(temperature=0.7)

    assert s.allow_device_sampled_ids([greedy, greedy], supports_device_sampling=True)
    # One stochastic request moves the whole batch off the device path, because
    # the flag goes onto the batch, not the request.
    assert not s.allow_device_sampled_ids([greedy, sampled], supports_device_sampling=True)
    # An executor that cannot sample at all never takes it.
    assert not s.allow_device_sampled_ids([greedy], supports_device_sampling=False)


def test_a_stochastic_executor_keeps_the_device_path_while_top_p_is_open():
    sampled = s.Sampling(temperature=0.7, top_p=1.0)
    narrowed = s.Sampling(temperature=0.7, top_p=0.9)
    kwargs = {"supports_device_sampling": True, "supports_device_stochastic_sampling": True}

    assert s.allow_device_sampled_ids([sampled], **kwargs)
    # top_p is a host-side filter over the distribution, so a narrowed one needs
    # the distribution back.
    assert not s.allow_device_sampled_ids([narrowed], **kwargs)


@pytest.mark.parametrize(
    ("params", "allowed"),
    [
        ([s.Sampling(temperature=0.7, top_k=40)], True),
        ([s.Sampling(temperature=0.7, top_k=32)], True),
        # Wider than the kernel selects: the host needs full logits.
        ([s.Sampling(temperature=0.7, top_k=64)], False),
        # No top_k at all -- the plain OpenAI request -- also falls to logits.
        ([s.Sampling(temperature=0.7)], False),
        # A greedy request in the batch takes the device-sampled path instead.
        ([s.Sampling(temperature=0.7, top_k=8), s.Sampling()], False),
        ([], False),
    ],
)
def test_device_top_k_needs_every_request_within_the_kernels_width(params, allowed):
    assert s.allow_device_topk_sampling(params, device_topk_sampling_k=40) is allowed


def test_an_executor_without_candidates_never_takes_the_top_k_path():
    assert not s.allow_device_topk_sampling([s.Sampling(temperature=0.7, top_k=8)], device_topk_sampling_k=0)


class FakeTensor:
    """Just enough tensor for the dispatch: dim/view/numel/item and indexing."""

    def __init__(self, rows):
        self._rows = list(rows)

    def dim(self):
        return 2 if self._rows and isinstance(self._rows[0], list) else 1

    def view(self, _shape):
        flat = []
        for row in self._rows:
            flat.extend(row) if isinstance(row, list) else flat.append(row)
        return FakeTensor(flat)

    def numel(self):
        return len(self._rows)

    def __getitem__(self, index):
        row = self._rows[index]
        return FakeTensor(row) if isinstance(row, list) else _Scalar(row)


class _Scalar:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class Result:
    def __init__(self, sampled=None, candidates=None, logits=None):
        self.sampled_token_ids = sampled
        self.sampling_candidates = candidates
        self.logits = logits


class RecordingSampler:
    def __init__(self, from_logits=101, from_candidates=202):
        self.from_logits = from_logits
        self.from_candidates = from_candidates
        self.calls = []

    def sample(self, logits, params, request_id=None):
        self.calls.append(("logits", request_id))
        return self.from_logits

    def sample_from_candidates(self, candidates, row_idx, params, request_id=None):
        self.calls.append(("candidates", request_id))
        return self.from_candidates


def test_device_sampled_ids_win_when_they_are_there():
    sampler = RecordingSampler()
    result = Result(sampled=FakeTensor([7, 8]))

    token = s.sample_row(sampler, result, None, None, "r", 1, allow_device=True, allow_topk=False)

    assert token == 8
    assert sampler.calls == []  # the host sampler must not be reached


def test_candidates_are_used_when_the_batch_asked_for_top_k():
    sampler = RecordingSampler()
    result = Result(candidates=object(), logits=FakeTensor([[1.0], [2.0]]))

    token = s.sample_row(sampler, result, None, None, "r", 0, allow_device=False, allow_topk=True)

    assert token == 202
    assert sampler.calls == [("candidates", "r")]


def test_logits_are_the_fallback():
    sampler = RecordingSampler()
    logits = FakeTensor([[1.0, 2.0], [3.0, 4.0]])

    token = s.sample_row(
        sampler, Result(logits=logits), s.logits_row(logits, 1), None, "r", 1, allow_device=False, allow_topk=False
    )

    assert token == 101
    assert sampler.calls == [("logits", "r")]


def test_a_result_that_ignored_the_flag_falls_through_rather_than_crashing():
    # The flags are what was asked for; an executor may still return something
    # else, and the row has to come from whatever actually arrived.
    sampler = RecordingSampler()
    logits = FakeTensor([[1.0, 2.0]])

    token = s.sample_row(
        sampler, Result(logits=logits), s.logits_row(logits, 0), None, "r", 0, allow_device=True, allow_topk=True
    )

    assert token == 101


def test_nothing_to_sample_from_names_both_flags():
    with pytest.raises(RuntimeError, match="neither sampled ids nor logits"):
        s.sample_row(RecordingSampler(), Result(), None, None, "r", 0, allow_device=True, allow_topk=False)


def test_a_short_sampled_tensor_is_an_error_not_a_wrong_token():
    with pytest.raises(RuntimeError, match="expected row 3"):
        s.sample_row(
            RecordingSampler(), Result(sampled=FakeTensor([1, 2])), None, None, "r", 3, allow_device=True, allow_topk=False
        )


def test_a_one_dimensional_logits_tensor_is_the_only_row():
    # A single-request step can come back as [vocab] rather than [1, vocab].
    flat = FakeTensor([1.0, 2.0, 3.0])
    assert s.logits_row(flat, 0) is flat
    assert s.logits_row(None, 0) is None
