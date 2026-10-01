# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Which sampling path a step takes.

Policy, not tensors: this module imports no torch, so the rules can be read and
tested without a device. The executor's three paths, in the order
``WorkerProcess._sample_result_row`` tries them:

* **device sampled ids** -- the kernel sampled already and the host reads one
  int. Needs every request in the batch to be greedy, or the executor to
  support stochastic sampling on device.
* **device top-k candidates** -- the kernel returns the k best values and ids,
  and the host samples among those. Needs every request to name a ``top_k``
  within the executor's width.
* **host logits** -- the full ``[vocab]`` row comes back and the host samples
  over it. The fallback, and the expensive one.

The flags are per batch, not per request: they go onto the batch the executor
receives, so one request that does not qualify moves the whole step.

``sample_row`` then takes whichever path the result actually supports. It
touches tensors only through ``dim``/``view``/``numel``/``item``, so it stays
testable without a device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: What a request gets when the caller sends no sampling block at all.
GREEDY_WIRE: dict[str, Any] = {"temperature": 0.0, "top_p": 1.0, "top_k": None, "seed": None}


@dataclass(frozen=True)
class Sampling:
    """One request's sampling parameters, as the C++ engine sends them."""

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    seed: int | None = None

    @classmethod
    def from_wire(cls, block: dict[str, Any] | None) -> "Sampling":
        """Read one item's ``sampling`` block. Absent means greedy."""
        if not block:
            return cls()
        top_k = block.get("top_k")
        seed = block.get("seed")
        return cls(
            temperature=float(block.get("temperature", 0.0)),
            top_p=float(block.get("top_p", 1.0)),
            top_k=None if top_k is None else int(top_k),
            seed=None if seed is None else int(seed),
        )

    @property
    def is_greedy(self) -> bool:
        """Greedy needs no distribution, so the device can sample it alone."""
        return self.temperature <= 0.0


def allow_device_sampled_ids(
    sampling: list[Sampling],
    *,
    supports_device_sampling: bool,
    supports_device_stochastic_sampling: bool = False,
) -> bool:
    """Whether the executor can sample every request in the batch itself.

    Mirrors ``WorkerProcess._allow_device_sampled_ids``.
    """
    if not supports_device_sampling:
        return False
    return all(
        params.is_greedy or (supports_device_stochastic_sampling and params.top_p >= 1.0)
        for params in sampling
    )


def allow_device_topk_sampling(sampling: list[Sampling], *, device_topk_sampling_k: int) -> bool:
    """Whether the batch can be sampled from device top-k candidates.

    Mirrors ``WorkerProcess._allow_device_topk_sampling``. Every request must be
    stochastic and name a ``top_k`` the executor's candidate width covers; a
    greedy request in the batch takes the device-sampled path instead, and one
    asking for a wider top_k than the kernel selects takes full logits.
    """
    if device_topk_sampling_k <= 0 or not sampling:
        return False
    return all(
        not params.is_greedy and params.top_k is not None and 0 < params.top_k <= device_topk_sampling_k
        for params in sampling
    )


def logits_row(logits: Any, row: int) -> Any:
    """One request's logits row. A device-sampling step returns none at all."""
    if logits is None:
        return None
    return logits[row] if logits.dim() > 1 else logits


def sample_row(
    sampler: Any,
    result: Any,
    logits: Any,
    params: Any,
    request_id: str,
    row: int,
    *,
    allow_device: bool,
    allow_topk: bool,
) -> int:
    """One token, by whichever path the step actually took.

    Mirrors ``WorkerProcess._sample_result_row``: the flags say what was asked
    for and the result says what came back, and both have to agree before a
    path is taken. An executor is free to ignore a flag, which is why this
    checks the result rather than trusting the request.
    """
    sampled = getattr(result, "sampled_token_ids", None)
    if allow_device and sampled is not None:
        flat = sampled.view(-1)
        if flat.numel() <= row:
            raise RuntimeError(f"sampled_token_ids has {flat.numel()} rows, expected row {row}")
        return int(flat[row].item())

    candidates = getattr(result, "sampling_candidates", None)
    if allow_topk and candidates is not None:
        return int(sampler.sample_from_candidates(candidates, row, params, request_id))

    if logits is None:
        raise RuntimeError(
            f"the step returned neither sampled ids nor logits for {request_id!r}; "
            f"device sampling was {'allowed' if allow_device else 'not allowed'} and "
            f"top-k candidates {'allowed' if allow_topk else 'not allowed'}"
        )
    return int(sampler.sample(logits, params, request_id))
