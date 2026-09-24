# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Runs the pypto model for the C++ engine.

``open_model`` loads and compiles the model; ``run_step`` executes one step given
prefill and decode items; ``close`` releases the executor. ``step`` is
``run_step`` over JSON, for a C++ process that embeds Python. No torch object
crosses the boundary: token ids, block ids and lengths only.

``open_model`` forks simpler's per-chip children, so it must run before any
other thread exists.

Items::

    prefill: {"request_id": str, "tokens": [int], "num_computed": int,
              "sample_at_length": int, "block_ids": [int]}
    decode:  {"request_id": str, "last_token": int, "seq_len": int,
              "block_ids": [int]}

    run_step(prefill, decode) -> {request_id: [int]}

Greedy only: Qwen samples on device and the fused decode kernel emits the
sampled ids; a request with temperature > 0 is refused.
"""

from __future__ import annotations

import json
import logging
import traceback
from typing import Any

logger = logging.getLogger(__name__)

_MODEL_ID = "bridge-model"

#: Process-wide bridge state. One model per process, matching the worker today.
_state: "_BridgeState | None" = None


class _BridgeState:
    """Everything the bridge needs between ``open_model`` and ``close``."""

    def __init__(self, executor: Any, record: Any, num_pages: int) -> None:
        self.executor = executor
        self.record = record
        self.num_pages = num_pages
        self.page_size = record.runtime.page_size


def _require_state() -> _BridgeState:
    if _state is None:
        raise RuntimeError("bridge is not open; call open_model() first")
    return _state


def open_model(
    model_dir: str,
    device_id: int,
    platform: str = "a2a3",
    max_model_len: int = 512,
    block_size: int = 128,
    max_num_seqs: int = 16,
    pypto_build_dir: str = "build_output/bridge",
) -> int:
    """Load the model, compile its kernels, and return the KV page count.

    This is the expensive call -- kernel compile plus weight staging, about two
    minutes cold for Qwen3-14B. It is also the call that makes simpler fork its
    per-chip children, so **the C++ side must complete it before starting any
    thread of its own**: simpler mmaps its heap ring and installs its
    ``pthread_atfork`` handler in the parent, and forking a process that already
    has threads running is how that gets corrupted.
    """
    global _state
    if _state is not None:
        raise RuntimeError("bridge is already open")

    # Imported lazily: these pull torch and the pypto toolchain, and that cost
    # belongs to this call rather than to `import pypto_serving.bridge`.
    from pypto_serving.config.types import ModelRecord, RuntimeConfig  # noqa: PLC0415
    from pypto_serving.model.model_loader import ModelLoader  # noqa: PLC0415
    from pypto_serving.model.qwen.npu_executor import Qwen314BPyptoExecutor  # noqa: PLC0415

    runtime_config = RuntimeConfig(
        page_size=block_size,
        max_batch_size=max_num_seqs,
        max_seq_len=max_model_len,
        device="cpu",
        kv_dtype="bfloat16",
        weight_dtype="bfloat16",
    )

    executor = Qwen314BPyptoExecutor(
        platform=platform,
        device_ids=(int(device_id),),
        pypto_build_dir=pypto_build_dir,
    )

    loaded = ModelLoader().load(
        model_id=_MODEL_ID,
        model_dir=model_dir,
        runtime_config=runtime_config,
    )
    record = ModelRecord(
        config=loaded.config,
        runtime=loaded.runtime_model.runtime,
        tokenizer=loaded.tokenizer,
        layer_specs=loaded.layer_specs,
        runtime_model=loaded.runtime_model,
    )

    num_pages = int(executor.register_model(_MODEL_ID, record))
    _state = _BridgeState(executor, record, num_pages)
    logger.info("bridge open: %d KV pages, page_size=%d", num_pages, _state.page_size)
    return num_pages


def page_size() -> int:
    """Tokens per KV block, so the caller can size its own block table."""
    return _require_state().page_size


def step(command_json: str) -> str:
    """One step over JSON. Errors are returned in the payload, not raised."""
    try:
        command = json.loads(command_json)
        tokens = _run_step(command)
        return json.dumps({"tokens": tokens})
    except Exception as exc:  # noqa: BLE001 -- the boundary must not raise
        logger.exception("bridge step failed")
        return json.dumps({"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})


def _run_step(command: dict[str, Any]) -> dict[str, list[int]]:
    return run_step(command.get("prefill") or [], command.get("decode") or [])


def run_step(prefill: list[dict[str, Any]], decode: list[dict[str, Any]]) -> dict[str, list[int]]:
    """One step given prefill and decode items; see the module docstring."""
    state = _require_state()
    tokens: dict[str, list[int]] = {}
    if prefill:
        tokens.update(_run_prefill(state, prefill))
    if decode:
        tokens.update(_run_decode(state, decode))
    return tokens


def _run_prefill(state: _BridgeState, items: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Batched prompt prefill, mirroring ``WorkerProcess._batch_prefill``."""
    from pypto_serving.serving.utils.prefill import pack_prefill_batch  # noqa: PLC0415

    request_ids = [item["request_id"] for item in items]
    token_chunks = [[int(t) for t in item["tokens"]] for item in items]
    chunk_starts = [int(item["num_computed"]) for item in items]
    seq_lens = [start + len(chunk) for start, chunk in zip(chunk_starts, token_chunks, strict=True)]
    block_ids = [[int(b) for b in item["block_ids"]] for item in items]

    batch = pack_prefill_batch(
        request_ids=request_ids,
        token_chunks=token_chunks,
        seq_lens=seq_lens,
        chunk_starts=chunk_starts,
        device=state.record.runtime.device,
        embedding_lookup=None,  # Qwen embeds on device
        allow_device_greedy_sampling=True,
        allow_device_topk_sampling=False,
        block_ids=block_ids,
        block_ids_by_group=[{} for _ in items],
        cache_partitions=[None for _ in items],
    )
    result = state.executor.run_prefill(state.record.runtime_model, batch)

    # Only a chunk that fed the last outstanding token sampled anything. For a
    # fresh request that is the end of the prompt; for one replaying after
    # preemption it is past it, so the replay does not emit a duplicate token.
    sampled: dict[str, list[int]] = {}
    completed_ids: list[str] = []
    completed_tokens: list[int] = []
    for index, item in enumerate(items):
        covered = chunk_starts[index] + len(token_chunks[index])
        if covered < int(item["sample_at_length"]):
            continue
        token_id = _sampled_id(result, index, "prefill")
        sampled[item["request_id"]] = [token_id]
        completed_ids.append(item["request_id"])
        completed_tokens.append(token_id)

    if completed_ids:
        # A no-op for Qwen, but it is what the worker does; keep the shapes identical.
        state.executor.finalize_prefill(
            state.record.runtime_model,
            completed_ids,
            completed_tokens,
            [_greedy_params() for _ in completed_ids],
        )
    return sampled


def _run_decode(state: _BridgeState, items: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Batched decode, mirroring ``WorkerProcess._make_decode_batch`` + ``_batch_decode``."""
    import torch  # noqa: PLC0415
    from pypto_serving.config.types import DecodeBatch  # noqa: PLC0415

    device = state.record.runtime.device
    decode_tokens = [int(item["last_token"]) for item in items]
    seq_lens = [int(item["seq_len"]) for item in items]

    batch = DecodeBatch(
        request_ids=[item["request_id"] for item in items],
        token_ids=torch.tensor(decode_tokens, dtype=torch.long, device=device).unsqueeze(1),
        hidden_states=None,  # Qwen gathers embeddings on device
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        allow_device_greedy_sampling=True,
        allow_device_topk_sampling=False,
        sampling_params=[_greedy_params() for _ in items],
        block_ids=[[int(b) for b in item["block_ids"]] for item in items],
        block_ids_by_group=[{} for _ in items],
        cache_partitions=[None for _ in items],
    )
    result = state.executor.run_decode(state.record.runtime_model, batch)

    return {
        item["request_id"]: [_sampled_id(result, index, "decode")]
        for index, item in enumerate(items)
    }


def close() -> None:
    """Release the executor and its device state."""
    global _state
    if _state is None:
        return
    try:
        _state.executor.close()
    finally:
        _state = None


def _greedy_params():
    from pypto_serving.config.types import SamplingParams  # noqa: PLC0415

    return SamplingParams(temperature=0.0, top_p=1.0, top_k=None, seed=None)


def _sampled_id(result: Any, row: int, phase: str) -> int:
    """Pull one row out of an executor result's device-sampled token ids."""
    sampled = getattr(result, "sampled_token_ids", None)
    if sampled is None:
        raise RuntimeError(
            f"{phase} returned no sampled_token_ids; the bridge is greedy-only and "
            "relies on the executor's device sampling"
        )
    flat = sampled.view(-1)
    if flat.numel() <= row:
        raise RuntimeError(f"{phase} returned {flat.numel()} sampled rows, expected row {row}")
    return int(flat[row].item())
