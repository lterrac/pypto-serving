# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The engine<->worker startup handshake, as an opaque payload on the ``results`` edge.

Two values cross the boundary exactly once, at startup, before the engine loop
runs: "the worker finished loading the model" (``ready_event``) and "the device
has this many KV-cache pages" (``num_pages_value``). With a worker *process*
they ride a ``multiprocessing.Event`` and a ``multiprocessing.Value``; with a
worker *rank* there is no shared memory to put them in.

They deliberately do NOT become platform concepts. ``num_pages`` is a
device-derived quantity that sizes a model-support structure, and
pypto-serving#32 forbids the platform growing into another model-execution
abstraction layer -- "keep the shortest path from request scheduling to
PyPTO/Simpler execution". So the handshake is msgpack bytes on the ``results``
edge that already exists, indistinguishable to the platform from a StepResult:
the platform moves bytes and knows nothing about pages.

Ordering makes one message safe to overload onto that edge. The worker publishes
the handshake as the FIRST thing it ever pushes on ``results``, before entering
its busy loop, and the engine consumes it inside ``ReplicaEngineCore.start()``
before ``_engine_loop`` is created -- so it can never be confused with a
``StepResult``, and the "one result per command" FIFO invariant the engine
relies on (``async_engine.py`` :539-547) is untouched. The protocol version and
the distinct struct shape make a future violation a decode error rather than a
misapplied token.
"""

from __future__ import annotations

import queue

import msgspec

# Bumped when the struct below changes shape. Mismatch is fatal, not tolerated:
# the two ranks are the same binary in the SPMD launch, so a mismatch means the
# ranks are running different code and nothing downstream can be trusted.
HANDSHAKE_PROTOCOL_VERSION: int = 1


class WorkerHandshake(msgspec.Struct):
    """One-shot worker startup report: readiness, KV page count, or a failure."""

    protocol: int = HANDSHAKE_PROTOCOL_VERSION
    num_pages: int = 0
    error: str | None = None


_handshake_encoder: msgspec.msgpack.Encoder = msgspec.msgpack.Encoder()
_handshake_decoder: msgspec.msgpack.Decoder = msgspec.msgpack.Decoder(WorkerHandshake)


def encode_handshake(handshake: WorkerHandshake) -> bytes:
    """Encode the startup handshake for transport as an opaque payload."""
    return _handshake_encoder.encode(handshake)


def decode_handshake(data: bytes) -> WorkerHandshake:
    """Decode a startup handshake payload, rejecting a protocol mismatch."""
    handshake = _handshake_decoder.decode(data)
    if handshake.protocol != HANDSHAKE_PROTOCOL_VERSION:
        raise RuntimeError(
            f"Worker startup handshake protocol {handshake.protocol} does not match this "
            f"process's {HANDSHAKE_PROTOCOL_VERSION}. Under the SPMD launch both ranks are "
            "the same binary, so this means the ranks are running different code."
        )
    return handshake


class PlatformStartupHandshake:
    """Stands in for BOTH ``ready_event`` and ``num_pages_value``.

    ``ReplicaEngineCore.start()`` consumes those two as
    ``ready_event.wait(timeout=...) -> bool`` and ``num_pages_value.value ->
    int`` and never stores either on the object (``async_engine.py`` :245,
    :259). One object satisfying both shapes lets the platform transport reuse
    ``start()``'s existing six-tuple unpacking unchanged, so the queue path
    keeps exactly the code it has today.

    ``wait()`` reads the single handshake message off the ``results`` input
    queue. It returns ``False`` on timeout (which ``start()`` turns into its own
    "worker failed to initialize within N s" error) and raises if the worker
    reported a failure, so a broken worker rank surfaces as an exception on the
    engine rank rather than as a silently zero-sized KV cache -- which is what
    the process path does today, because ``_worker_entry`` sets ``ready_event``
    even on failure and leaves ``num_pages_value`` at 0.
    """

    def __init__(self, results, *, edge_name: str = "") -> None:
        self._results = results
        self._edge_name = edge_name
        self._num_pages: int | None = None

    def wait(self, timeout: float | None = None) -> bool:
        """Block for the worker's startup handshake. ``ready_event.wait`` shape.

        Single-use, and it says so. Unlike a ``multiprocessing.Event``, which is
        idempotent, this consumes a message off a shared edge: a second call
        would eat the next ``StepResult`` and try to decode it as a handshake.
        Nothing calls it twice today (``start()`` is the only caller), but the
        object is reachable for as long as the replica lives.
        """
        if self._num_pages is not None:
            raise RuntimeError(
                f"The startup handshake on {self._edge_name!r} has already been "
                "consumed. Calling wait() again would read the next message on "
                "that edge -- a StepResult -- and decode it as a handshake."
            )
        try:
            raw = self._results.get(timeout=timeout)
        except queue.Empty:
            return False
        handshake = decode_handshake(raw)
        if handshake.error:
            raise RuntimeError(
                f"Serving worker rank failed to initialise: {handshake.error}"
            )
        self._num_pages = handshake.num_pages
        return True

    @property
    def value(self) -> int:
        """The worker's reported KV-cache page count. ``num_pages_value.value`` shape."""
        if self._num_pages is None:
            raise RuntimeError(
                "The worker startup handshake has not been received yet; "
                "num_pages is only known after wait() has returned True."
            )
        return self._num_pages
