# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Queue-shaped adapters over a platform ``Output``/``Input`` channel pair.

``ReplicaEngineCore`` (``serving/engine/async_engine.py``) only ever moves
opaque msgpack ``bytes`` through two call shapes, never anything richer:

    put(payload: bytes) -> None          # called synchronously on the
                                          # asyncio loop; must not block
                                          # indefinitely
    get(timeout: float | None) -> bytes  # always inside asyncio.to_thread;
                                          # MUST raise queue.Empty on timeout

``PlatformOutputQueue`` and ``PlatformInputQueue`` below give a platform
``Output``/``Input`` edge (see ``protocols.py``) that exact surface, so they
are a drop-in replacement for a ``multiprocessing.Queue`` end at each of the
call sites in ``async_engine.py`` (:315, :509, :709, :883 for ``put``; :317,
:575 for ``get``; ``queue.Empty`` caught at :321, :531, :714).
"""

from __future__ import annotations

import queue
import time

from pypto_serving.serving.transport.protocols import ChannelInput, ChannelOutput

# `Output.push` (`pushMessageLocking`) spins internally with no timeout of its
# own until the ring buffer has room (platform/docs/python-channel-contract.md,
# Layer 1). `put()` here is called synchronously on the asyncio event loop --
# never awaited, never wrapped in asyncio.to_thread -- so delegating straight
# to `push()` would let a full channel stall the whole loop indefinitely, not
# just the caller. This adapter instead polls `Output.is_full` itself and
# bounds the wait: `DEFAULT_PUT_TIMEOUT_SECONDS` is this module's own choice
# (the contract does not specify one), long enough to absorb a slow consumer
# but short enough to fail loudly rather than hang the process.
DEFAULT_PUT_TIMEOUT_SECONDS = 30.0
DEFAULT_PUT_POLL_INTERVAL_SECONDS = 0.0005
DEFAULT_GET_POLL_INTERVAL_SECONDS = 0.0005


class PlatformOutputQueue:
    """Adapts a platform ``Output`` edge to ``put(payload: bytes) -> None``.

    Full-channel behaviour (a design decision the contract leaves open):
    polls ``Output.is_full(len(payload))`` and only calls ``Output.push``
    once there is room, instead of calling ``push`` unconditionally (which
    would spin inside the native call with no way for this adapter to bound
    the wait). If the channel is still full after ``put_timeout`` seconds,
    raises ``queue.Full`` -- mirroring stdlib
    ``queue.Queue.put(block=True, timeout=...)`` -- instead of blocking the
    asyncio event loop forever. Passing ``put_timeout=None`` restores
    unbounded blocking (matches stdlib default when timeout is unset); do
    not do this for an instance driving the asyncio loop directly.

    The channel is expected to be sized (by the deployment/edge config, out
    of scope here) to comfortably hold the engine's in-flight pipeline depth
    (2 by default, see ``EngineConfig``/``_max_in_flight``), so under normal
    operation ``is_full`` should rarely if ever be true; ``queue.Full`` is a
    hard failure signal, not an expected steady-state event.
    """

    def __init__(
        self,
        output: ChannelOutput,
        *,
        edge_name: str = "",
        put_timeout: float | None = DEFAULT_PUT_TIMEOUT_SECONDS,
        poll_interval: float = DEFAULT_PUT_POLL_INTERVAL_SECONDS,
    ) -> None:
        self._output = output
        self._edge_name = edge_name
        self._put_timeout = put_timeout
        self._poll_interval = poll_interval

    def put(self, payload: bytes) -> None:
        """Push ``payload`` onto the channel; see class docstring for the
        full-channel policy."""
        message_size = len(payload)
        deadline = (
            None if self._put_timeout is None else time.monotonic() + self._put_timeout
        )
        while self._output.is_full(message_size):
            if deadline is not None and time.monotonic() >= deadline:
                raise queue.Full(
                    f"platform output channel {self._edge_name!r} still full "
                    f"after {self._put_timeout:g}s (message_size={message_size})"
                )
            time.sleep(self._poll_interval)
        self._output.push(payload)


class PlatformInputQueue:
    """Adapts a platform ``Input`` edge to ``get(timeout: float | None) -> bytes``.

    Polls ``Input.has_message()`` and calls ``Input.read()`` (which copies
    the payload out and pops the message, per the contract's Layer 1 rule 1)
    once one is available. Raises ``queue.Empty`` if no message arrives
    within ``timeout`` seconds -- required exactly, since three call sites in
    ``async_engine.py`` catch only ``queue.Empty``, not a bare ``TimeoutError``
    or anything else. ``timeout=None`` blocks indefinitely (matches stdlib
    ``queue.Queue.get`` semantics). Every call site in this codebase invokes
    ``get`` inside ``asyncio.to_thread``, so blocking here does not stall the
    asyncio event loop.
    """

    def __init__(
        self,
        input_: ChannelInput,
        *,
        edge_name: str = "",
        poll_interval: float = DEFAULT_GET_POLL_INTERVAL_SECONDS,
    ) -> None:
        self._input = input_
        self._edge_name = edge_name
        self._poll_interval = poll_interval

    def get(self, timeout: float | None) -> bytes:
        """Return the next message's payload, raising ``queue.Empty`` on timeout."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self._input.has_message():
            if deadline is None:
                time.sleep(self._poll_interval)
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise queue.Empty(
                    f"platform input channel {self._edge_name!r} empty "
                    f"after {timeout:g}s"
                )
            time.sleep(min(self._poll_interval, remaining))
        return self._input.read()
