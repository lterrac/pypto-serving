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
                                          # asyncio loop; must never block
    get(timeout: float | None) -> bytes  # always inside asyncio.to_thread;
                                          # MUST raise queue.Empty on timeout

``PlatformOutputQueue`` and ``PlatformInputQueue`` below give a platform
``Output``/``Input`` edge (see ``protocols.py``) that exact surface, so they
are a drop-in replacement for a ``multiprocessing.Queue`` end at each of the
call sites in ``async_engine.py`` (:315, :509, :709, :883 for ``put``; :317,
:575 for ``get``; ``queue.Empty`` caught at :321, :531, :714) and both
``input_queue.get()``/``output_queue.get()`` call sites in
``serving_worker.py`` (:232, :281), which pass no timeout at all.
"""

from __future__ import annotations

import queue
import time

from pypto_serving.serving.transport.protocols import ChannelInput, ChannelOutput

DEFAULT_GET_POLL_INTERVAL_SECONDS = 0.0005


class ChannelNotReadyError(RuntimeError):
    """Raised when ``put``/``get``/``full`` is called before the channel has
    finished its startup handshake.

    ``Output.is_full`` and ``Input.has_message`` call ``checkReady()``
    natively and throw ``HICR_THROW_LOGIC`` if invoked before
    ``Platform.start()`` completes (see
    ``platform/include/system/channels/base.hpp``). This adapter gates on
    ``is_ready()`` itself first, so that precondition violation surfaces as a
    clear Python exception with a clear message instead of whatever a native
    ``HICR_THROW_LOGIC`` maps to across the pybind boundary. Ordering
    precondition: callers must not use a ``PlatformOutputQueue`` /
    ``PlatformInputQueue`` until the wrapped channel's ``is_ready()`` is
    ``True`` -- i.e. not before ``Platform.start()`` (or
    ``wait_until_ready()``) has returned for that edge.
    """


class MessageTooLargeError(ValueError):
    """Raised by ``put``/``try_put`` when a payload can never fit the channel.

    Distinguished on purpose from ``queue.Full``: a payload larger than the
    edge's configured buffer size makes ``Output.is_full`` return ``True``
    forever, for that payload, regardless of how long a caller waits or how
    many times it retries -- unlike an ordinary full channel, which clears
    once the consumer drains it. Retrying (or waiting) on
    ``MessageTooLargeError`` is never correct; retrying on ``queue.Full`` is
    the intended pattern.

    Detecting this from the ``Output`` object alone is not possible: the
    contract's ``Output.is_full(message_size)`` does not distinguish "full
    because of backlog" from "full because this message can never fit", and
    no accessor exposes the edge's configured buffer size. This class is
    therefore only raised when the adapter is constructed with an explicit
    ``max_message_size`` (sourced from the edge/deployment config by the
    caller, out of scope for this module); with no ``max_message_size``
    every full result reads as ``queue.Full``.
    """


class PlatformOutputQueue:
    """Adapts a platform ``Output`` edge to ``put(payload: bytes) -> None``.

    Full-channel behaviour (a design decision the contract leaves open, and
    the one point this adapter must get right): ``put()`` NEVER sleeps and
    NEVER retries. It checks ``Output.is_full(len(payload))`` exactly once
    and either pushes immediately or raises ``queue.Full`` immediately.

    ``put()`` is called synchronously on the asyncio event loop -- never
    awaited, never wrapped in ``asyncio.to_thread``
    (``async_engine.py`` :315, :509, :709, :883) -- so any sleep here, even a
    short bounded one, stalls the whole loop for that duration, not just the
    caller. An earlier version of this adapter polled with a bounded
    timeout; measured on hardware a 2s timeout blocked the event loop for a
    full 2.00s, and the loop that drives ``put()``
    (``ReplicaEngineCore._engine_loop``) has no ``try/except`` around its
    dispatch call and no ``add_done_callback`` on the loop task
    (``async_engine.py`` :286, :446-480) -- an uncaught exception there kills
    the loop task silently while ``_running`` stays ``True``, so the replica
    keeps accepting HTTP requests and serves nothing. A bounded sleep-based
    wait does not fix that; it just changes the outage from "instant" to
    "however long the timeout was". So: zero sleep, always.

    The edge's default buffer capacity is **1 message**
    (``__SERVING_PARTITION_DEFAULT_BUFFER_CAPACITY`` in
    ``platform/include/modules/configuration/edge.hpp``), not some multiple
    matching the engine's pipeline depth -- so ``is_full`` going ``True``
    the moment one message is in flight and undrained is the *expected*
    steady-state case, not a rare failure. Callers that need backpressure
    instead of an exception should check ``full()`` (or use ``try_put()``)
    before calling ``put()``, and fall back to draining the input side first
    -- exactly the pattern ``ReplicaEngineCore._engine_loop`` already has for
    a full ``_batch_queue`` (falls through to ``_await_and_apply_oldest()``).
    Wiring that fallback into ``async_engine.py`` is a separate subproblem;
    this module only provides the non-blocking surface it needs.
    """

    def __init__(
        self,
        output: ChannelOutput,
        *,
        edge_name: str = "",
        max_message_size: int | None = None,
    ) -> None:
        self._output = output
        self._edge_name = edge_name
        self._max_message_size = max_message_size

    def full(self, message_size: int) -> bool:
        """Non-blocking backpressure check: would a push of ``message_size``
        bytes fit right now?

        Raises ``ChannelNotReadyError`` if the channel has not finished its
        startup handshake yet. Never sleeps.
        """
        if not self._output.is_ready():
            raise ChannelNotReadyError(
                f"platform output channel {self._edge_name!r} is not ready "
                "(Platform.start() has not completed the channel handshake yet)"
            )
        return self._output.is_full(message_size)

    def try_put(self, payload: bytes) -> bool:
        """Push ``payload`` if there is room; return ``False`` instead of
        raising if the channel is full. Never sleeps, never retries.

        Still raises ``ChannelNotReadyError`` (not ready) and
        ``MessageTooLargeError`` (payload exceeds ``max_message_size``): both
        are precondition violations, not backpressure, and returning
        ``False`` for either would hide a bug behind the normal
        "try again later" signal.
        """
        message_size = len(payload)
        if self._max_message_size is not None and message_size > self._max_message_size:
            raise MessageTooLargeError(
                f"payload of {message_size} bytes exceeds platform output "
                f"channel {self._edge_name!r}'s configured max_message_size="
                f"{self._max_message_size} bytes; this channel can never "
                "accept it, regardless of backlog"
            )
        if self.full(message_size):
            return False
        self._output.push(payload)
        return True

    def put(self, payload: bytes) -> None:
        """Push ``payload`` onto the channel.

        Raises ``queue.Full`` immediately (zero sleep, zero retry) if the
        channel is currently full; raises ``MessageTooLargeError`` instead if
        the payload can never fit; raises ``ChannelNotReadyError`` if the
        channel has not completed its startup handshake. See the class
        docstring for why this never blocks.
        """
        if not self.try_put(payload):
            raise queue.Full(
                f"platform output channel {self._edge_name!r} is full "
                f"(message_size={len(payload)}); not waiting -- put() never "
                "blocks the event loop. Use full()/try_put() for a "
                "non-blocking backpressure check instead of catching this."
            )


class PlatformInputQueue:
    """Adapts a platform ``Input`` edge to ``get(timeout: float | None) -> bytes``.

    Polls ``Input.has_message()`` and calls ``Input.read()`` (which copies
    the payload out and pops the message, per the contract's Layer 1 rule 1)
    once one is available. Raises ``queue.Empty`` if no message arrives
    within ``timeout`` seconds -- required exactly, since three call sites in
    ``async_engine.py`` catch only ``queue.Empty``, not a bare ``TimeoutError``
    or anything else. ``timeout`` defaults to ``None``, which blocks
    indefinitely (matches stdlib ``queue.Queue.get`` semantics) -- required
    because ``serving_worker.py`` (:232, :281) calls ``get()`` with no
    argument at all. Every call site in this codebase invokes ``get`` inside
    ``asyncio.to_thread`` or on a plain worker thread, so blocking here does
    not stall the asyncio event loop the way ``PlatformOutputQueue.put``
    would.

    Raises ``ChannelNotReadyError`` if the channel has not finished its
    startup handshake -- checked once, before the poll loop starts, matching
    the ordering precondition documented on ``ChannelNotReadyError``.
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

    def get(self, timeout: float | None = None) -> bytes:
        """Return the next message's payload, raising ``queue.Empty`` on timeout."""
        if not self._input.is_ready():
            raise ChannelNotReadyError(
                f"platform input channel {self._edge_name!r} is not ready "
                "(Platform.start() has not completed the channel handshake yet)"
            )
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
