# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""In-memory double for a platform SPSC channel, implementing the Layer 1
``Output``/``Input`` contract shape (see ``platform/docs/python-channel-contract.md``)
without MPI, HiCR, or the native extension.

A single ``FakeChannel`` instance exposes both the ``Output`` methods
(``is_full``, ``push``, ``is_ready``) and the ``Input`` methods
(``has_message``, ``read``, ``is_ready``) backed by one FIFO buffer, so it can
be passed directly as both ends of a ``PlatformOutputQueue`` /
``PlatformInputQueue`` pair in tests.
"""

from __future__ import annotations

from collections import deque


class FakeChannel:
    """One FIFO exposing both channel ends.

    ``is_full`` models BOTH dimensions of ``Output::isFull``, because the real
    one is two-dimensional -- it checks the metadata channel's slot count and
    then the payload ring's free bytes -- and the sizing rule the serving layer
    depends on (``Buffer Size = capacity x max_message_bytes``, which is what
    makes ``full_for_worst_case()`` exact) lives entirely in the byte dimension.
    A slot-count-only double cannot test it.

    ``buffer_size=None`` keeps the byte dimension unbounded, which is the right
    default for tests that only care about slots.
    """

    def __init__(self, capacity: int = 8, buffer_size: int | None = None) -> None:
        self._buf: deque[bytes] = deque()
        self._capacity = capacity
        self._buffer_size = buffer_size
        # Recorded for assertions: every push() call's full argument tuple.
        self.push_calls: list[tuple[bytes, int, int, int]] = []
        self.ready = True

    # -- Output surface -----------------------------------------------------
    def used_bytes(self) -> int:
        """Bytes currently occupying the payload ring."""
        return sum(len(payload) for payload in self._buf)

    def is_full(self, message_size: int) -> bool:
        if len(self._buf) >= self._capacity:
            return True
        if self._buffer_size is None:
            return False
        return self.used_bytes() + message_size > self._buffer_size

    def push(
        self,
        payload: bytes,
        message_type: int = 0,
        group_id: int = 0,
        sequence_id: int = 0,
    ) -> None:
        if self.is_full(len(payload)):
            raise AssertionError(
                "push() called on a full FakeChannel -- the adapter under test "
                "should have polled is_full() and waited"
            )
        payload_bytes = bytes(payload)
        self.push_calls.append((payload_bytes, message_type, group_id, sequence_id))
        self._buf.append(payload_bytes)

    # -- Input surface --------------------------------------------------
    def has_message(self) -> bool:
        return bool(self._buf)

    def read(self) -> bytes:
        return self._buf.popleft()

    # -- Shared ---------------------------------------------------------
    def is_ready(self) -> bool:
        return self.ready

    def force_fill(self, count: int, payload: bytes = b"x") -> None:
        """Directly stuff the buffer, bypassing push()'s capacity check --
        used to put the channel in a full state for full-channel tests."""
        for _ in range(count):
            self._buf.append(payload)
