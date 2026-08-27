# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for the queue-shaped platform channel adapter (Layer 2).

Uses ``FakeChannel`` (an in-memory double for the Layer 1 Output/Input
contract) -- no MPI, no native extension, no NPU.
"""

from __future__ import annotations

import queue
import threading
import time

import pytest

from pypto_serving.serving.transport.channel_queue import (
    PlatformInputQueue,
    PlatformOutputQueue,
)

from .fakes import FakeChannel


def test_round_trip_of_opaque_bytes() -> None:
    channel = FakeChannel()
    output = PlatformOutputQueue(channel)
    input_ = PlatformInputQueue(channel)

    payload = b"\x00\x01\xff opaque msgpack bytes \xfe"
    output.put(payload)

    assert input_.get(timeout=1.0) == payload


def test_fifo_order_preserved_across_multiple_messages() -> None:
    channel = FakeChannel()
    output = PlatformOutputQueue(channel)
    input_ = PlatformInputQueue(channel)

    messages = [f"msg-{i}".encode() for i in range(5)]
    for message in messages:
        output.put(message)

    received = [input_.get(timeout=1.0) for _ in messages]
    assert received == messages


def test_get_raises_queue_empty_on_timeout() -> None:
    channel = FakeChannel()
    input_ = PlatformInputQueue(channel, poll_interval=0.001)

    start = time.monotonic()
    with pytest.raises(queue.Empty):
        input_.get(timeout=0.05)
    elapsed = time.monotonic() - start

    # Must actually wait roughly the requested timeout, not return instantly.
    assert elapsed >= 0.04


def test_get_returns_promptly_when_message_already_present() -> None:
    channel = FakeChannel()
    channel.push(b"already-here")
    input_ = PlatformInputQueue(channel, poll_interval=0.001)

    start = time.monotonic()
    result = input_.get(timeout=5.0)
    elapsed = time.monotonic() - start

    assert result == b"already-here"
    # A large timeout must not be consumed when data is immediately available.
    assert elapsed < 0.5


def test_get_blocks_until_message_arrives_within_timeout() -> None:
    channel = FakeChannel()
    input_ = PlatformInputQueue(channel, poll_interval=0.001)

    def _delayed_push() -> None:
        time.sleep(0.05)
        channel.push(b"late-arrival")

    thread = threading.Thread(target=_delayed_push)
    thread.start()
    try:
        result = input_.get(timeout=2.0)
    finally:
        thread.join()

    assert result == b"late-arrival"


def test_put_raises_queue_full_when_channel_stays_full() -> None:
    channel = FakeChannel(capacity=1)
    channel.force_fill(1)
    output = PlatformOutputQueue(channel, put_timeout=0.05, poll_interval=0.001)

    with pytest.raises(queue.Full):
        output.put(b"no-room")


def test_put_waits_for_room_then_succeeds() -> None:
    """put() polls is_full() and only calls push() once there is room --
    never calls push() while full (FakeChannel.push asserts this itself)."""
    channel = FakeChannel(capacity=1)
    channel.force_fill(1)
    output = PlatformOutputQueue(channel, put_timeout=2.0, poll_interval=0.001)

    def _free_room_after_delay() -> None:
        time.sleep(0.05)
        channel.read()  # drains the one slot, freeing room

    thread = threading.Thread(target=_free_room_after_delay)
    thread.start()
    try:
        output.put(b"fits-eventually")
    finally:
        thread.join()

    assert channel.push_calls == [(b"fits-eventually", 0, 0, 0)]


def test_put_uses_push_defaults() -> None:
    channel = FakeChannel()
    output = PlatformOutputQueue(channel)

    output.put(b"payload")

    assert channel.push_calls == [(b"payload", 0, 0, 0)]
