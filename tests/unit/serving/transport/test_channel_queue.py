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
    ChannelNotReadyError,
    MessageTooLargeError,
    PlatformInputQueue,
    PlatformOutputQueue,
)
from pypto_serving.serving.transport.protocols import ChannelInput, ChannelOutput

from .fakes import FakeChannel


def test_fake_channel_conforms_to_the_protocols() -> None:
    # The one thing that guarantees this adapter binds to the real native
    # extension: it must accept anything structurally shaped like the
    # contract, and the contract's shape is expressed as these Protocols.
    channel = FakeChannel()
    assert isinstance(channel, ChannelOutput)
    assert isinstance(channel, ChannelInput)


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


def test_get_default_timeout_is_none_and_blocks_indefinitely() -> None:
    # serving_worker.py:232,281 call get() with no arguments at all -- the
    # default must exist and must mean "block until a message arrives",
    # matching stdlib queue.Queue.get semantics.
    channel = FakeChannel()
    input_ = PlatformInputQueue(channel, poll_interval=0.001)

    def _delayed_push() -> None:
        time.sleep(0.05)
        channel.push(b"no-timeout-given")

    thread = threading.Thread(target=_delayed_push)
    thread.start()
    try:
        result = input_.get()  # no timeout argument
    finally:
        thread.join()

    assert result == b"no-timeout-given"


def test_get_raises_channel_not_ready_error() -> None:
    channel = FakeChannel()
    channel.ready = False
    input_ = PlatformInputQueue(channel)

    with pytest.raises(ChannelNotReadyError):
        input_.get(timeout=0.01)


def test_put_raises_queue_full_immediately_with_zero_wait() -> None:
    """The load-bearing fix: put() must never sleep. A full channel that
    never drains must fail in effectively zero time, not after some bounded
    wait -- any sleep here stalls the asyncio event loop that calls put()
    synchronously."""
    channel = FakeChannel(capacity=1)
    channel.force_fill(1)
    output = PlatformOutputQueue(channel)

    start = time.monotonic()
    with pytest.raises(queue.Full):
        output.put(b"no-room")
    elapsed = time.monotonic() - start

    assert elapsed < 0.01
    # push() must never have been called while full.
    assert channel.push_calls == []


def test_put_uses_push_defaults() -> None:
    channel = FakeChannel()
    output = PlatformOutputQueue(channel)

    output.put(b"payload")

    assert channel.push_calls == [(b"payload", 0, 0, 0)]


def test_put_raises_channel_not_ready_error() -> None:
    channel = FakeChannel()
    channel.ready = False
    output = PlatformOutputQueue(channel)

    with pytest.raises(ChannelNotReadyError):
        output.put(b"payload")


def test_full_is_a_non_blocking_predicate_matching_put_outcome() -> None:
    channel = FakeChannel(capacity=1)
    output = PlatformOutputQueue(channel)

    assert output.full(len(b"x")) is False
    output.put(b"x")
    assert output.full(len(b"y")) is True


def test_full_raises_channel_not_ready_error() -> None:
    channel = FakeChannel()
    channel.ready = False
    output = PlatformOutputQueue(channel)

    with pytest.raises(ChannelNotReadyError):
        output.full(1)


def test_try_put_returns_false_on_full_channel_without_raising() -> None:
    """The non-blocking surface S3 needs: check before dispatching, get False
    back, and fall through to draining instead -- no exception, no wait."""
    channel = FakeChannel(capacity=1)
    channel.force_fill(1)
    output = PlatformOutputQueue(channel)

    start = time.monotonic()
    accepted = output.try_put(b"no-room")
    elapsed = time.monotonic() - start

    assert accepted is False
    assert elapsed < 0.01
    assert channel.push_calls == []


def test_try_put_returns_true_and_pushes_when_room_available() -> None:
    channel = FakeChannel()
    output = PlatformOutputQueue(channel)

    accepted = output.try_put(b"fits")

    assert accepted is True
    assert channel.push_calls == [(b"fits", 0, 0, 0)]


def test_put_raises_message_too_large_error_without_touching_is_full() -> None:
    channel = FakeChannel(capacity=8)
    output = PlatformOutputQueue(channel, max_message_size=4)

    with pytest.raises(MessageTooLargeError):
        output.put(b"way-too-long")

    # Oversize is a hard, permanent condition -- distinct from queue.Full,
    # and detected before ever calling push().
    assert channel.push_calls == []


def test_try_put_raises_message_too_large_error_even_though_channel_has_room() -> None:
    channel = FakeChannel(capacity=8)
    output = PlatformOutputQueue(channel, max_message_size=4)

    # Plenty of room in the fake's ring, but the payload itself can never
    # fit -- must raise, not return False (False would suggest "try again
    # later", which is never true here).
    with pytest.raises(MessageTooLargeError):
        output.try_put(b"way-too-long")


def test_message_too_large_error_is_not_a_queue_full() -> None:
    # Distinguishable by type: callers that catch queue.Full for retry logic
    # must NOT accidentally catch this and retry a payload that can never fit.
    assert not issubclass(MessageTooLargeError, queue.Full)
