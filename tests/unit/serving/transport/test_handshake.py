# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The startup handshake that replaces ``ready_event`` / ``num_pages_value``."""

from __future__ import annotations

import queue

import pytest

from pypto_serving.serving.transport.channel_queue import PlatformInputQueue
from pypto_serving.serving.transport.handshake import (
    HANDSHAKE_PROTOCOL_VERSION,
    PlatformStartupHandshake,
    WorkerHandshake,
    decode_handshake,
    encode_handshake,
)
from .fakes import FakeChannel


def _handshake_over(channel: FakeChannel) -> PlatformStartupHandshake:
    return PlatformStartupHandshake(
        PlatformInputQueue(channel, edge_name="results", poll_interval=0.0),
        edge_name="results",
    )


def test_round_trips_the_page_count():
    payload = encode_handshake(WorkerHandshake(num_pages=1234))
    assert decode_handshake(payload).num_pages == 1234


def test_rejects_a_protocol_mismatch():
    payload = encode_handshake(
        WorkerHandshake(protocol=HANDSHAKE_PROTOCOL_VERSION + 1, num_pages=1)
    )
    with pytest.raises(RuntimeError, match="protocol"):
        decode_handshake(payload)


def test_satisfies_both_the_event_and_the_value_shapes():
    """``start()`` calls ``ready_event.wait(timeout=...)`` then reads
    ``num_pages_value.value``; one object has to answer both."""
    channel = FakeChannel()
    channel.push(encode_handshake(WorkerHandshake(num_pages=99)))
    handshake = _handshake_over(channel)

    assert handshake.wait(timeout=1.0) is True
    assert handshake.value == 99


def test_wait_returns_false_on_timeout_rather_than_raising():
    """``start()`` turns a False into its own 'worker failed to initialize
    within N s' error, so a timeout must not surface as queue.Empty."""
    handshake = _handshake_over(FakeChannel())
    assert handshake.wait(timeout=0.01) is False


def test_page_count_is_not_readable_before_the_handshake_arrives():
    handshake = _handshake_over(FakeChannel())
    with pytest.raises(RuntimeError, match="has not been received"):
        _ = handshake.value


def test_a_failed_worker_raises_instead_of_reporting_zero_pages():
    """The process path sets ready_event even when init fails and leaves
    num_pages at 0, which initialises a zero-block KV cache. The rank path
    must not repeat that."""
    channel = FakeChannel()
    channel.push(encode_handshake(WorkerHandshake(error="checkpoint missing")))
    handshake = _handshake_over(channel)

    with pytest.raises(RuntimeError, match="checkpoint missing"):
        handshake.wait(timeout=1.0)


def test_reading_the_handshake_leaves_the_edge_empty_for_step_results():
    """It is one message on the same edge StepResults use; it must be popped."""
    channel = FakeChannel()
    channel.push(encode_handshake(WorkerHandshake(num_pages=8)))
    handshake = _handshake_over(channel)
    assert handshake.wait(timeout=1.0) is True

    with pytest.raises(queue.Empty):
        PlatformInputQueue(channel, poll_interval=0.0).get(timeout=0.01)


def test_the_handshake_can_only_be_consumed_once():
    """Unlike an mp.Event this pops a message off a shared edge, so a second
    wait() would eat the next StepResult and decode it as a handshake."""
    channel = FakeChannel()
    channel.push(encode_handshake(WorkerHandshake(num_pages=8)))
    handshake = _handshake_over(channel)

    assert handshake.wait(timeout=1.0) is True
    with pytest.raises(RuntimeError, match="already been consumed"):
        handshake.wait(timeout=1.0)
