# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""What changes in ``ReplicaEngineCore`` when the transport is a bounded channel.

Three behaviours, none of which the ``mp.Queue`` path can exhibit:
backpressure instead of blocking, a shutdown command that cannot be dropped,
and an engine loop whose death is loud.
"""

from __future__ import annotations

import asyncio
import queue
import signal
from collections import deque
from types import SimpleNamespace

import pytest

from pypto_serving.serving.engine.async_engine import (
    ProfilingUnsupportedError,
    ReplicaEngineCore,
    TokenOutput,
)
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.serving.sched.scheduler import Scheduler, SchedulerConfig
from pypto_serving.serving.transport.channel_queue import PlatformOutputQueue
from pypto_serving.serving.transport.selection import TransportKind, resolve_transport_kind
from ..transport.fakes import FakeChannel


def _core(*, input_queue=None) -> ReplicaEngineCore:
    """A minimally wired core: enough state for the paths under test, nothing more."""
    manager = KvCacheManager(num_blocks=32, block_size=2, enable_prefix_cache=False)
    core = ReplicaEngineCore.__new__(ReplicaEngineCore)
    core.scheduler = Scheduler(
        SchedulerConfig(enable_prefix_cache=False, async_scheduling=True), manager
    )
    core.kv_cache_manager = manager
    core.config = SimpleNamespace(executor_cls="PyptoQwen14BExecutor", engine_loop_interval=0.001)
    core._async_scheduling = True
    core._pending_free_ids = []
    core._worker_known_req_ids = set()
    core._request_contexts = {}
    core._batch_queue = deque()
    core._discard_result_step_ids = set()
    core._step_timeout = 30.0
    core._max_in_flight = 2
    core._step_counter = 0
    core._running = False
    core._loop_task = None
    core._loop_failure = None
    core._worker_process = None
    core._transport_kind = TransportKind.PLATFORM
    core._input_queue = input_queue
    core._output_queue = None
    core._profile_output_queue = None
    core._profile_lock = asyncio.Lock()
    return core


def _platform_output(capacity: int = 1) -> tuple[FakeChannel, PlatformOutputQueue]:
    channel = FakeChannel(capacity=capacity)
    return channel, PlatformOutputQueue(channel, edge_name="commands", max_message_size=64)


# -- backpressure -----------------------------------------------------------


def test_a_full_command_channel_refuses_dispatch_without_scheduling(monkeypatch):
    """The check has to come before ``scheduler.schedule()``: schedule()
    allocates blocks and promotes requests, and there is no way to undo that if
    the dispatch is then refused."""
    channel, output = _platform_output(capacity=1)
    channel.force_fill(1)
    core = _core(input_queue=output)

    def _fail(*args, **kwargs):
        raise AssertionError("schedule() must not run when the channel is full")

    monkeypatch.setattr(core.scheduler, "schedule", _fail)
    assert core._try_dispatch_step() is False


def test_dispatch_resumes_once_the_consumer_drains(monkeypatch):
    channel, output = _platform_output(capacity=1)
    channel.force_fill(1)
    core = _core(input_queue=output)
    assert core._command_channel_full() is True

    channel.read()
    assert core._command_channel_full() is False


def test_the_queue_transport_never_reports_backpressure():
    """An mp.Queue is unbounded, so today's path must be untouched: no
    ``full_for_worst_case`` attribute means the answer is always False."""
    core = _core(input_queue=SimpleNamespace(put=lambda payload: None))
    assert core._command_channel_full() is False


def test_a_full_channel_defers_the_cleanup_step_instead_of_dropping_frees():
    """``_flush_pending_frees`` must leave the ids pending, or an aborted
    request's device slot stays pinned until unrelated work carries it."""
    channel, output = _platform_output(capacity=1)
    channel.force_fill(1)
    core = _core(input_queue=output)
    core._pending_free_ids = ["req-a"]

    asyncio.run(core._flush_pending_frees())

    assert core._pending_free_ids == ["req-a"]
    assert core._step_counter == 0


# -- shutdown ---------------------------------------------------------------


def test_the_shutdown_command_waits_for_room_rather_than_being_dropped():
    """``_shutdown_worker`` suppresses every exception, so a plain
    ``queue.Full`` here would silently leave the worker rank running forever."""
    channel, output = _platform_output(capacity=1)
    channel.force_fill(1)
    core = _core(input_queue=output)

    with pytest.raises(queue.Full):
        core._put_shutdown_command(output, timeout=0.05)

    channel.read()
    core._put_shutdown_command(output, timeout=0.5)
    assert channel.has_message()


def test_shutdown_runs_on_the_calling_thread_for_platform_channels(monkeypatch):
    """``is_full``/``push`` on one edge must stay on one thread, and every other
    put on ``commands`` happens on the event loop thread."""
    channel, output = _platform_output(capacity=2)
    core = _core(input_queue=output)

    def _unexpected(*args, **kwargs):
        raise AssertionError("the platform path must not hop threads to shut down")

    monkeypatch.setattr(asyncio, "to_thread", _unexpected)
    asyncio.run(core._shutdown_worker_async(timeout=1.0))
    assert channel.has_message()


# -- profiling --------------------------------------------------------------


def test_profiling_is_refused_loudly_rather_than_multiplexed():
    core = _core(input_queue=_platform_output()[1])
    # A dedicated type, so the HTTP layer answers 400 (you asked for something
    # this deployment does not offer) rather than 500 (the server broke).
    with pytest.raises(ProfilingUnsupportedError, match="not available on the platform"):
        asyncio.run(core._set_profile_active(True))


# -- the engine loop must fail loudly ---------------------------------------


def test_a_dying_engine_loop_takes_the_replica_down(monkeypatch):
    """Reproduces the reviewed failure: the loop task died, its exception was
    never retrieved, ``_running`` stayed True and the replica kept accepting
    HTTP while scheduling nothing."""
    core = _core()
    core._running = True

    ctx = SimpleNamespace(queue=None, request=SimpleNamespace(request_id="req-a"))
    signals: list[int] = []
    monkeypatch.setattr("os.kill", lambda pid, sig: signals.append(sig))

    async def _boom() -> None:
        raise RuntimeError("channel exploded")

    monkeypatch.setattr(core, "_engine_loop", _boom)

    async def _drive() -> None:
        ctx.queue = asyncio.Queue()
        core._request_contexts["req-a"] = ctx
        task = asyncio.create_task(core._run_engine_loop())
        await task
        # Not re-raised: that is what makes it retrieved rather than orphaned.
        assert task.exception() is None

    asyncio.run(_drive())

    assert core._running is False
    assert isinstance(core._loop_failure, RuntimeError)
    assert signals == [signal.SIGTERM]
    output: TokenOutput = ctx.queue.get_nowait()
    # Must be a reason the HTTP mapping understands. An unknown one falls through
    # to "stop", i.e. the client is told its request completed normally.
    assert output.finished and output.finish_reason == "FINISHED_ABORTED"
    from pypto_serving.serving.server.server import ServingServer

    assert ServingServer._map_finish_reason(output.finish_reason) == "aborted"
    assert core.loop_failure is core._loop_failure
    assert core._request_contexts == {}


# -- the queue transport must be bit-for-bit what it was --------------------


def test_the_default_transport_still_spawns_a_worker_process(monkeypatch):
    """PYPTO_SERVING_TRANSPORT unset means QUEUE, and QUEUE means spawn_worker
    with its six-tuple of mp primitives -- no channels anywhere near it."""
    monkeypatch.delenv("PYPTO_SERVING_TRANSPORT", raising=False)
    core = _core()
    core._transport_kind = resolve_transport_kind()
    assert core._transport_kind is TransportKind.QUEUE
    assert core._uses_platform_transport is False

    sentinel = ("process", "in", "out", "profile", "ready", "pages")
    calls: list[object] = []

    def _spawn(config):
        calls.append(config)
        return sentinel

    monkeypatch.setattr("pypto_serving.serving.engine.async_engine.spawn_worker", _spawn)
    assert core._acquire_worker_endpoints() == sentinel
    assert calls == [core.config]


def test_the_default_transport_still_reaps_the_worker_off_the_event_loop(monkeypatch):
    """``process.join()`` blocks, so the queue path must keep using
    ``asyncio.to_thread`` -- only the platform path runs shutdown inline."""
    monkeypatch.delenv("PYPTO_SERVING_TRANSPORT", raising=False)
    core = _core(input_queue=SimpleNamespace(put=lambda payload: None))
    core._transport_kind = resolve_transport_kind()

    offloaded: list[object] = []
    real_to_thread = asyncio.to_thread

    async def _record(func, *args, **kwargs):
        offloaded.append(func)
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _record)
    asyncio.run(core._shutdown_worker_async(timeout=1.0))
    assert len(offloaded) == 1, "the queue path must still hand shutdown to a thread"


def test_the_default_transport_delivers_shutdown_with_a_plain_put(monkeypatch):
    """No retry loop, no timeout bookkeeping: an mp.Queue is unbounded."""
    monkeypatch.delenv("PYPTO_SERVING_TRANSPORT", raising=False)
    puts: list[bytes] = []
    core = _core(input_queue=SimpleNamespace(put=puts.append))
    core._transport_kind = resolve_transport_kind()

    core._put_shutdown_command(core._input_queue, timeout=0.0)
    assert len(puts) == 1, "a zero timeout must not stop the unbounded path putting"


# -- a step that can never fit must not take the replica down ---------------


def test_an_oversized_step_fails_its_requests_not_the_replica(monkeypatch):
    """MessageTooLargeError is raised after schedule() has already committed, so
    it cannot simply be returned from -- but killing the replica over one
    runaway request is the wrong blast radius."""
    from pypto_serving.serving.sched.scheduler import Request, RequestStatus
    from pypto_serving.serving.transport.channel_queue import MessageTooLargeError

    channel, output = _platform_output(capacity=4)
    # Anything the engine builds will exceed this.
    output._max_message_size = 8
    core = _core(input_queue=output)

    request = Request(
        request_id="req-big",
        prompt_token_ids=[1, 2, 3, 4],
        max_new_tokens=4,
        status=RequestStatus.WAITING,
    )
    ctx = SimpleNamespace(queue=asyncio.Queue(), request=request)
    core._request_contexts["req-big"] = ctx
    core.scheduler.add_request(request)

    assert core._try_dispatch_step() is False
    # The replica is still running and the channel is untouched.
    assert core._loop_failure is None
    assert channel.push_calls == []
    assert not channel.has_message()
    # The request was failed, with a reason the HTTP mapping understands.
    output_token: TokenOutput = ctx.queue.get_nowait()
    assert output_token.finished
    assert output_token.finish_reason == "error"
    assert core._batch_queue == deque()
    # Sanity: the adapter really did raise the containment-worthy error.
    with pytest.raises(MessageTooLargeError):
        output.put(b"x" * 9)


def test_shutdown_delivery_is_reported_to_the_launcher():
    """platform_launch refuses to exit normally without this signal."""
    channel, output = _platform_output(capacity=2)
    core = _core(input_queue=output)

    notified: list[bool] = []
    output.note_shutdown_sent = lambda: notified.append(True)

    core._put_shutdown_command(output, timeout=1.0)
    assert notified == [True]
    assert channel.has_message()
