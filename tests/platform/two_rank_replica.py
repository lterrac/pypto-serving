# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""End-to-end proof that a serving replica runs over platform channels.

Run it exactly the way the CLI runs::

    OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \\
    PYPTO_SERVING_TRANSPORT=platform \\
    mpirun -np 2 --oversubscribe -x PYTHONPATH -x PYPTO_SERVING_TRANSPORT \\
      python tests/platform/two_rank_replica.py

Rank 0 runs a real ``ReplicaEngineCore``; rank 1 runs the real serving worker
busy loop. Commands cross the ``commands`` edge and results come back over
``results``, both created by ``platform`` and merely obtained by the serving
layer -- which is the whole point of the exercise.

WHAT IS REAL HERE (do not take this list on trust -- everything named is
imported from the product modules below, not reimplemented):

* ``platform`` bring-up: ``Runtime``/``Deployment``/``Platform``, the claim
  agreement, the memory-slot exchange, ``wait_until_ready``, ordered teardown --
  all of it through ``run_platform_replica``, the same function
  ``pypto_serving.cli.main`` calls.
* The two channels, and the queue-shaped adapters over them.
* ``ReplicaEngineCore``: ``start()`` (including the startup handshake and
  ``KvCacheManager.initialize``), ``_engine_loop``, ``_try_dispatch_step``,
  ``_build_step_command``, ``_await_and_apply_oldest``, ``_process_step_output``,
  ``_flush_pending_frees``, ``stop()`` -- unmodified, on the async (pipelined)
  path.
* The real ``Scheduler`` and ``KvCacheManager``.
* The real msgpack IPC codec (``encode_command``/``decode_command``/
  ``encode_result``).
* The real worker: ``run_worker_over_channels`` and, depending on the mode,
  either ``WorkerProcess._serial_busy_loop`` or the production
  ``_pipelined_busy_loop`` (three FIFO lanes, results pushed from the
  ``pypto-output`` thread) -> ``_run_step_command`` ->
  ``_apply_command_lifecycle``, including ``ShutdownCommand`` handling and the
  request cache / last-token bookkeeping.

WHAT IS STUBBED, and only this:

1. ``_StubWorker.init_device_and_model`` -- returns a fixed KV page count instead
   of loading a checkpoint onto an NPU. This test must not touch a device.
2. ``_StubWorker._execute_step`` -- the single method that runs the model.
   It samples ``last_token + 1`` per request (resolving ``PLACEHOLDER_TOKEN``
   from the worker's own ``_last_tokens`` cache exactly as the real path does),
   so the returned tokens are deterministic and checkable.
3. ``_StubTokenizer`` -- prompt encoding and detokenisation, so no tokenizer
   files are needed.
4. ``_StubExecutor`` (pipelined mode only) -- three capability booleans. It is
   never called: with ``supports_device_decode_embedding`` False,
   ``_prepare_step_command`` returns before it touches the executor or the model.

MODES, all off by default, each selected by an environment variable:

* ``TWO_RANK_FORCE_BACKPRESSURE=1`` -- raise the engine's ``_max_in_flight``
  above the channel capacity so the bounded ring genuinely refuses dispatches
  and the loop falls through to ``_await_and_apply_oldest()``.
* ``TWO_RANK_PIPELINED_WORKER=1`` -- run the worker's production pipelined busy
  loop instead of the serial one.
* ``TWO_RANK_INJECT_LOOP_FAILURE=1`` -- break the engine loop and assert the
  replica notices, fails its requests, and exits non-zero.
* ``TWO_RANK_REPRO_F2=1`` -- reproduce the reviewed "worker wedged in a blocking
  push while the engine stopped draining" hang, and assert it now terminates.
* ``TWO_RANK_REPRO_F3=1`` -- reproduce the reviewed "engine returned without a
  ShutdownCommand" hang, and assert it now terminates.
* ``TWO_RANK_REPRO_OVERSIZED=1`` -- with a deliberately tiny
  ``PYPTO_SERVING_PLATFORM_MAX_MESSAGE_BYTES``, assert that a StepCommand which
  can never fit fails its REQUESTS and leaves the replica serving, instead of
  killing it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from pypto_serving.config.types import GenerateConfig, RuntimeConfig  # noqa: E402
from pypto_serving.serving.engine.async_engine import EngineConfig, ReplicaEngineCore  # noqa: E402
from pypto_serving.serving.server.ipc import (  # noqa: E402
    PLACEHOLDER_TOKEN,
    StepCommand,
    StepResult,
    encode_command,
)
from pypto_serving.serving.server.serving_worker import (  # noqa: E402
    WorkerProcess,
    run_worker_over_channels,
)
from pypto_serving.serving.transport.platform_launch import (  # noqa: E402
    ChannelSettings,
    WorkerEndpoints,
    run_platform_replica,
)

FORCE_BACKPRESSURE_ENV_VAR = "TWO_RANK_FORCE_BACKPRESSURE"
PIPELINED_WORKER_ENV_VAR = "TWO_RANK_PIPELINED_WORKER"
INJECT_LOOP_FAILURE_ENV_VAR = "TWO_RANK_INJECT_LOOP_FAILURE"
REPRO_F2_ENV_VAR = "TWO_RANK_REPRO_F2"
REPRO_F3_ENV_VAR = "TWO_RANK_REPRO_F3"
REPRO_OVERSIZED_ENV_VAR = "TWO_RANK_REPRO_OVERSIZED"

STUB_NUM_PAGES = 64
PAGE_SIZE = 16
PROMPT_TOKENS = [11, 12, 13, 14, 15]
MAX_NEW_TOKENS = 6
NUM_REQUESTS = 3

logger = logging.getLogger("two_rank_replica")


def _enabled(name: str) -> bool:
    return os.environ.get(name) == "1"


class _StubTokenizer:
    """Stub 1 of 3: prompt encoding and detokenisation without tokenizer files."""

    eos_token_id = None
    bos_token_id = None

    def encode(self, text: str) -> list[int]:
        return list(PROMPT_TOKENS)

    def decode(self, token_ids) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


class _StubExecutor:
    """Stub 4 of 4, pipelined mode only: the capability booleans the busy loop reads.

    ``supports_async_decode_prepare`` True selects ``_pipelined_busy_loop``, the
    production path under async scheduling. The other two are False so
    ``_prepare_step_command`` returns ``None`` before it dereferences the
    executor or the model record -- which is what lets the pipelined lanes run
    with nothing loaded on a device.
    """

    supports_async_decode_prepare = True
    supports_device_decode_embedding = False
    supports_async_decode_reclaim = False


class _StubWorker(WorkerProcess):
    """The real worker with its two device-bound methods replaced.

    Everything the busy loop does around these -- decode the command, apply the
    lifecycle deltas, encode and push the result, honour ShutdownCommand -- is
    inherited unchanged from ``WorkerProcess``.
    """

    def init_device_and_model(self) -> int:
        """Stub 2 of 4: no checkpoint, no NPU. Report a fixed page count."""
        logger.info("stub worker: reporting %d KV pages (no device touched)", STUB_NUM_PAGES)
        # The real method sets this from the loaded model's runtime config.
        self._page_size = PAGE_SIZE
        if _enabled(PIPELINED_WORKER_ENV_VAR):
            self.executor = _StubExecutor()
            logger.info("stub worker: pipelined busy loop selected")
        return STUB_NUM_PAGES

    def _execute_step(self, cmd: StepCommand, prepared_decode=None) -> StepResult:
        """Stub 3 of 4: the one method that runs the model.

        Samples ``last_token + 1``, resolving ``PLACEHOLDER_TOKEN`` from
        ``self._last_tokens`` the way the real decode path does, so the engine's
        async-scheduling placeholder protocol is genuinely exercised.
        """
        new_tokens: dict[str, list[int]] = {}
        for prefill in cmd.prefill_requests:
            cached = self._req_cache[prefill.request_id]
            computed = prefill.num_computed_tokens + len(prefill.chunk_tokens)
            if computed < len(cached.prompt_token_ids):
                continue  # a chunk that does not complete the prompt samples nothing
            new_tokens[prefill.request_id] = [prefill.chunk_tokens[-1] + 1]
        for decode in cmd.decode_requests:
            last = decode.last_token
            if last == PLACEHOLDER_TOKEN:
                previous = self._last_tokens.get(decode.request_id)
                if not previous:
                    raise AssertionError(
                        f"placeholder for {decode.request_id} with no cached token"
                    )
                last = previous[-1]
            new_tokens[decode.request_id] = [last + 1]
        for request_id, tokens in new_tokens.items():
            self._record_last_tokens(request_id, tokens)
        return StepResult(new_tokens=new_tokens, step_id=cmd.step_id)


def _engine_config() -> EngineConfig:
    """The EngineConfig both ranks build. Identical by construction (SPMD)."""
    return EngineConfig(
        model_id="stub-model",
        model_dir="",
        runtime_config=RuntimeConfig(page_size=PAGE_SIZE, max_seq_len=256, max_batch_size=8),
        max_num_running_reqs=4,
        max_num_scheduled_tokens=64,
        long_prefill_token_threshold=64,
        enable_prefix_cache=False,
        async_scheduling=True,
        device_ids=(0,),
    )


async def _drive_requests(core: ReplicaEngineCore) -> None:
    """Send real requests through the real engine and check what comes back."""
    generate = GenerateConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
        stream=True,
        ignore_eos=True,
    )
    for index in range(NUM_REQUESTS):
        request_id = core.generate_request_id()
        tokens: list[int] = []
        finish_reason = ""
        async for output in core.add_request(
            request_id,
            "irrelevant",
            generate,
            prompt_token_ids=list(PROMPT_TOKENS),
        ):
            if output.token_id is not None:
                tokens.append(output.token_id)
            if output.finished:
                finish_reason = output.finish_reason
        expected = [PROMPT_TOKENS[-1] + 1 + step for step in range(MAX_NEW_TOKENS)]
        print(
            f"[engine rank] request {index}: tokens={tokens} "
            f"finish_reason={finish_reason!r}",
            flush=True,
        )
        if tokens != expected:
            raise AssertionError(f"request {index}: got {tokens}, expected {expected}")


async def _assert_a_dying_loop_is_loud(core: ReplicaEngineCore) -> None:
    """Break the engine loop on purpose and check the replica does not pretend.

    The reviewed failure was silent: the loop task died with its exception never
    retrieved, ``_running`` stayed True, and the replica kept accepting HTTP.
    """
    delivered: list[int] = []
    previous = signal.signal(signal.SIGTERM, lambda number, frame: delivered.append(number))
    try:

        def _boom() -> bool:
            raise RuntimeError("injected engine loop failure")

        core._try_dispatch_step = _boom  # noqa: SLF001 -- fault injection is the point

        generate = GenerateConfig(max_new_tokens=1, temperature=0.0, stream=True, ignore_eos=True)
        request_id = core.generate_request_id()
        outputs = [
            output
            async for output in core.add_request(
                request_id, "irrelevant", generate, prompt_token_ids=list(PROMPT_TOKENS)
            )
        ]
    finally:
        signal.signal(signal.SIGTERM, previous)

    print(
        f"[engine rank] injected failure: _running={core._running} "
        f"loop_failure={core.loop_failure!r} sigterm={delivered} "
        f"final_output={outputs[-1].finish_reason!r}",
        flush=True,
    )
    if core._running is not False:
        raise AssertionError("_running stayed True after the loop died")
    if not isinstance(core.loop_failure, RuntimeError):
        raise AssertionError(f"loop failure not recorded: {core.loop_failure!r}")
    if delivered != [signal.SIGTERM]:
        raise AssertionError(f"the replica did not ask to shut down: {delivered}")
    if not outputs or outputs[-1].finish_reason != "FINISHED_ABORTED":
        raise AssertionError(f"in-flight request was not failed: {outputs!r}")


async def _assert_an_oversized_step_only_fails_its_requests(
    core: ReplicaEngineCore,
) -> None:
    """A step larger than the edge can carry must not take the replica down.

    ``MessageTooLargeError`` surfaces after ``schedule()`` has already committed
    scheduler state, so it cannot simply be returned from -- but the reviewed
    behaviour, killing the whole replica over one runaway request, is the wrong
    blast radius. ``_try_dispatch_step`` now routes it through
    ``_handle_step_error``, which is the machinery for "this batch is gone".
    """
    generate = GenerateConfig(
        max_new_tokens=MAX_NEW_TOKENS, temperature=0.0, stream=True, ignore_eos=True
    )
    request_id = core.generate_request_id()
    outputs = [
        output
        async for output in core.add_request(
            request_id, "irrelevant", generate, prompt_token_ids=list(PROMPT_TOKENS)
        )
    ]
    print(
        f"[engine rank] oversized step: final_output={outputs[-1].finish_reason!r} "
        f"running={core._running} loop_failure={core.loop_failure!r}",
        flush=True,
    )
    if outputs[-1].finish_reason != "error":
        raise AssertionError(
            f"expected the request to be failed, got {outputs[-1].finish_reason!r}"
        )
    if core.loop_failure is not None:
        raise AssertionError(f"the replica died with it: {core.loop_failure!r}")
    if core._running is not True:
        raise AssertionError("the replica stopped serving")


async def _engine_async_main() -> int:
    core = ReplicaEngineCore(_engine_config(), _StubTokenizer())
    print("[engine rank] starting ReplicaEngineCore over platform channels", flush=True)
    await core.start()

    # Instrumentation only: count how often the bounded 'commands' ring made
    # _try_dispatch_step return False, i.e. how often the loop fell through to
    # _await_and_apply_oldest() instead of dispatching.
    backpressure = {"count": 0}
    real_check = core._command_channel_full

    def _counting_check() -> bool:
        full = real_check()
        backpressure["count"] += int(full)
        return full

    core._command_channel_full = _counting_check

    if _enabled(FORCE_BACKPRESSURE_ENV_VAR):
        # The default sizing makes backpressure unreachable on purpose: capacity
        # >= _max_in_flight means a healthy pipeline never fills the ring. To
        # exercise the refusal on real channels, let the engine try to keep more
        # steps in flight than the ring has slots. Raising _max_in_flight is the
        # honest way to do that -- lowering the capacity below _max_in_flight is
        # now rejected outright, because that configuration can wedge the worker
        # in a blocking push with nobody left to drain it.
        capacity = ChannelSettings.from_env().capacity
        core._max_in_flight = capacity + 2
        print(
            f"[engine rank] forcing backpressure: _max_in_flight="
            f"{core._max_in_flight} against a {capacity}-slot command ring",
            flush=True,
        )

    # start() consumed the opaque startup handshake off the 'results' edge and
    # sized the KV pool from it: proof that ready/num_pages crossed the ranks.
    reported_blocks = len(core.kv_cache_manager.blocks)
    print(f"[engine rank] KV blocks from the worker handshake: {reported_blocks}", flush=True)
    if reported_blocks != STUB_NUM_PAGES:
        raise AssertionError(f"expected {STUB_NUM_PAGES} KV blocks, got {reported_blocks}")

    if _enabled(REPRO_OVERSIZED_ENV_VAR):
        await _assert_an_oversized_step_only_fails_its_requests(core)
        await core.stop()
        print(
            "[engine rank] OK: an unfittable step failed its requests, not the replica",
            flush=True,
        )
        return 0

    if _enabled(INJECT_LOOP_FAILURE_ENV_VAR):
        await _assert_a_dying_loop_is_loud(core)
        await core.stop()
        print("[engine rank] OK: a dying engine loop took the replica down", flush=True)
        # Non-zero on purpose: the process exit status is the only thing a
        # supervisor sees, and a replica whose engine loop died must not look
        # like a clean shutdown.
        return 1

    try:
        await _drive_requests(core)
    finally:
        await core.stop()

    if core.loop_failure is not None:
        raise AssertionError(f"engine loop failed: {core.loop_failure!r}")
    print(
        f"[engine rank] backpressure: the command channel was full "
        f"{backpressure['count']} time(s); every one of those returned False from "
        "_try_dispatch_step and fell through to _await_and_apply_oldest()",
        flush=True,
    )
    print("[engine rank] OK: commands and results crossed the platform channels", flush=True)
    return 0


def _repro_f2_engine_main() -> int:
    """Stop draining 'results' while the worker keeps producing.

    The reviewed hang: the worker enters ``pushMessageLocking`` on a full ring
    with the GIL released and SIGINT ignored, and the only thing that could free
    it -- the engine reading -- never happens. Nothing could interrupt it and the
    job had to be SIGKILLed.

    Here rank 0 stuffs the command ring and then simply waits. It never reads a
    single result. The bounded put on rank 1 must therefore give up and take the
    job down; if it does not, the harness's own sleep expires and the test's
    timeout catches the hang.
    """
    from pypto_serving.serving.transport.platform_launch import current_engine_endpoints

    endpoints = current_engine_endpoints()
    settings = ChannelSettings.from_env()
    print(
        f"[engine rank] F2 repro: stuffing the command ring "
        f"(capacity {settings.capacity}, put timeout {settings.put_timeout_s:g}s) "
        "and never reading 'results'",
        flush=True,
    )
    # Consume the startup handshake message so the worker's first real push is
    # the one that has to wait; after that, nothing on this rank ever reads.
    endpoints.handshake.wait(timeout=60.0)

    # The command ring only holds `capacity` at a time, so delivering more than
    # that means retrying as the worker pops them. It has to be more: the worker
    # wedges on the push of result number capacity+1, so it must be given
    # capacity+1 commands, which it can only receive by draining some first.
    wanted = settings.capacity + 3
    accepted = 0
    deadline = time.monotonic() + 30.0
    step_id = 0
    while accepted < wanted and time.monotonic() < deadline:
        step_id += 1
        command = StepCommand(
            new_requests=[],
            prefill_requests=[],
            decode_requests=[],
            finished_request_ids=[],
            step_id=step_id,
        )
        payload = encode_command(command)
        if endpoints.commands.try_put(payload):
            accepted += 1
        else:
            time.sleep(0.005)
    print(
        f"[engine rank] F2 repro: {accepted}/{wanted} command(s) accepted; the worker "
        "is now wedged pushing a result nobody will read. Idling.",
        flush=True,
    )
    if accepted < wanted:
        raise AssertionError(
            f"F2 repro could not deliver {wanted} commands (got {accepted}); "
            "the worker stopped draining earlier than the scenario needs"
        )
    # Long enough that a hang is unmistakable, short enough that the test's own
    # timeout is not what proves the point: rank 1's bounded put fires first.
    time.sleep(60.0)
    raise AssertionError(
        "F2 repro: rank 1 never failed its blocking push, so the wedge is back"
    )


def _repro_f3_engine_main() -> int:
    """Return cleanly without ever telling the worker to stop.

    Stands in for ``run_serve`` returning without its lifespan shutdown having
    run -- a uvicorn startup-event failure being the obvious way in. Previously
    this left rank 1 in ``get(timeout=None)`` forever and the job had to be
    SIGKILLed; ``_run_role`` must now treat it as a failure.
    """
    print(
        "[engine rank] F3 repro: returning without a ShutdownCommand", flush=True
    )
    return 0


def _engine_main() -> int:
    if _enabled(REPRO_F2_ENV_VAR):
        return _repro_f2_engine_main()
    if _enabled(REPRO_F3_ENV_VAR):
        return _repro_f3_engine_main()
    return asyncio.run(_engine_async_main())


def _worker_main(endpoints: WorkerEndpoints) -> None:
    run_worker_over_channels(
        _engine_config(),
        endpoints.commands,
        endpoints.results,
        worker_factory=_StubWorker,
    )
    print("[worker rank] busy loop exited on ShutdownCommand", flush=True)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return run_platform_replica(
        _engine_config(),
        engine_main=_engine_main,
        worker_main=_worker_main,
    )


if __name__ == "__main__":
    sys.exit(main())
