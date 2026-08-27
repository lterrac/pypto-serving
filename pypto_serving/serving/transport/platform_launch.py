# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""SPMD bring-up of one serving replica on platform channels (Layer 3).

This is the piece that inverts the ownership: today the serving layer creates
its own IPC (``spawn_worker`` makes the ``multiprocessing.Queue`` pair), and
here the *platform* creates the channels and the serving layer obtains them.

The shape is forced, not chosen -- see ``platform/docs/python-channel-contract.md``
"Why the Python process must be the MPI rank". A channel is nine globally
exchanged HiCR memory slots, i.e. MPI RMA windows over process heap; there is
no handle to inherit or pass, and the MPI backend cannot spawn instances at
runtime. So the only workable launch is SPMD::

    mpirun -np 2 python -m pypto_serving.cli --model ...

Every rank runs the same argv and builds the same ``EngineConfig``. Rank 0 (the
root instance) takes the ``engine`` partition and runs the API server and the
engine loop exactly as it does today; rank 1 takes the ``worker`` partition and
runs the worker busy loop *instead of* rank 0 spawning a worker process.

Topology, per Layer 3 of the contract::

    partition "engine" -- output "commands", input  "results"
    partition "worker" -- input  "commands", output "results"

Both edges are opened by exactly one producer and one consumer, which is what
``Platform.start()``'s claim agreement requires; opening a different set on one
rank is a segmentation fault the binding turns into an exception.

Rank -> partition assignment follows ``readAndParseConfiguration``: partitions
take instances in declaration order, so the first partition ("engine") lands on
the first instance, which is the root. ``roundTrip.py`` makes the same
assumption and is the reference for this whole file.

Thread affinity, which the contract makes a hard rule ("``is_full()`` and
``push()`` on one edge must run on the same thread"), holds as follows:

* ``commands`` producer -- the engine calls ``put()`` synchronously on the
  asyncio event-loop thread (``async_engine.py`` :509, :709), and the platform
  transport keeps ``_shutdown_worker`` on that same thread rather than
  off-loading it to ``asyncio.to_thread`` as the queue transport does.
* ``commands`` consumer -- the worker rank's busy loop, one thread.
* ``results`` producer -- the worker's output lane: the busy-loop thread on the
  serial path, the single ``pypto-output`` thread on the pipelined one. Never
  both.
* ``results`` consumer -- the engine reads inside ``asyncio.to_thread``, so the
  *thread identity* varies across calls. What must not vary, and does not, is
  that only one read is ever in flight: the engine awaits each ``get`` before
  issuing the next, and the profile lane -- the one path that could have run a
  second concurrent read -- is refused outright on this transport.

Profiling is not available on this transport. The profile acknowledgement path
needs a third worker->engine channel and Layer 3 fixes the topology at two
edges, so ``ReplicaEngineCore._set_profile_active`` refuses loudly rather than
multiplexing profile acks onto ``results`` and putting the engine's
one-result-per-command invariant at risk.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import sys
import tempfile
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pypto_serving.serving.transport.channel_queue import (
    BlockingPlatformOutputQueue,
    PlatformInputQueue,
    PlatformOutputQueue,
)
from pypto_serving.serving.transport.handshake import PlatformStartupHandshake
from pypto_serving.serving.transport.selection import require_platform_extension_available

if TYPE_CHECKING:
    from pypto_serving.serving.engine.async_engine import EngineConfig

logger = logging.getLogger(__name__)

ENGINE_PARTITION = "engine"
WORKER_PARTITION = "worker"
COMMANDS_EDGE = "commands"
RESULTS_EDGE = "results"

POLICY_ENV_VAR = "PYPTO_SERVING_PLATFORM_POLICY"
MAX_MESSAGE_BYTES_ENV_VAR = "PYPTO_SERVING_PLATFORM_MAX_MESSAGE_BYTES"
CAPACITY_ENV_VAR = "PYPTO_SERVING_PLATFORM_CHANNEL_CAPACITY"
READY_TIMEOUT_ENV_VAR = "PYPTO_SERVING_PLATFORM_READY_TIMEOUT"
PUT_TIMEOUT_ENV_VAR = "PYPTO_SERVING_PLATFORM_PUT_TIMEOUT"

# Sized from measurement, not from the "~1 KB steady state" figure in ipc.py's
# docstring -- that figure counts only the per-request deltas and omits the two
# block tables, which dominate. `_build_step_command` ships `block_ids` AND
# `block_ids_by_group` for every scheduled request on every step, and DeepSeek V4
# declares six cache groups ('ori', 'cmp', 'idx', 'hca_state', 'csa_state',
# 'csa_inner_state'), so the tables are sent seven times over. Measured msgpack
# sizes of a real StepCommand (page_size 16):
#
#   32 reqs x 8192 blocks (128K ctx), single generic pool ...  0.74 MiB
#   32 reqs x 2048 blocks ( 32K ctx), DeepSeek V4 grouped ...  1.24 MiB
#   32 reqs x 8192 blocks (128K ctx), DeepSeek V4 grouped ...  5.17 MiB
#   ... plus 32 NewRequestData with 128K prompt tokens each .. 21.16 MiB
#
# 16 MiB covers the steady-state worst case (5.17 MiB) with room for the one or
# two long-prompt admissions a 4096-token scheduling budget can actually let in
# per step. The last line is not covered and is not meant to be: admitting 32
# fresh 128K-token prompts in a single step is pathological, and
# `_try_dispatch_step` fails those requests rather than the replica.
DEFAULT_MAX_MESSAGE_BYTES = 16 * 1024 * 1024

# Message slots per edge. MUST be >= ReplicaEngineCore._max_in_flight (2 under
# async scheduling), and that is now enforced rather than merely asserted in a
# comment: with capacity 1 the engine can fill the ring and stop draining while
# the worker is inside a blocking push, which was a reproduced hang. Both edges
# get the same capacity -- the engine dispatches at most _max_in_flight commands
# before applying a result and the worker emits exactly one result per command,
# so neither direction can outrun the other by more than that. The third slot is
# slack for a momentarily slow consumer.
DEFAULT_CHANNEL_CAPACITY = 3
MIN_CHANNEL_CAPACITY = 2

# Covers the channel handshake only (memory-slot exchange + fence), not model
# loading -- the worker's model init is behind the startup handshake and uses
# PYPTO_WORKER_INIT_TIMEOUT as it does on the queue transport.
DEFAULT_READY_TIMEOUT_S = 120.0

# How long the worker rank's blocking push waits for the engine to drain a
# result before declaring the engine gone. Generous, because a single decode step
# on a large model can be slow, but never unbounded: see
# BlockingPlatformOutputQueue.put.
DEFAULT_PUT_TIMEOUT_S = 300.0

# TaskR compute resources for the platform's own service workers, matching
# roundTrip.py. Not a serving-side knob: the channel controller's reconcile()
# runs on these, nothing model-related does.
_COMPUTE_RESOURCE_COUNT = 2


@dataclass(frozen=True)
class ChannelSettings:
    """Sizing for the two edges, resolved once per process from the environment."""

    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES
    capacity: int = DEFAULT_CHANNEL_CAPACITY
    ready_timeout_s: float = DEFAULT_READY_TIMEOUT_S
    put_timeout_s: float = DEFAULT_PUT_TIMEOUT_S

    @staticmethod
    def from_env(env: dict[str, str] | None = None) -> "ChannelSettings":
        """Read the sizing knobs, rejecting values that cannot work."""
        source = os.environ if env is None else env
        settings = ChannelSettings(
            max_message_bytes=int(source.get(MAX_MESSAGE_BYTES_ENV_VAR, DEFAULT_MAX_MESSAGE_BYTES)),
            capacity=int(source.get(CAPACITY_ENV_VAR, DEFAULT_CHANNEL_CAPACITY)),
            ready_timeout_s=float(source.get(READY_TIMEOUT_ENV_VAR, DEFAULT_READY_TIMEOUT_S)),
            put_timeout_s=float(source.get(PUT_TIMEOUT_ENV_VAR, DEFAULT_PUT_TIMEOUT_S)),
        )
        if settings.max_message_bytes <= 0:
            raise ValueError(f"{MAX_MESSAGE_BYTES_ENV_VAR} must be positive")
        if settings.capacity < MIN_CHANNEL_CAPACITY:
            # Not a taste question. At capacity 1 the engine can hold the single
            # slot while the worker is inside a blocking push on the other edge;
            # the engine then stops draining (shutdown, or a dead loop) and the
            # worker cannot be reached, because the only thing that would free it
            # is the engine reading. Reproduced: SIGKILL after 90s.
            raise ValueError(
                f"{CAPACITY_ENV_VAR} must be at least {MIN_CHANNEL_CAPACITY} "
                "(ReplicaEngineCore dispatches up to _max_in_flight=2 steps before "
                f"applying a result); got {settings.capacity}"
            )
        if settings.ready_timeout_s <= 0:
            raise ValueError(f"{READY_TIMEOUT_ENV_VAR} must be positive")
        if settings.put_timeout_s <= 0:
            raise ValueError(f"{PUT_TIMEOUT_ENV_VAR} must be positive")
        return settings

    @property
    def buffer_size(self) -> int:
        """Payload ring size per edge.

        Deliberately ``capacity * max_message_bytes`` rather than a free
        parameter. It is what makes ``PlatformOutputQueue.full_for_worst_case()``
        exact: a producer that finds room for one worst-case message can always
        push whatever it then builds, so the engine can commit to scheduling a
        step before it knows the step's size.
        """
        return self.capacity * self.max_message_bytes


def build_replica_policy(settings: ChannelSettings) -> dict[str, Any]:
    """Build the deployment policy for one engine/worker replica pair.

    Generated rather than shipped as a file so the edge sizing can follow
    ``ChannelSettings``. Every rank builds it from the same code with the same
    environment, so every rank walks the same edge list in the same order --
    which is what ``Platform.start()``'s claim agreement assumes.
    """
    edge = {"Buffer Capacity": settings.capacity, "Buffer Size": settings.buffer_size}
    return {
        "Name": "PyPTO Serving Replica",
        "Settings": {
            "Heartbeat": {"Enabled": True, "Visible": False, "Interval": 500, "Tolerance": 1000},
            "Control Buffer": {"Capacity": 32, "Size": 16384},
        },
        "Request Manager": {"Input": "", "Output": ""},
        "Partitions": [
            {
                "Name": ENGINE_PARTITION,
                "Tasks": [
                    {
                        "Function Name": "servingEngine",
                        "Inputs": [RESULTS_EDGE],
                        "Outputs": [COMMANDS_EDGE],
                        "Dependencies": [],
                    }
                ],
            },
            {
                "Name": WORKER_PARTITION,
                "Tasks": [
                    {
                        "Function Name": "servingWorker",
                        "Inputs": [COMMANDS_EDGE],
                        "Outputs": [RESULTS_EDGE],
                        "Dependencies": [],
                    }
                ],
            },
        ],
        "Edges": [
            {"Name": COMMANDS_EDGE, **edge},
            {"Name": RESULTS_EDGE, **edge},
        ],
    }


@contextlib.contextmanager
def _policy_file(policy_path: str | None, settings: ChannelSettings) -> Iterator[str]:
    """Yield a path to the deployment policy, generating one if none was given."""
    override = policy_path or os.environ.get(POLICY_ENV_VAR)
    if override:
        if not os.path.isfile(override):
            raise FileNotFoundError(f"Deployment policy {override!r} does not exist")
        yield override
        return
    with tempfile.TemporaryDirectory(prefix="pypto-serving-platform-") as directory:
        path = os.path.join(directory, "replica.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(build_replica_policy(settings), handle, indent=2)
        yield path


class EngineCommandQueue(PlatformOutputQueue):
    """The engine's ``commands`` end, plus a record that shutdown was sent.

    The launcher owns the worker rank's lifetime and therefore has to be able to
    tell whether the engine actually told it to stop. It cannot infer that: the
    worker sits in ``get(timeout=None)`` and looks identical whether it is idle
    or abandoned. ``ReplicaEngineCore._put_shutdown_command`` calls
    ``note_shutdown_sent()`` (duck-typed, so the ``mp.Queue`` path is untouched)
    and ``_run_role`` refuses to return normally without it -- otherwise any path
    where ``run_serve`` RETURNS instead of raising, such as a uvicorn startup-event
    failure, leaves rank 1 blocked forever. Reproduced: SIGKILL after 90s.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.shutdown_sent = False

    def note_shutdown_sent(self) -> None:
        """Record that a ShutdownCommand reached the channel."""
        self.shutdown_sent = True


@dataclass(frozen=True)
class EngineEndpoints:
    """What the engine rank obtains from the platform, in queue shape."""

    commands: EngineCommandQueue
    results: PlatformInputQueue
    handshake: PlatformStartupHandshake


@dataclass(frozen=True)
class WorkerEndpoints:
    """What the worker rank obtains from the platform, in queue shape."""

    commands: PlatformInputQueue
    results: BlockingPlatformOutputQueue


# One process is one MPI rank is one platform, so "the endpoints this process
# obtained" is genuinely process-global state rather than a shortcut around
# threading them through AsyncLLMEngine. Published here for the whole lifetime of
# the engine rank's serving loop and consumed by ReplicaEngineCore.start().
_ACTIVE_ENGINE_ENDPOINTS: EngineEndpoints | None = None

# Set once the endpoints have been handed to a ReplicaEngineCore. There is one
# channel pair per process, so a second core acquiring them would silently share
# one SPSC edge with the first -- two producers on 'commands', two consumers on
# 'results' -- which the contract forbids and nothing detects at runtime.
_ENDPOINTS_ACQUIRED: bool = False


def current_engine_endpoints() -> EngineEndpoints:
    """Return this process's engine-side channels, or raise if there are none."""
    if _ACTIVE_ENGINE_ENDPOINTS is None:
        raise RuntimeError(
            "PYPTO_SERVING_TRANSPORT=platform, but this process has no platform "
            "channels: the replica was not launched through "
            "pypto_serving.serving.transport.platform_launch.run_platform_replica. "
            "The platform transport only works under 'mpirun -np 2 python -m "
            "pypto_serving.cli ...'; see platform/docs/python-channel-contract.md."
        )
    return _ACTIVE_ENGINE_ENDPOINTS


def acquire_engine_worker_endpoints():
    """Return ``spawn_worker``'s six-tuple, backed by platform channels.

    Same shape as ``spawn_worker`` so ``ReplicaEngineCore.start()`` keeps the
    unpacking it already has:

    * no worker *process* -- the worker is rank 1, which the launcher started;
    * no profile queue -- Layer 3 has two edges and profiling would need a third;
    * ``ready_event`` and ``num_pages_value`` are the same
      ``PlatformStartupHandshake`` object, which satisfies both shapes
      (``.wait(timeout=...)`` and ``.value``).
    """
    global _ENDPOINTS_ACQUIRED

    endpoints = current_engine_endpoints()
    if _ENDPOINTS_ACQUIRED:
        raise RuntimeError(
            "The platform channels for this process have already been handed to a "
            "ReplicaEngineCore. There is one channel pair per MPI rank, and both "
            "edges are SPSC, so a second core would put two producers on 'commands' "
            "and two consumers on 'results'. Run one replica per rank."
        )
    _ENDPOINTS_ACQUIRED = True
    return (
        None,
        endpoints.commands,
        endpoints.results,
        None,
        endpoints.handshake,
        endpoints.handshake,
    )


def _check_single_replica(config: "EngineConfig") -> None:
    """Reject a configuration this topology cannot serve, before MPI starts."""
    parallel = config.parallel_config
    if parallel is not None and parallel.num_replicas != 1:
        raise ValueError(
            f"The platform transport serves exactly one replica, but the requested "
            f"parallel configuration has {parallel.num_replicas}. Multiple replicas need "
            "one engine/worker partition pair each, which Layer 3 of the channel contract "
            "explicitly defers (see 'What is deliberately not in scope')."
        )


def _default_worker_main(config: "EngineConfig") -> Callable[[WorkerEndpoints], None]:
    """The real serving worker loop, bound to this rank's EngineConfig."""

    def worker_main(endpoints: WorkerEndpoints) -> None:
        from pypto_serving.serving.server.serving_worker import run_worker_over_channels

        run_worker_over_channels(config, endpoints.commands, endpoints.results)

    return worker_main


def run_platform_replica(
    config: "EngineConfig",
    *,
    engine_main: Callable[[], Any],
    worker_main: Callable[[WorkerEndpoints], None] | None = None,
    policy_path: str | None = None,
) -> int:
    """Bring the platform up on this rank and run its half of the replica.

    Returns the engine's exit code once both halves have shut down -- non-zero
    if the engine loop died, so a supervisor sees a crashed replica rather than
    a clean exit. Never falls back to the queue transport: every failure either
    raises before MPI is touched or takes the whole job down with
    ``Runtime.abort``.
    """
    require_platform_extension_available()
    _check_single_replica(config)
    settings = ChannelSettings.from_env()
    if worker_main is None:
        worker_main = _default_worker_main(config)

    from pypto_serving.platform import Deployment, Platform, Runtime

    with _policy_file(policy_path, settings) as resolved_policy_path:
        with Runtime(compute_resource_count=_COMPUTE_RESOURCE_COUNT) as runtime:
            if runtime.instance_count != 2:
                print(
                    "Error: PYPTO_SERVING_TRANSPORT=platform needs exactly 2 instances "
                    f"(one engine rank, one worker rank), got {runtime.instance_count}. "
                    "Launch with 'mpirun -np 2 python -m pypto_serving.cli ...'.",
                    file=sys.stderr,
                    flush=True,
                )
                runtime.abort(1)
            try:
                return _run_role(
                    runtime,
                    Deployment,
                    Platform,
                    resolved_policy_path,
                    settings,
                    engine_main=engine_main,
                    worker_main=worker_main,
                )
            except BaseException:
                # A rank that fails alone is not survivable: its peer is blocked in
                # Engine::await() waiting for a STOP RPC, in the claim agreement's
                # allreduce, or in the collective MPI_Finalize. Note that this handler
                # runs BEFORE any Platform.stop(): stopping here is exactly the wrong
                # move, because a failing non-root rank's stop() waits for a terminate
                # the healthy root will never send. abort() does not return.
                traceback.print_exc()
                sys.stderr.flush()
                runtime.abort(1)
                return 1  # unreachable; abort() does not return


def _run_role(
    runtime,
    deployment_cls,
    platform_cls,
    policy_path: str,
    settings: ChannelSettings,
    *,
    engine_main: Callable[[], Any],
    worker_main: Callable[[WorkerEndpoints], None],
) -> int:
    """Open this rank's half of the topology and run it. Raises on any failure."""
    global _ACTIVE_ENGINE_ENDPOINTS

    deployment = deployment_cls.from_json_file(policy_path)
    deployment.assign_edge_managers(runtime)
    deployment.assign_instances(runtime)

    is_engine = runtime.is_root
    role = ENGINE_PARTITION if is_engine else WORKER_PARTITION
    logger.info(
        "Platform transport: instance %s is the %r rank (edges: %s)",
        runtime.instance_id,
        role,
        deployment.edge_names(),
    )

    # NOT `with platform_cls(...)`: __exit__ calls stop() unconditionally, and on a
    # rank that is failing alone stop() blocks in Engine::await() forever instead of
    # letting the caller abort. stop() is reached only on the success path below;
    # every failure path goes through run_platform_replica's abort.
    platform = platform_cls(runtime, deployment)

    if is_engine:
        commands = platform.open_output(COMMANDS_EDGE)
        results = platform.open_input(RESULTS_EDGE)
    else:
        commands = platform.open_input(COMMANDS_EDGE)
        results = platform.open_output(RESULTS_EDGE)

    platform.start()
    platform.wait_until_ready(timeout_s=settings.ready_timeout_s)
    logger.info("Platform transport: channels ready on the %r rank", role)

    exit_code = 0
    if is_engine:
        results_queue = PlatformInputQueue(results, edge_name=RESULTS_EDGE)
        endpoints = EngineEndpoints(
            commands=EngineCommandQueue(
                commands,
                edge_name=COMMANDS_EDGE,
                max_message_size=settings.max_message_bytes,
            ),
            results=results_queue,
            handshake=PlatformStartupHandshake(results_queue, edge_name=RESULTS_EDGE),
        )
        _ACTIVE_ENGINE_ENDPOINTS = endpoints
        try:
            exit_code = int(engine_main() or 0)
        finally:
            _ACTIVE_ENGINE_ENDPOINTS = None
        if not endpoints.commands.shutdown_sent:
            # Raising reaches run_platform_replica's Runtime.abort(1), which is the
            # only thing that can free a peer already blocked in a collective. A
            # clean return here would instead walk into Platform.stop() and then
            # MPI_Finalize while rank 1 is still in get(timeout=None), and the job
            # would hang rather than fail.
            raise RuntimeError(
                "The engine returned without sending the worker rank a "
                "ShutdownCommand, so rank 1 is still blocked reading 'commands'. "
                "That happens when run_serve returns without its lifespan shutdown "
                "running -- a uvicorn startup-event failure is the usual cause. "
                "Failing the job instead of hanging it."
            )
    else:
        # The worker rank is shut down by the engine's ShutdownCommand, exactly as
        # the worker process is on the queue transport, so it ignores SIGINT for the
        # same reason _worker_entry does: Ctrl-C must let rank 0 unwind uvicorn and
        # send that command rather than killing the consumer mid-handshake.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        worker_main(
            WorkerEndpoints(
                commands=PlatformInputQueue(commands, edge_name=COMMANDS_EDGE),
                results=BlockingPlatformOutputQueue(
                    results,
                    edge_name=RESULTS_EDGE,
                    max_message_size=settings.max_message_bytes,
                    put_timeout=settings.put_timeout_s,
                ),
            )
        )

    logger.info(
        "Platform transport: %r rank finished (exit code %d), stopping the platform",
        role,
        exit_code,
    )
    platform.stop()
    return exit_code
