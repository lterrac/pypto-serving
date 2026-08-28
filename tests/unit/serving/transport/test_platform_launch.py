# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The generated deployment policy and the endpoint hand-off.

These run without MPI and without the native extension: they cover the parts of
the SPMD launch that are pure Python. The bring-up itself is covered
end-to-end by ``tests/platform/two_rank_replica.py`` under ``mpirun``.
"""

from __future__ import annotations

import json

import pytest

from pypto_serving.serving.transport import platform_launch
from pypto_serving.serving.transport.platform_launch import (
    COMMANDS_EDGE,
    ENGINE_PARTITION,
    RESULTS_EDGE,
    WORKER_PARTITION,
    ChannelSettings,
    build_replica_policy,
)


def test_topology_matches_layer_three_of_the_contract():
    policy = build_replica_policy(ChannelSettings())

    partitions = {partition["Name"]: partition for partition in policy["Partitions"]}
    assert list(partitions) == [ENGINE_PARTITION, WORKER_PARTITION], (
        "declaration order is the rank assignment: rank 0 must be the engine"
    )

    engine_task = partitions[ENGINE_PARTITION]["Tasks"][0]
    worker_task = partitions[WORKER_PARTITION]["Tasks"][0]
    assert engine_task["Outputs"] == [COMMANDS_EDGE]
    assert engine_task["Inputs"] == [RESULTS_EDGE]
    assert worker_task["Inputs"] == [COMMANDS_EDGE]
    assert worker_task["Outputs"] == [RESULTS_EDGE]


def test_every_edge_has_exactly_one_producer_and_one_consumer():
    """``Platform.start()`` agrees the claim set; an edge claimed by anything
    other than one producer and one consumer used to be a segfault."""
    policy = build_replica_policy(ChannelSettings())
    produced: list[str] = []
    consumed: list[str] = []
    for partition in policy["Partitions"]:
        for task in partition["Tasks"]:
            produced.extend(task["Outputs"])
            consumed.extend(task["Inputs"])

    edge_names = [edge["Name"] for edge in policy["Edges"]]
    assert sorted(produced) == sorted(edge_names)
    assert sorted(consumed) == sorted(edge_names)
    assert len(set(produced)) == len(produced)
    assert len(set(consumed)) == len(consumed)


def test_payload_ring_holds_a_full_capacity_of_worst_case_messages():
    """What makes ``full_for_worst_case()`` exact rather than advisory."""
    settings = ChannelSettings(max_message_bytes=1024, capacity=3)
    policy = build_replica_policy(settings)
    for edge in policy["Edges"]:
        assert edge["Buffer Capacity"] == 3
        assert edge["Buffer Size"] == 3 * 1024


def test_default_capacity_covers_the_engines_pipeline_depth():
    """Depth 2 under async scheduling; a smaller ring would make the healthy
    steady state look like backpressure on every step."""
    assert ChannelSettings().capacity >= 2


def test_both_ranks_generate_byte_identical_policies():
    """The claim agreement assumes every rank walks the same edge list in the
    same order, so the two ranks' generated JSON must not differ."""
    settings = ChannelSettings()
    first = json.dumps(build_replica_policy(settings), indent=2)
    second = json.dumps(build_replica_policy(settings), indent=2)
    assert first == second


@pytest.mark.parametrize(
    "env",
    [
        {platform_launch.MAX_MESSAGE_BYTES_ENV_VAR: "0"},
        {platform_launch.CAPACITY_ENV_VAR: "0"},
        {platform_launch.READY_TIMEOUT_ENV_VAR: "0"},
        {platform_launch.PUT_TIMEOUT_ENV_VAR: "0"},
    ],
)
def test_unusable_channel_settings_are_rejected(env):
    with pytest.raises(ValueError):
        ChannelSettings.from_env(env)


def test_a_capacity_below_the_pipeline_depth_is_rejected():
    """Not taste: at capacity 1 the engine can hold the only slot while the
    worker is inside a blocking push on the other edge, and the engine is the
    only thing that could free it. Reproduced as a SIGKILL-after-90s hang."""
    with pytest.raises(ValueError, match="must be at least"):
        ChannelSettings.from_env({platform_launch.CAPACITY_ENV_VAR: "1"})


def test_the_default_message_limit_covers_the_measured_steady_state():
    """Measured worst case is a 32-request grouped-KV decode step at 128K
    context: 5.17 MiB. The old 4 MiB default did not cover it."""
    assert platform_launch.DEFAULT_MAX_MESSAGE_BYTES >= 6 * 1024 * 1024


def test_endpoints_are_refused_when_the_process_was_not_launched_as_a_rank():
    """Never a silent fallback: without the SPMD launch there are no channels,
    and asking for them has to say so."""
    assert platform_launch._ACTIVE_ENGINE_ENDPOINTS is None
    with pytest.raises(RuntimeError, match="mpirun"):
        platform_launch.current_engine_endpoints()


def test_multiple_replicas_are_rejected_before_mpi_is_touched():
    from pypto_serving.config.parallel import ParallelConfig
    from pypto_serving.serving.engine.async_engine import EngineConfig

    config = EngineConfig(
        parallel_config=ParallelConfig(data_parallel_size=2, devices=(0, 1)),
    )
    assert config.parallel_config.num_replicas == 2
    with pytest.raises(ValueError, match="exactly one replica"):
        platform_launch._check_single_replica(config)


def test_the_endpoints_can_only_be_acquired_once(monkeypatch):
    """One channel pair per rank, both edges SPSC: a second core would put two
    producers on 'commands' and two consumers on 'results'."""
    endpoints = platform_launch.EngineEndpoints(
        commands="commands-queue", results="results-queue", handshake="handshake"
    )
    monkeypatch.setattr(platform_launch, "_ACTIVE_ENGINE_ENDPOINTS", endpoints)
    monkeypatch.setattr(platform_launch, "_ENDPOINTS_ACQUIRED", False)

    platform_launch.acquire_engine_worker_endpoints()
    with pytest.raises(RuntimeError, match="already been handed"):
        platform_launch.acquire_engine_worker_endpoints()


def test_the_command_queue_records_that_shutdown_was_sent():
    """The launcher cannot otherwise tell an idle worker rank from an abandoned
    one: both look like a thread blocked in get(timeout=None)."""
    from tests.unit.serving.transport.fakes import FakeChannel

    channel = FakeChannel(capacity=4)
    commands = platform_launch.EngineCommandQueue(
        channel, edge_name="commands", max_message_size=64
    )
    assert commands.shutdown_sent is False
    commands.note_shutdown_sent()
    assert commands.shutdown_sent is True


def test_engine_endpoints_fill_spawn_workers_six_tuple(monkeypatch):
    """``ReplicaEngineCore.start()`` unpacks six values; the platform path has
    to answer in the same shape so that code stays untouched."""
    endpoints = platform_launch.EngineEndpoints(
        commands="commands-queue",
        results="results-queue",
        handshake="handshake",
    )
    monkeypatch.setattr(platform_launch, "_ACTIVE_ENGINE_ENDPOINTS", endpoints)
    monkeypatch.setattr(platform_launch, "_ENDPOINTS_ACQUIRED", False)

    process, input_q, output_q, profile_q, ready, num_pages = (
        platform_launch.acquire_engine_worker_endpoints()
    )
    assert process is None, "the worker is rank 1, not a child process"
    assert profile_q is None, "Layer 3 has two edges; profiling would need a third"
    assert (input_q, output_q) == ("commands-queue", "results-queue")
    # One object stands in for both ready_event and num_pages_value.
    assert ready is endpoints.handshake
    assert num_pages is endpoints.handshake
