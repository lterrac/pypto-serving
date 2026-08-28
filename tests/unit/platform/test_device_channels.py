# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The serving/platform seam for device channels, exercised without a device.

The hardware proof lives in ``tests/platform/device_channel_ring``. What is checked here is
the property that no amount of hardware testing would show: that the platform side is a
pass-through and decides nothing. A fake orchestrator records the allocation request verbatim,
so a sizing rule or a defaulted field introduced later fails here.
"""

from __future__ import annotations

import dataclasses

import pytest

from pypto_serving.platform.device_channels import (
    DeviceBufferSpec,
    DeviceChannelRequest,
    DeviceChannelSet,
    open_device_channels,
)


class FakeDomainContext:
    """Stand-in for ``simpler.task_interface.ChipDomainContext``."""

    def __init__(self, *, name, domain_rank, domain_size, device_ctx, local_window_base, buffer_ptrs):
        self.name = name
        self.domain_rank = domain_rank
        self.domain_size = domain_size
        self.device_ctx = device_ctx
        self.local_window_base = local_window_base
        self.actual_window_size = 4096
        self.buffer_ptrs = buffer_ptrs


class FakeDomainHandle:
    """Stand-in for ``simpler.task_interface.CommDomainHandle``, with the same two-stage release."""

    def __init__(self, name, workers, contexts, allocation_id):
        self.name = name
        self.workers = tuple(workers)
        self.contexts = contexts
        self.allocation_id = allocation_id
        self.released = False
        self.freed = False
        self.release_calls = 0

    def __getitem__(self, chip_idx):
        if self.released:
            raise RuntimeError("already released")
        return self.contexts[chip_idx]

    def release(self):
        self.release_calls += 1
        self.released = True

    def fence(self):
        """What ``Worker.run`` does after its completion wait."""
        self.freed = True


class FakeOrchestrator:
    """Records what the platform asked for, and hands back a handle."""

    def __init__(self):
        self.calls = []

    def allocate_domain(self, *, name, workers, window_size, buffers):
        self.calls.append(
            {"name": name, "workers": workers, "window_size": window_size, "buffers": buffers}
        )
        contexts = {
            worker: FakeDomainContext(
                name=name,
                domain_rank=rank,
                domain_size=len(workers),
                device_ctx=0xC000 + rank,
                local_window_base=0xD000 + rank * 0x1000,
                buffer_ptrs={
                    spec.name: 0xD000 + rank * 0x1000 + index * 0x100
                    for index, spec in enumerate(buffers)
                },
            )
            for rank, worker in enumerate(workers)
        }
        return FakeDomainHandle(name, workers, contexts, allocation_id=7)


def a_request(**overrides) -> DeviceChannelRequest:
    base = {
        "name": "ring",
        "workers": (0, 1),
        "window_size": 4096,
        "buffers": (
            DeviceBufferSpec(name="payload", dtype="float32", count=256, nbytes=1024),
            DeviceBufferSpec(name="signal", dtype="int32", count=16, nbytes=64),
        ),
    }
    base.update(overrides)
    return DeviceChannelRequest(**base)


# -- the request is the serving layer's, whole ----------------------------------------------


def test_window_size_has_no_default_so_the_platform_cannot_supply_one():
    """A defaulted window size would be a sizing policy living on the platform side."""
    with pytest.raises(TypeError):
        DeviceChannelRequest(name="ring", workers=(0, 1))  # type: ignore[call-arg]


def test_sequence_fields_are_normalised_without_being_reinterpreted():
    request = DeviceChannelRequest(
        name="ring", workers=[0, 1], window_size=4096, buffers=[DeviceBufferSpec("p", "float32", 4, 16)]
    )
    assert request.workers == (0, 1)
    assert isinstance(request.buffers, tuple)
    assert request.window_size == 4096


def test_the_request_is_frozen():
    request = a_request()
    with pytest.raises(dataclasses.FrozenInstanceError):
        request.window_size = 8192  # type: ignore[misc]


# -- the platform forwards it verbatim ------------------------------------------------------


def test_open_forwards_the_request_field_for_field():
    pytest.importorskip("simpler.task_interface")
    orch = FakeOrchestrator()
    request = a_request()

    channels = open_device_channels(orch, request)

    assert len(orch.calls) == 1
    call = orch.calls[0]
    assert call["name"] == "ring"
    assert list(call["workers"]) == [0, 1]
    assert call["window_size"] == 4096
    assert [(b.name, b.dtype, b.count, b.nbytes) for b in call["buffers"]] == [
        ("payload", "float32", 256, 1024),
        ("signal", "int32", 16, 64),
    ]
    assert channels.request is request


def test_a_window_far_larger_than_its_buffers_is_passed_through_unshrunk():
    """The platform must not 'helpfully' right-size what the caller asked for."""
    pytest.importorskip("simpler.task_interface")
    orch = FakeOrchestrator()

    open_device_channels(orch, a_request(window_size=1 << 30))

    assert orch.calls[0]["window_size"] == 1 << 30


def test_host_staging_flags_are_carried_through():
    pytest.importorskip("simpler.task_interface")
    orch = FakeOrchestrator()
    request = a_request(
        buffers=(DeviceBufferSpec("p", "float32", 4, 16, load_from_host=True, store_to_host=True),)
    )

    open_device_channels(orch, request)

    spec = orch.calls[0]["buffers"][0]
    assert (spec.load_from_host, spec.store_to_host) == (True, True)


# -- the getters ----------------------------------------------------------------------------


def _a_channel_set() -> tuple[DeviceChannelSet, FakeDomainHandle]:
    request = a_request()
    orch = FakeOrchestrator()
    handle = orch.allocate_domain(
        name=request.name,
        workers=list(request.workers),
        window_size=request.window_size,
        buffers=list(request.buffers),
    )
    return DeviceChannelSet(request, handle), handle


def test_getters_expose_the_per_rank_context():
    channels, handle = _a_channel_set()
    assert channels.name == "ring"
    assert channels.workers == (0, 1)
    assert channels.size() == 2
    assert channels.allocation_id == 7
    assert channels.domain_rank(1) == 1
    assert channels.device_ctx(1) == 0xC001
    assert channels.window_base(1) == 0xE000
    assert channels.buffer_ptr(1, "signal") == 0xE100
    assert channels.context(0) is handle.contexts[0]


def test_an_unknown_buffer_name_names_the_ones_that_exist():
    channels, _ = _a_channel_set()
    with pytest.raises(KeyError) as excinfo:
        channels.buffer_ptr(0, "kv")
    message = str(excinfo.value)
    assert "kv" in message
    assert "payload" in message and "signal" in message


def test_a_non_member_chip_is_absent():
    channels, _ = _a_channel_set()
    with pytest.raises(KeyError):
        channels.context(9)


# -- teardown -------------------------------------------------------------------------------


def test_release_is_idempotent_and_marks_before_it_frees():
    channels, handle = _a_channel_set()
    assert (channels.released, channels.freed) == (False, False)

    channels.release()
    assert channels.released is True
    # Still alive: the backend free waits for the owning run's fence.
    assert channels.freed is False

    channels.release()
    assert handle.release_calls == 2  # delegated both times; the runtime handle absorbs the repeat

    handle.fence()
    assert channels.freed is True


def test_a_released_set_refuses_further_use():
    channels, _ = _a_channel_set()
    channels.release()
    with pytest.raises(RuntimeError):
        channels.device_ctx(0)


def test_the_context_manager_releases_on_exit():
    channels, _ = _a_channel_set()
    with channels as entered:
        assert entered is channels
        assert channels.released is False
    assert channels.released is True


def test_the_context_manager_releases_when_the_body_raises():
    channels, _ = _a_channel_set()
    with pytest.raises(ValueError), channels:
        raise ValueError("boom")
    assert channels.released is True


def test_repr_reports_the_lifecycle_state():
    channels, handle = _a_channel_set()
    assert "live" in repr(channels)
    channels.release()
    assert "released-pending-free" in repr(channels)
    handle.fence()
    assert "freed" in repr(channels)
