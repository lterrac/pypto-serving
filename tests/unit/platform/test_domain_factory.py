# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The platform-as-allocator seam, exercised without a device.

The hardware proof lives in ``tests/platform/domain_factory_ring``. What is checked here is
the property hardware cannot show: that the factory is a *translation* and nothing else. It
is called with what the pypto compiler emitted, and what reaches ``allocate_domain`` must be
the same thing, field for field, with no sizing, renaming, padding or reordering in between.
The other half is the return value: pypto keys its own bookkeeping on the identity of the
handle it gets back, so returning the convenience wrapper instead would break it silently.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

import pytest

from pypto_serving.platform.device_channels import DeviceChannelSet
from pypto_serving.platform.domain_factory import DomainCreation, PlatformDomainFactory, RankWindow


@dataclass
class SpecFromTheCompiler:
    """What generated orchestration passes as one ``CommBufferSpec``."""

    name: str
    dtype: str
    count: int
    nbytes: int
    load_from_host: bool = False
    store_to_host: bool = False


class FakeDomainContext:
    def __init__(self, *, name, domain_rank, domain_size, device_ctx, local_window_base, buffer_ptrs):
        self.name = name
        self.domain_rank = domain_rank
        self.domain_size = domain_size
        self.device_ctx = device_ctx
        self.local_window_base = local_window_base
        self.actual_window_size = 4096
        self.buffer_ptrs = buffer_ptrs


class FakeDomainHandle:
    def __init__(self, name, workers, contexts):
        self.name = name
        self.workers = tuple(workers)
        self.contexts = contexts
        self.allocation_id = 11
        self.released = False
        self.freed = False

    def __getitem__(self, chip_idx):
        return self.contexts[chip_idx]

    def release(self):
        self.released = True


class FakeOrchestrator:
    """Stands in for the run's orchestrator, recording the call verbatim."""

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
        return FakeDomainHandle(name, workers, contexts)


COMPILER_BUFFERS = (
    SpecFromTheCompiler(name="src_buf", dtype="opaque", count=256, nbytes=256),
    SpecFromTheCompiler(name="dst_buf", dtype="opaque", count=256, nbytes=256),
    SpecFromTheCompiler(name="signal_buf", dtype="opaque", count=4, nbytes=32),
)


def call_factory(factory: PlatformDomainFactory, orch: FakeOrchestrator, **overrides):
    kwargs = {
        "name": "__comm_d0",
        "workers": [0, 1],
        "window_size": 544,
        "buffers": list(COMPILER_BUFFERS),
    }
    kwargs.update(overrides)
    return factory(orch, **kwargs)


# -- the translation is lossless and adds nothing -------------------------------------------


def test_the_compilers_request_reaches_the_runtime_field_for_field():
    pytest.importorskip("simpler.task_interface")
    orch = FakeOrchestrator()

    call_factory(PlatformDomainFactory(), orch)

    assert len(orch.calls) == 1
    call = orch.calls[0]
    assert call["name"] == "__comm_d0"
    assert list(call["workers"]) == [0, 1]
    assert call["window_size"] == 544
    assert [(b.name, b.dtype, b.count, b.nbytes) for b in call["buffers"]] == [
        ("src_buf", "opaque", 256, 256),
        ("dst_buf", "opaque", 256, 256),
        ("signal_buf", "opaque", 4, 32),
    ]


def test_the_window_is_neither_grown_nor_shrunk_to_fit_its_buffers():
    """A window far larger than its buffers is the compiler's business, not ours."""
    pytest.importorskip("simpler.task_interface")
    orch = FakeOrchestrator()

    call_factory(PlatformDomainFactory(), orch, window_size=1 << 20)

    assert orch.calls[0]["window_size"] == 1 << 20


def test_buffer_order_is_preserved_because_it_fixes_the_offsets():
    pytest.importorskip("simpler.task_interface")
    orch = FakeOrchestrator()

    call_factory(PlatformDomainFactory(), orch)

    assert [b.name for b in orch.calls[0]["buffers"]] == ["src_buf", "dst_buf", "signal_buf"]


def test_host_staging_flags_survive_the_translation():
    pytest.importorskip("simpler.task_interface")
    orch = FakeOrchestrator()

    call_factory(
        PlatformDomainFactory(),
        orch,
        buffers=[SpecFromTheCompiler("w", "opaque", 8, 64, load_from_host=True, store_to_host=True)],
    )

    spec = orch.calls[0]["buffers"][0]
    assert (spec.load_from_host, spec.store_to_host) == (True, True)


def test_a_domain_with_no_buffers_is_still_allocated():
    pytest.importorskip("simpler.task_interface")
    orch = FakeOrchestrator()

    call_factory(PlatformDomainFactory(), orch, buffers=[])

    assert list(orch.calls[0]["buffers"]) == []


def test_an_unexpected_keyword_from_the_compiler_is_a_loud_failure():
    """Better a TypeError than a field the platform quietly drops on the floor."""
    factory = PlatformDomainFactory()
    with pytest.raises(TypeError):
        call_factory(factory, FakeOrchestrator(), placement="numa0")


# -- what comes back is the runtime's handle, not our wrapper -------------------------------


def test_the_runtime_handle_is_returned_so_pypto_can_key_on_its_identity():
    pytest.importorskip("simpler.task_interface")
    orch = FakeOrchestrator()
    factory = PlatformDomainFactory()

    handle = call_factory(factory, orch)

    assert isinstance(handle, FakeDomainHandle)
    assert handle.name == "__comm_d0"


# -- the records are the evidence the seam was used -----------------------------------------


def test_each_creation_is_recorded_with_what_it_was_asked_for():
    pytest.importorskip("simpler.task_interface")
    orch = FakeOrchestrator()
    factory = PlatformDomainFactory()

    call_factory(factory, orch)

    (record,) = factory.records
    assert isinstance(record, DomainCreation)
    assert record.name == "__comm_d0"
    assert record.workers == (0, 1)
    assert record.window_size == 544
    assert record.buffers == (
        ("src_buf", "opaque", 256, 256),
        ("dst_buf", "opaque", 256, 256),
        ("signal_buf", "opaque", 4, 32),
    )
    assert record.allocation_id == 11


def test_a_record_holds_no_reference_to_the_runtimes_objects():
    """A frozen serving replica would otherwise pin nanobind instances past shutdown."""
    pytest.importorskip("simpler.task_interface")
    factory = PlatformDomainFactory()

    handle = call_factory(factory, FakeOrchestrator())

    reachable = gc.get_referents(factory.records[0])
    assert handle not in reachable
    assert not any(isinstance(obj, (FakeDomainHandle, DeviceChannelSet)) for obj in reachable)
    # And nothing anywhere in the factory refers to the handle either.
    assert not any(
        handle is obj for obj in gc.get_referents(factory, factory.records, *factory.records)
    )


def test_the_per_rank_addresses_are_snapshotted_so_they_outlive_release():
    """pypto frees every retained window at close; a record read afterwards must still speak."""
    pytest.importorskip("simpler.task_interface")
    factory = PlatformDomainFactory()
    handle = call_factory(factory, FakeOrchestrator())

    handle.release()

    (record,) = factory.records
    assert record.ranks == (
        RankWindow(worker_id=0, domain_rank=0, device_ctx=0xC000, window_base=0xD000),
        RankWindow(worker_id=1, domain_rank=1, device_ctx=0xC001, window_base=0xE000),
    )
    assert "worker 1: rank=1 device_ctx=0xc001 window_base=0xe000" == str(record.ranks[1])


def test_records_accumulate_in_creation_order_across_domains():
    pytest.importorskip("simpler.task_interface")
    orch = FakeOrchestrator()
    factory = PlatformDomainFactory()

    call_factory(factory, orch, name="__comm_d0")
    call_factory(factory, orch, name="__comm_d1")

    assert [record.name for record in factory.records] == ["__comm_d0", "__comm_d1"]


def test_records_is_a_snapshot_the_caller_cannot_use_to_mutate_the_factory():
    pytest.importorskip("simpler.task_interface")
    factory = PlatformDomainFactory()
    call_factory(factory, FakeOrchestrator())

    snapshot = factory.records
    call_factory(factory, FakeOrchestrator(), name="__comm_d1")

    assert len(snapshot) == 1
    assert len(factory.records) == 2


def test_a_creation_renders_the_window_it_describes():
    pytest.importorskip("simpler.task_interface")
    factory = PlatformDomainFactory()
    call_factory(factory, FakeOrchestrator())

    rendered = str(factory.records[0])
    assert "__comm_d0" in rendered
    assert "window_size=544" in rendered
    assert "allocation_id=11" in rendered
    assert "src_buf(256B)" in rendered


def test_the_creation_is_logged_so_a_run_shows_who_allocated(caplog):
    pytest.importorskip("simpler.task_interface")
    factory = PlatformDomainFactory()

    with caplog.at_level(logging.INFO, logger="pypto_serving.platform.domain_factory"):
        call_factory(factory, FakeOrchestrator())

    assert "platform created comm window" in caplog.text
    assert "__comm_d0" in caplog.text
