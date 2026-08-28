#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""End-to-end proof for device-payload channels, driven from the serving side.

This script plays the serving layer. It decides what it needs -- two named buffers, their
dtypes and byte counts, the window size, and which chip workers take part -- hands that to
:func:`pypto_serving.platform.device_channels.open_device_channels`, gets a handle back, and
feeds the per-rank contexts to a kernel. The platform module chooses none of it.

Two things are proved:

**Transport.** Every rank publishes its own input into the window and reads its neighbour's
back out. Each rank's output is therefore a *different* rank's input, which only holds if the
bytes actually crossed the device channel.

**Teardown.** Device allocations that leak hold NPU memory forever, so release is checked twice
over. Once structurally, after the owning run's fence: the handle reports ``freed`` and the
worker reports no live domains. Once by exhaustion: allocate and release a large window enough
times that the running total dwarfs the card, which simply cannot complete unless each release
really returns the memory.

Run (2 chips):
    python tests/platform/device_channel_ring/main.py -p a2a3    -d 0-1
    python tests/platform/device_channel_ring/main.py -p a2a3sim -d 0-1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch  # noqa: E402
from simpler.task_interface import (  # noqa: E402
    ArgDirection,
    CallConfig,
    ChipCallable,
    CoreCallable,
    DataType,
    TaskArgs,
    Tensor,
    TensorArgType,
)
from simpler.worker import Worker  # noqa: E402

from simpler_setup.elf_parser import extract_text_section  # noqa: E402
from simpler_setup.kernel_compiler import KernelCompiler  # noqa: E402
from simpler_setup.pto_isa import ensure_pto_isa_root  # noqa: E402
from simpler_setup.torch_interop import make_tensor_arg  # noqa: E402

REPOSITORY_ROOT = str(Path(__file__).resolve().parents[3])
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

from pypto_serving.platform.device_channels import (  # noqa: E402
    DeviceBufferSpec,
    DeviceChannelRequest,
    open_device_channels,
)

HERE = os.path.dirname(os.path.abspath(__file__))

# Must match RING_COUNT in kernels/aiv/ring_exchange.cpp -- tile extents are compile-time, so
# the payload element count is the kernel's property. Everything else about the channel below
# is this layer's choice.
RING_COUNT = 256
FLOAT32_NBYTES = 4
INT32_NBYTES = 4
# One signal slot per rank; kMaxSupportedRanks in the kernel.
SIGNAL_SLOTS = 16

# The exhaustion check: cumulative bytes far beyond any single card, at a peak of one window.
LEAK_PROBE_WINDOW_BYTES = 8 * 1024**3
LEAK_PROBE_ITERATIONS = 20


def serving_channel_request(name: str, workers: list[int]) -> DeviceChannelRequest:
    """Every sizing decision for the channel, made here, by the serving layer.

    The platform module receives this fully specified and allocates exactly it. Note in
    particular that ``window_size`` is computed here: the platform has no rule for it.
    """
    payload_nbytes = RING_COUNT * FLOAT32_NBYTES
    signal_nbytes = SIGNAL_SLOTS * INT32_NBYTES
    return DeviceChannelRequest(
        name=name,
        workers=tuple(workers),
        # Round up to a page so the window is comfortably larger than the two buffers; the
        # runtime rejects the request outright if the buffers do not fit.
        window_size=max(payload_nbytes + signal_nbytes, 4096),
        buffers=(
            DeviceBufferSpec(name="payload", dtype="float32", count=RING_COUNT, nbytes=payload_nbytes),
            DeviceBufferSpec(name="signal", dtype="int32", count=SIGNAL_SLOTS, nbytes=signal_nbytes),
        ),
    )


def parse_device_ids(spec: str) -> list[int]:
    """Accept either a range ('0-1') or an explicit list ('0,1'), which is what the broker gives."""
    if "-" in spec:
        low, high = (int(part) for part in spec.split("-"))
        ids = list(range(low, high + 1))
    else:
        ids = [int(part) for part in spec.split(",") if part != ""]
    if len(ids) < 2:
        raise ValueError(f"the ring exchange needs at least 2 devices, got {ids}")
    return ids


def build_ring_callable(platform: str) -> ChipCallable:
    """Compile the transport-proof kernel and its orchestration."""
    kc = KernelCompiler(platform=platform)
    runtime = "tensormap_and_ringbuffer"
    pto_isa_root = ensure_pto_isa_root()
    include_dirs = kc.get_orchestration_include_dirs(runtime)
    kernel_include_dirs = [*include_dirs, str(kc.project_root / "src" / "common")]
    kernel_bytes = kc.compile_incore(
        source_path=os.path.join(HERE, "kernels/aiv/ring_exchange.cpp"),
        core_type="aiv",
        pto_isa_root=pto_isa_root,
        extra_include_dirs=kernel_include_dirs,
    )
    if not platform.endswith("sim"):
        kernel_bytes = extract_text_section(kernel_bytes)
    orch_bytes = kc.compile_orchestration(
        runtime_name=runtime,
        source_path=os.path.join(HERE, "kernels/orchestration/ring_exchange_orch.cpp"),
    )
    signature = [ArgDirection.IN, ArgDirection.OUT, ArgDirection.INOUT, ArgDirection.INOUT]
    core_callable = CoreCallable.build(signature=signature, binary=kernel_bytes)
    return ChipCallable.build(
        signature=signature,
        func_name="ring_exchange_orchestration",
        binary=orch_bytes,
        children=[(0, core_callable)],
    )


def _window_tensor(address: int, count: int, dtype: DataType) -> Tensor:
    """Wrap a device address from the channel as a kernel tensor argument."""
    return Tensor.make(data=address, shapes=(count,), dtype=dtype, child_memory=True)


def allocate_host_buffers(workers: list[int]) -> tuple[dict[int, "torch.Tensor"], dict[int, "torch.Tensor"]]:
    """Allocate the per-rank host tensors. **Must run before ``Worker.init()``.**

    ``init()`` forks one chip child per device, and a child reaches a host tensor through the
    parent virtual address recorded in its ``TaskArgs``. That address only resolves in the
    child if the mapping was inherited across the fork, so a ``share_memory_()`` tensor created
    *after* ``init()`` is a parent-only address the child will dereference and die on -- a
    SIGSEGV in the chip process, reported here only as "child process exited before mailbox
    completion", with no indication of the cause. Allocating before the fork is the fix.
    """
    host_inputs = {
        worker_idx: torch.tensor(
            [worker_idx * 1000 + i for i in range(RING_COUNT)],
            dtype=torch.float32,
        ).share_memory_()
        for worker_idx in workers
    }
    outputs = {
        worker_idx: torch.zeros(RING_COUNT, dtype=torch.float32).share_memory_() for worker_idx in workers
    }
    return host_inputs, outputs


def run_transport_proof(worker: Worker, ring_handle, workers: list[int], host_inputs, outputs) -> bool:
    """Move a distinct payload from every rank to its neighbour, and check it arrived."""

    # Captured so teardown can be asserted after the run fence, which is the only point at
    # which the backend free has actually happened.
    captured: dict[str, object] = {}

    def orch_fn(orch, _args, cfg):
        # THE SEAM: serving states its requirement, platform materialises it.
        request = serving_channel_request("ring", workers)
        with open_device_channels(orch, request) as channels:
            captured["channels"] = channels
            print(
                f"[device-channel] allocated {channels!r} allocation_id={channels.allocation_id} "
                f"window_size={request.window_size}"
            )
            args_list = []
            for worker_idx in workers:
                payload_ptr = channels.buffer_ptr(worker_idx, "payload")
                signal_ptr = channels.buffer_ptr(worker_idx, "signal")
                print(
                    f"[device-channel] worker {worker_idx}: rank={channels.domain_rank(worker_idx)}/"
                    f"{channels.size()} device_ctx=0x{channels.device_ctx(worker_idx):x} "
                    f"payload=0x{payload_ptr:x} signal=0x{signal_ptr:x}"
                )
                args = TaskArgs()
                args.add_tensor(make_tensor_arg(host_inputs[worker_idx]), TensorArgType.INPUT)
                args.add_tensor(make_tensor_arg(outputs[worker_idx]), TensorArgType.OUTPUT_EXISTING)
                args.add_tensor(_window_tensor(payload_ptr, RING_COUNT, DataType.FLOAT32), TensorArgType.INOUT)
                args.add_tensor(_window_tensor(signal_ptr, SIGNAL_SLOTS, DataType.INT32), TensorArgType.INOUT)
                args.add_scalar(channels.size())
                args.add_scalar(channels.device_ctx(worker_idx))
                args_list.append(args)
            # One group: the kernel's Phase 2 barrier makes every rank wait for its peers, so a
            # rank dispatched without them could never finish.
            orch.submit_next_level_group(ring_handle, args_list, cfg, workers=workers)

    worker.run(orch_fn, args=None, config=CallConfig())

    ok = True
    rank_count = len(workers)
    for worker_idx in workers:
        rank = workers.index(worker_idx)
        source_worker = workers[(rank + rank_count - 1) % rank_count]
        expected = host_inputs[source_worker]
        got = outputs[worker_idx]
        max_diff = float(torch.max(torch.abs(got - expected)))
        print(
            f"[device-channel] worker {worker_idx} received worker {source_worker}'s payload: "
            f"first={got[0].item():.1f} last={got[-1].item():.1f} "
            f"expected first={expected[0].item():.1f} last={expected[-1].item():.1f} max_diff={max_diff:.3e}"
        )
        if max_diff != 0.0:
            print(f"[device-channel] worker {worker_idx} payload MISMATCH")
            ok = False
        if torch.equal(got, host_inputs[worker_idx]):
            # Would mean the kernel read its own window: no transport happened.
            print(f"[device-channel] worker {worker_idx} received its OWN payload; nothing crossed")
            ok = False

    channels = captured["channels"]
    print(f"[device-channel] after the run fence: {channels!r}")
    if not channels.released:
        print("[device-channel] handle is not released")
        ok = False
    if not channels.freed:
        print("[device-channel] handle is NOT freed after the run fence -- device memory leaked")
        ok = False
    live = worker.live_domains
    if live:
        print(f"[device-channel] worker still reports live domains: {live}")
        ok = False
    else:
        print("[device-channel] worker reports no live domains")
    return ok


def run_release_exhaustion_probe(worker: Worker, workers: list[int]) -> bool:
    """Allocate and release far more device memory in total than a card holds.

    Peak usage is one window; cumulative is ``LEAK_PROBE_ITERATIONS`` windows. If release did
    not return the memory this cannot get past the first few iterations, so completing it is
    the evidence -- and it does not depend on ``committed_device_memory``, which by its own
    documentation excludes HCCL/VMM communication windows and so never counts these at all.
    """
    total = LEAK_PROBE_WINDOW_BYTES * LEAK_PROBE_ITERATIONS
    print(
        f"[device-channel] release probe: {LEAK_PROBE_ITERATIONS} x "
        f"{LEAK_PROBE_WINDOW_BYTES / 1024**3:.1f} GiB = {total / 1024**3:.1f} GiB cumulative, "
        f"peak {LEAK_PROBE_WINDOW_BYTES / 1024**3:.1f} GiB"
    )
    seen: list[object] = []

    def probe_orch_fn(iteration: int):
        def _orch_fn(orch, _args, _cfg):
            request = DeviceChannelRequest(
                name=f"probe-{iteration}",
                workers=tuple(workers),
                window_size=LEAK_PROBE_WINDOW_BYTES,
                buffers=(
                    DeviceBufferSpec(
                        name="bulk",
                        dtype="float32",
                        count=LEAK_PROBE_WINDOW_BYTES // FLOAT32_NBYTES,
                        nbytes=LEAK_PROBE_WINDOW_BYTES,
                    ),
                ),
            )
            channels = open_device_channels(orch, request)
            seen.append(channels)
            # No task submitted: this measures allocation and release, nothing else.
            channels.release()

        return _orch_fn

    for iteration in range(LEAK_PROBE_ITERATIONS):
        worker.run(probe_orch_fn(iteration), args=None, config=CallConfig())

    ok = True
    not_freed = [channels for channels in seen if not channels.freed]
    if not_freed:
        print(f"[device-channel] {len(not_freed)} probe window(s) never reached freed: {not_freed}")
        ok = False
    else:
        print(f"[device-channel] all {len(seen)} probe windows reached freed")
    live = worker.live_domains
    if live:
        print(f"[device-channel] worker still reports live domains after the probe: {live}")
        ok = False
    else:
        print("[device-channel] worker reports no live domains after the probe")
    return ok


def run(platform: str, device_ids: list[int], skip_probe: bool) -> int:
    print(f"[device-channel] platform={platform} devices={device_ids}")
    workers = list(range(len(device_ids)))

    worker = Worker(
        level=3,
        platform=platform,
        runtime="tensormap_and_ringbuffer",
        device_ids=device_ids,
        num_sub_workers=0,
    )
    print("[device-channel] compiling the ring-exchange kernel...")
    ring_handle = worker.register(build_ring_callable(platform))
    # Before init(), which forks the chip children -- see allocate_host_buffers.
    host_inputs, outputs = allocate_host_buffers(workers)

    try:
        print("[device-channel] init worker...")
        worker.init()
        committed_before = [worker.committed_device_memory(worker_idx) for worker_idx in workers]
        print(f"[device-channel] committed_device_memory before: {committed_before}")

        ok = run_transport_proof(worker, ring_handle, workers, host_inputs, outputs)

        if not skip_probe:
            ok = run_release_exhaustion_probe(worker, workers) and ok

        committed_after = [worker.committed_device_memory(worker_idx) for worker_idx in workers]
        print(f"[device-channel] committed_device_memory after:  {committed_after}")

        if not ok:
            print("[device-channel] device channel checks FAILED")
            return 1
        print("[device-channel] device channel checks PASSED")
        return 0
    finally:
        worker.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-p", "--platform", required=True, choices=["a2a3sim", "a2a3"])
    parser.add_argument("-d", "--device", default="0-1", help="Devices as a range ('0-1') or a list ('0,1'). At least two.")
    parser.add_argument(
        "--skip-release-probe",
        action="store_true",
        help="Skip the large-window release probe (it allocates several GiB per rank).",
    )
    cli = parser.parse_args()
    return run(cli.platform, parse_device_ids(cli.device), cli.skip_release_probe)


if __name__ == "__main__":
    sys.exit(main())
