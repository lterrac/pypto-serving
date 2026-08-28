#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The platform allocating the windows a *compiled model* asks for, proved on two cards.

``tests/platform/device_channel_ring`` proved the earlier half of this: serving can state a
device-channel requirement and the platform materialises it. But there the requirement was
hand-written -- a ``DeviceChannelRequest`` literal in the test -- and the kernel was driven
through a bare ``simpler`` ``Worker``. Nothing a *model* declares went through the seam.

This is the other half. The window here is not the test's idea: it is emitted by the pypto
compiler from a ``@pl.jit.host`` orchestration, exactly the way DeepSeek's MoE windows are.
The program is the ring shuffle from pypto's own ``tests/st/distributed/test_l3_put.py`` --
rank ``r`` pushes its input into rank ``(r + 1) % n``'s window slice -- so the transport
assertion is the same one the device-channel ring test makes:

* each rank's output equals its **neighbour's** input, ``max_diff == 0``;
* no rank receives its own input, which is what would happen if nothing crossed.

Three runs of that identical program, differing only in who allocated the window:

1. ``baseline``  -- ``prepare(persistent=True)``, no factory. The runtime allocates, exactly
   as it does today. This is the control: it must pass unchanged, and it is the evidence that
   omitting ``domain_factory`` changes nothing.
2. ``factory``   -- ``prepare(persistent=True, domain_factory=PlatformDomainFactory())``.
   Same program, same assertions, but every window came from
   ``pypto_serving.platform.device_channels.open_device_channels``. The factory's records are
   printed and checked, because "it still worked" alone is equally consistent with the hook
   never having fired.
3. ``factory-transient`` -- the same with ``persistent=False``, where pypto allocates and
   releases the domain inside the single run rather than retaining it. The hook has to be
   live on both paths or ``domain_factory`` would be a silent no-op in one of them.

Run (2 chips)::

    python tests/platform/domain_factory_ring/main.py -d 0,1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pypto.language as pl  # noqa: E402
import pypto.language.distributed as pld  # noqa: E402
import torch  # noqa: E402
from pypto import ir  # noqa: E402
from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: E402

REPOSITORY_ROOT = str(Path(__file__).resolve().parents[3])
if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

from pypto_serving.platform.domain_factory import PlatformDomainFactory  # noqa: E402

# Payload width. Small on purpose: this measures who allocated the window, not bandwidth.
SIZE = 64
RANKS = 2


def build_ring_put_program():
    """The ring shuffle, as a distributed pypto program.

    Lifted from pypto's ``tests/st/distributed/test_l3_put.py::_build_ring_put_program`` so
    the program under test is one whose correct answer is already established upstream --
    a failure here is then about the allocator, not about a kernel this test invented.

    The window is declared inside ``host_orch`` via ``pld.alloc_window_buffer``; the compiler
    turns those three declarations into one ``allocate_domain`` call in generated Python. That
    call is what ``domain_factory`` intercepts.
    """

    @pl.program
    class RingPut:
        @pl.function(type=pl.FunctionType.InCore)
        def ring_step(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            src: pld.DistributedTensor[[1, SIZE], pl.FP32],
            dst: pld.DistributedTensor[[1, SIZE], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            # Stage our own input into our own window slice.
            local = pl.load(inp, [0, 0], [1, SIZE])
            src = pl.store(local, [0, 0], src)

            # Push it into the peer's slice: this is the traffic that must cross.
            pld.tensor.put(dst, peer=peer, src=src, atomic=pld.AtomicType.None_)

            # Tell the peer our write landed, and wait for the rank that targets us.
            pld.system.notify(
                target=signal,
                peer=peer,
                offsets=[0, 0],
                value=1,
                op=pld.NotifyOp.AtomicAdd,
            )
            pld.system.wait(signal=signal, offsets=[0, 0], expected=1, cmp=pld.WaitCmp.Ge)

            # Read back our own slice -- written by the rank whose peer is us.
            recv = pl.load(dst, [0, 0], [1, SIZE])
            return pl.store(recv, [0, 0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(
            self,
            inp: pl.Tensor[[1, SIZE], pl.FP32],
            out: pl.Out[pl.Tensor[[1, SIZE], pl.FP32]],
            src: pld.DistributedTensor[[1, SIZE], pl.FP32],
            dst: pld.DistributedTensor[[1, SIZE], pl.FP32],
            signal: pld.DistributedTensor[[1, 1], pl.INT32],
            peer: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[1, SIZE], pl.FP32]:
            return self.ring_step(inp, out, src, dst, signal, peer)

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            inputs: pl.Tensor[[RANKS, 1, SIZE], pl.FP32],
            outputs: pl.Out[pl.Tensor[[RANKS, 1, SIZE], pl.FP32]],
        ) -> pl.Tensor[[RANKS, 1, SIZE], pl.FP32]:
            src_buf = pld.alloc_window_buffer(SIZE * pl.FP32.get_byte())
            dst_buf = pld.alloc_window_buffer(SIZE * pl.FP32.get_byte())
            signal_buf = pld.alloc_window_buffer(pl.INT32.get_byte())

            for r in pl.range(pld.world_size()):
                src = pld.window(src_buf, [1, SIZE], dtype=pl.FP32)
                dst = pld.window(dst_buf, [1, SIZE], dtype=pl.FP32)
                signal = pld.window(signal_buf, [1, 1], dtype=pl.INT32)
                self.chip_orch(
                    inputs[r], outputs[r], src, dst, signal, (r + 1) % pld.world_size(), device=r
                )
            return outputs

    return RingPut


def make_host_buffers() -> tuple[torch.Tensor, torch.Tensor]:
    """Shared-memory IO buffers, allocated **before** any ``prepare()``.

    ``prepare()`` forks one chip child per device and the child reaches these through the
    mapping it inherited, so a tensor created afterwards is a parent-only address it would
    dereference and die on. Allocating up front and reusing in place across all three runs is
    both the supported shape and what makes the three runs comparable.
    """
    inputs = torch.stack(
        [
            torch.arange(SIZE, dtype=torch.float32).reshape(1, SIZE),
            torch.arange(100.0, 100.0 + SIZE, dtype=torch.float32).reshape(1, SIZE),
        ]
    ).share_memory_()
    outputs = torch.zeros((RANKS, 1, SIZE), dtype=torch.float32).share_memory_()
    return inputs, outputs


def check_ring(label: str, inputs: torch.Tensor, outputs: torch.Tensor) -> bool:
    """The transport assertion: every rank holds its neighbour's payload, exactly."""
    ok = True
    for rank in range(RANKS):
        source = (rank - 1) % RANKS
        expected = inputs[source]
        got = outputs[rank]
        max_diff = float(torch.max(torch.abs(got - expected)))
        print(
            f"[domain-factory:{label}] rank {rank} received rank {source}'s payload: "
            f"first={got[0, 0].item():.1f} last={got[0, -1].item():.1f} "
            f"expected first={expected[0, 0].item():.1f} last={expected[0, -1].item():.1f} "
            f"max_diff={max_diff:.3e}"
        )
        if max_diff != 0.0:
            print(f"[domain-factory:{label}] rank {rank} payload MISMATCH")
            ok = False
        if torch.equal(got, inputs[rank]):
            print(f"[domain-factory:{label}] rank {rank} received its OWN payload; nothing crossed")
            ok = False
    return ok


def check_records(label: str, factory: PlatformDomainFactory, device_ids: list[int]) -> bool:
    """The provenance assertion: these windows came from the platform, and match the program.

    Without this the run proves only that the program still works, which it would also do if
    ``domain_factory`` were never consulted.
    """
    records = factory.records
    if not records:
        print(f"[domain-factory:{label}] factory was NEVER called -- the hook did not fire")
        return False
    ok = True
    for record in records:
        print(f"[domain-factory:{label}] factory allocated {record}")
        # A record is a value copy: pypto has already released and freed every retained
        # window by the time the worker is closed, and the factory deliberately keeps no
        # reference to the handle, so everything printed here was read at creation time.
        for rank_window in record.ranks:
            print(f"[domain-factory:{label}]   -> {rank_window}")
        if len(record.ranks) != len(device_ids):
            print(f"[domain-factory:{label}] expected {len(device_ids)} ranks, got {len(record.ranks)}")
            ok = False
        if {rank_window.domain_rank for rank_window in record.ranks} != set(range(len(device_ids))):
            print(f"[domain-factory:{label}] ranks are not a dense 0..n-1 set: {record.ranks}")
            ok = False
        if any(rank_window.device_ctx == 0 for rank_window in record.ranks):
            print(f"[domain-factory:{label}] a rank has a null device context: {record.ranks}")
            ok = False
        if record.workers != tuple(range(len(device_ids))):
            print(f"[domain-factory:{label}] unexpected member set {record.workers}")
            ok = False
        # The program declares three window buffers; the compiler carves them out of one
        # window, so the request must carry all three and a window big enough for them.
        if len(record.buffers) != 3:
            print(f"[domain-factory:{label}] expected 3 window buffers, got {len(record.buffers)}")
            ok = False
        declared = sum(nbytes for _name, _dtype, _count, nbytes in record.buffers)
        if record.window_size < declared:
            print(
                f"[domain-factory:{label}] window_size {record.window_size} is smaller than the "
                f"{declared} bytes of buffers it must hold"
            )
            ok = False
    return ok


def run_once(
    label: str,
    compiled,
    device_ids: list[int],
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    *,
    persistent: bool,
    factory: PlatformDomainFactory | None,
) -> bool:
    """One prepared worker, one dispatch, one set of assertions."""
    print(f"\n[domain-factory:{label}] persistent={persistent} factory={factory is not None}")
    outputs.zero_()
    kwargs = {} if factory is None else {"domain_factory": factory}
    worker = compiled.prepare(persistent=persistent, **kwargs)
    try:
        worker(inputs, outputs)
    finally:
        worker.close()

    ok = check_ring(label, inputs, outputs)
    if factory is None:
        print(f"[domain-factory:{label}] no factory supplied; the runtime allocated its own windows")
    else:
        ok = check_records(label, factory, device_ids) and ok
    return ok


def run(platform: str, device_ids: list[int]) -> int:
    print(f"[domain-factory] platform={platform} devices={device_ids}")
    program = build_ring_put_program()
    compiled = ir.compile(
        program,
        platform=platform,
        distributed_config=DistributedConfig(device_ids=device_ids[:RANKS], num_sub_workers=0),
    )
    inputs, outputs = make_host_buffers()

    ok = run_once("baseline", compiled, device_ids, inputs, outputs, persistent=True, factory=None)
    ok = (
        run_once(
            "factory",
            compiled,
            device_ids,
            inputs,
            outputs,
            persistent=True,
            factory=PlatformDomainFactory(),
        )
        and ok
    )
    ok = (
        run_once(
            "factory-transient",
            compiled,
            device_ids,
            inputs,
            outputs,
            persistent=False,
            factory=PlatformDomainFactory(),
        )
        and ok
    )

    if not ok:
        print("[domain-factory] platform domain-factory checks FAILED")
        return 1
    print("[domain-factory] platform domain-factory checks PASSED")
    return 0


def parse_device_ids(spec: str) -> list[int]:
    """Accept a range ('0-1') or an explicit list ('0,1'), which is what the broker gives."""
    if "-" in spec:
        low, high = (int(part) for part in spec.split("-"))
        ids = list(range(low, high + 1))
    else:
        ids = [int(part) for part in spec.split(",") if part != ""]
    if len(ids) < RANKS:
        raise ValueError(f"the ring put needs at least {RANKS} devices, got {ids}")
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-p", "--platform", default="a2a3")
    parser.add_argument("-d", "--device", default="0-1", help="Range ('0-1') or list ('0,1').")
    cli = parser.parse_args()
    return run(cli.platform, parse_device_ids(cli.device))


if __name__ == "__main__":
    sys.exit(main())
