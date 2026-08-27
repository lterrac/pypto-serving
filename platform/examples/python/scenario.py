# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""One misuse of the platform bindings per scenario, run under mpirun by failureModes.py.

Each of these used to segfault every rank or wedge the job forever. They must now fail
loudly and promptly, or clean up and exit 0. Nothing here is a supported usage pattern;
roundTrip.py is the reference for that.
"""

from __future__ import annotations

import sys
import time

from pypto_serving.platform import Deployment, Platform, Runtime


def prepare(runtime: Runtime, policy_path: str, *, assign_instances: bool = True) -> Deployment:
    """Load the policy and attach the runtime's managers to it."""
    deployment = Deployment.from_json_file(policy_path)
    deployment.assign_edge_managers(runtime)
    if assign_instances:
        deployment.assign_instances(runtime)
    return deployment


def open_side(open_call, name: str, not_ours: str, opened: list) -> None:
    """Open one edge, tolerating only "this instance does not own that side of it".

    Both HICR_THROW_LOGIC and HICR_THROW_RUNTIME surface as RuntimeError, so a bare
    ``except RuntimeError: pass`` would hide a real failure as "not ours".
    """
    try:
        opened.append(open_call(name))
    except RuntimeError as error:
        if not_ours not in str(error):
            raise


def open_all(
    platform: Platform, deployment: Deployment, runtime: Runtime, *, skip: tuple[str, ...] = ()
) -> tuple[list, list]:
    """Open every edge this instance owns, without needing to know which side it is on."""
    inputs, outputs = [], []
    for name in deployment.edge_names():
        if name in skip:
            continue
        open_side(platform.open_output, name, "is not produced by this instance", outputs)
        open_side(platform.open_input, name, "is not consumed by this instance", inputs)
    print(f"[Instance {runtime.instance_id}] opened {len(inputs)} inputs, {len(outputs)} outputs", flush=True)
    return inputs, outputs


def scenario_crash_after_start(runtime: Runtime, policy_path: str) -> int:
    """D1: uncaught exception between start() and stop(), with no `with` and no handler.

    Only the Platform destructor stands between this and unwinding into ~Engine while TaskR
    service workers are still inside reconcile() on boost fibers.
    """
    deployment = prepare(runtime, policy_path)
    platform = Platform(runtime, deployment)
    open_all(platform, deployment, runtime)
    platform.start()
    platform.wait_until_ready(timeout_s=60.0)
    raise RuntimeError("deliberate failure between start() and stop()")


def scenario_finalize_live_platform(runtime: Runtime, policy_path: str) -> int:
    """D2: Runtime.finalize() underneath a started platform. Must stop it first, not segfault."""
    deployment = prepare(runtime, policy_path)
    platform = Platform(runtime, deployment)
    open_all(platform, deployment, runtime)
    platform.start()
    platform.wait_until_ready(timeout_s=60.0)

    instance_id = runtime.instance_id
    runtime.finalize()
    print(
        f"[Instance {instance_id}] finalize returned, platform stopped: {platform.is_stopped}, "
        f"runtime finalized: {runtime.is_finalized}",
        flush=True,
    )
    return 0


def scenario_zero_channels(runtime: Runtime, policy_path: str) -> int:
    """D4a: a policy that leaves one rank with no edges at all.

    reconcile() only enters the (collective) memory-slot exchange on ranks that have
    channels to create, so that rank would sail past a collective its peers sit in.
    """
    deployment = prepare(runtime, policy_path)
    platform = Platform(runtime, deployment)
    open_all(platform, deployment, runtime)
    platform.start()
    platform.stop()
    return 0


def scenario_missing_assign_instances(runtime: Runtime, policy_path: str) -> int:
    """D4b: forgetting assign_instances, after which every partition keeps coordinator id 0."""
    deployment = prepare(runtime, policy_path, assign_instances=False)
    platform = Platform(runtime, deployment)
    open_all(platform, deployment, runtime)
    platform.start()
    platform.stop()
    return 0


def scenario_partial_open(runtime: Runtime, policy_path: str) -> int:
    """Ranks that open DIFFERENT edge sets.

    The peers agree on how many channels exist but not on which, so both pass a count-based
    guard, both enter the memory slot exchange, and one then asks for a global key nobody
    registered -- which faults rather than raising. Rank 0 opens both its edges, rank 1 skips
    the "results" edge it is the producer of, which is exactly the shape the serving layer
    produces when it opens only the edges it needs.
    """
    deployment = prepare(runtime, policy_path)
    platform = Platform(runtime, deployment)
    open_all(platform, deployment, runtime, skip=() if runtime.is_root else ("results",))
    platform.start()
    platform.stop()
    return 0


def scenario_oversized_push(runtime: Runtime, policy_path: str) -> int:
    """D6: a payload larger than the edge's payload ring can never fit, so push would spin forever."""
    deployment = prepare(runtime, policy_path)
    with Platform(runtime, deployment) as platform:
        inputs, outputs = open_all(platform, deployment, runtime)
        platform.start()
        platform.wait_until_ready(timeout_s=60.0)

        if runtime.is_root:
            output = outputs[0]
            oversized = b"x" * (output.max_message_size + 1)
            print(
                f"[Instance {runtime.instance_id}] pushing {len(oversized)} bytes into a "
                f"{output.max_message_size} byte edge",
                flush=True,
            )
            started = time.monotonic()
            try:
                output.push(oversized)
            except ValueError as error:
                print(f"[Instance {runtime.instance_id}] rejected after {time.monotonic() - started:.3f}s: {error}", flush=True)
            else:
                raise AssertionError("oversized push was accepted")
        del inputs
    return 0


def scenario_open_then_bail(runtime: Runtime, policy_path: str) -> int:
    """D7: open channels -- which allocates MPI memory slots -- then never start or stop."""
    deployment = prepare(runtime, policy_path)
    platform = Platform(runtime, deployment)
    open_all(platform, deployment, runtime)
    print(f"[Instance {runtime.instance_id}] abandoning the platform without starting it", flush=True)
    return 0


def scenario_worker_abort(runtime: Runtime, policy_path: str) -> int:
    """D8: a non-root rank fails. Without abort() it would leave the root blocked forever."""
    deployment = prepare(runtime, policy_path)
    with Platform(runtime, deployment) as platform:
        inputs, outputs = open_all(platform, deployment, runtime)
        platform.start()
        platform.wait_until_ready(timeout_s=60.0)

        if runtime.is_root:
            # Wait for a message the worker is never going to send.
            deadline = time.monotonic() + 120.0
            while not inputs[0].has_message():
                if time.monotonic() > deadline:
                    raise AssertionError("root was not aborted")
                time.sleep(0.01)
        else:
            print(f"[Instance {runtime.instance_id}] aborting the job", flush=True)
            sys.stderr.flush()
            sys.stdout.flush()
            runtime.abort(3)
        del outputs
    return 0


SCENARIOS = {
    "crash-after-start": scenario_crash_after_start,
    "finalize-live-platform": scenario_finalize_live_platform,
    "zero-channels": scenario_zero_channels,
    "missing-assign-instances": scenario_missing_assign_instances,
    "partial-open": scenario_partial_open,
    "oversized-push": scenario_oversized_push,
    "open-then-bail": scenario_open_then_bail,
    "worker-abort": scenario_worker_abort,
}


def main() -> int:
    """Run one named scenario. Failures are deliberately left uncaught."""
    if len(sys.argv) != 3 or sys.argv[1] not in SCENARIOS:
        print(f"Usage: scenario.py <{'|'.join(SCENARIOS)}> <policy.json>", file=sys.stderr)
        return 2

    runtime = Runtime(compute_resource_count=2)
    return SCENARIOS[sys.argv[1]](runtime, sys.argv[2])


if __name__ == "__main__":
    sys.exit(main())
