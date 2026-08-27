# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Two-rank round trip over the platform channels, driven entirely from Python.

Run under ``mpirun -np 2``. Rank 0 takes the ``engine`` partition, rank 1 the ``worker``
partition; rank 0 pushes bytes on the ``commands`` edge, rank 1 echoes them back on the
``results`` edge, and rank 0 asserts the payload survived the round trip.

This is the Python equivalent of ``examples/modules/channelController``, and it is the
reference for the teardown shape: ``with`` on both the runtime and the platform, and
``Runtime.abort()`` on any error, because a rank that merely exits leaves its peers
blocked in ``Engine::await()`` and in the collective ``MPI_Finalize``.
"""

from __future__ import annotations

import sys
import time

from pypto_serving.platform import Deployment, Platform, Runtime

READY_TIMEOUT_S = 60.0
MESSAGE_TIMEOUT_S = 60.0
POLL_INTERVAL_S = 0.005

PAYLOAD = b"\x00\x01\x02 the quick brown fox \xfe\xff"


def await_message(channel, label: str) -> bytes:
    """Poll ``channel`` until a message arrives, then copy it out and pop it."""
    deadline = time.monotonic() + MESSAGE_TIMEOUT_S
    while not channel.has_message():
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out waiting for a message on '{label}'")
        time.sleep(POLL_INTERVAL_S)
    return channel.read()


def round_trip(runtime: Runtime, policy_path: str) -> int:
    """Bring up the platform, exchange one message each way, and tear it down."""
    deployment = Deployment.from_json_file(policy_path)
    deployment.assign_edge_managers(runtime)
    deployment.assign_instances(runtime)

    instance_id = runtime.instance_id
    is_root = runtime.is_root
    print(f"[Instance {instance_id}] edges: {deployment.edge_names()}, root: {is_root}", flush=True)

    with Platform(runtime, deployment) as platform:
        # Rank assignment follows the partition declaration order: rank 0 is "engine".
        if is_root:
            commands = platform.open_output("commands")
            results = platform.open_input("results")
        else:
            commands = platform.open_input("commands")
            results = platform.open_output("results")

        platform.start()
        platform.wait_until_ready(timeout_s=READY_TIMEOUT_S)
        print(f"[Instance {instance_id}] channels ready", flush=True)

        if is_root:
            print(
                f"[Instance {instance_id}] sending {len(PAYLOAD)} bytes on 'commands' "
                f"(max {commands.max_message_size} bytes, capacity {commands.capacity})",
                flush=True,
            )
            commands.push(PAYLOAD, message_type=7, group_id=3, sequence_id=11)

            echoed = await_message(results, "results")
            print(f"[Instance {instance_id}] received {len(echoed)} bytes on 'results'", flush=True)

            if echoed != PAYLOAD:
                raise AssertionError(f"round trip mismatch: {echoed!r} != {PAYLOAD!r}")
            assert isinstance(echoed, bytes)
            print(f"[Instance {instance_id}] round trip OK: {echoed!r}", flush=True)
        else:
            received = await_message(commands, "commands")
            print(f"[Instance {instance_id}] received {len(received)} bytes on 'commands', echoing", flush=True)
            results.push(received)

    print(f"[Instance {instance_id}] done", flush=True)
    return 0


def main() -> int:
    """Parse arguments, run the round trip, and abort the whole job on any failure."""
    if len(sys.argv) != 2:
        print("Usage: roundTrip.py <policy.json>", file=sys.stderr)
        return 1
    policy_path = sys.argv[1]

    with Runtime(compute_resource_count=2) as runtime:
        if runtime.instance_count != 2:
            print(f"Error: this example needs exactly 2 instances, got {runtime.instance_count}", file=sys.stderr)
            runtime.abort(1)

        try:
            return round_trip(runtime, policy_path)
        except Exception:
            # One rank failing on its own is not survivable: its peers are either blocked in
            # Engine::await() waiting for a STOP RPC, or in the collective MPI_Finalize.
            import traceback

            traceback.print_exc()
            sys.stderr.flush()
            runtime.abort(1)


if __name__ == "__main__":
    sys.exit(main())
