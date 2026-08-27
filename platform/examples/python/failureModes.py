# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Drive every misuse in scenario.py under mpirun and assert it fails loudly, not fatally.

Each scenario here used to either segfault every rank or hang the job with no diagnostic.
The bar is: terminate well within the timeout, never with a signal, and either exit 0 after
cleaning up or exit non-zero with an error message that names the mistake.

This is not itself an MPI program: it spawns mpirun once per scenario.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

TIMEOUT_S = 90.0

FATAL_MARKERS = (
    "Segmentation fault",
    "Signal: Segmentation",
    "signal 11",
    "Bus error",
    "corrupted",
    "double free",
)


class Scenario:
    """One misuse, plus what the bindings must do about it."""

    def __init__(
        self,
        name: str,
        ranks: int,
        policy: str,
        expect_success: bool,
        expect_text: tuple[str, ...],
        summary: str,
    ) -> None:
        self.name = name
        self.ranks = ranks
        self.policy = policy
        self.expect_success = expect_success
        self.expect_text = expect_text
        self.summary = summary


SCENARIOS = (
    Scenario(
        "crash-after-start",
        2,
        "policy.json",
        False,
        ("deliberate failure between start() and stop()",),
        "D1: uncaught exception between start() and stop() must not segfault or hang the peers",
    ),
    Scenario(
        "finalize-live-platform",
        2,
        "policy.json",
        True,
        (
            "Runtime.finalize() with a live Platform; stopping it first",
            "platform stopped: True, runtime finalized: True",
        ),
        "D2: finalize() under a live platform must stop it first",
    ),
    Scenario(
        "zero-channels",
        3,
        "zeroChannelPolicy.json",
        False,
        ("some instance opened no channels",),
        "D4a: a rank with no edges must refuse to start, not wedge the collective",
    ),
    Scenario(
        "missing-assign-instances",
        2,
        "policy.json",
        False,
        ("assign_instances(runtime) must be called",),
        "D4b: a deployment without instance assignment must be rejected up front",
    ),
    Scenario(
        "oversized-push",
        2,
        "policy.json",
        True,
        ("can never be pushed", "rejected after"),
        "D6: a payload that can never fit must be rejected, not spun on uninterruptibly",
    ),
    Scenario(
        "open-then-bail",
        2,
        "policy.json",
        True,
        ("abandoning the platform without starting it",),
        "D7: channels opened but never started must still be released before MPI_Finalize",
    ),
    Scenario(
        "worker-abort",
        2,
        "policy.json",
        False,
        ("aborting the job",),
        "D8: a failing non-root rank must be able to take the job down instead of hanging it",
    ),
)


def run_scenario(scenario: Scenario, mpirun: str, interpreter: str, source_dir: Path, repo_root: Path) -> tuple[bool, str]:
    """Run one scenario under mpirun; return whether it met the bar, and why not if it did not."""
    command = [
        mpirun,
        "-np",
        str(scenario.ranks),
        "--oversubscribe",
        "-x",
        "PYTHONPATH",
        interpreter,
        str(source_dir / "scenario.py"),
        scenario.name,
        str(source_dir / scenario.policy),
    ]

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repo_root)

    started = time.monotonic()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(repo_root),
        env=environment,
        start_new_session=True,
    )
    try:
        output = process.communicate(timeout=TIMEOUT_S)[0]
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output = process.communicate()[0] or ""
        print(output, flush=True)
        return False, f"HUNG: still running after {TIMEOUT_S:.0f}s"
    elapsed = time.monotonic() - started

    print(output, flush=True)
    print(f"    exit={process.returncode} elapsed={elapsed:.2f}s", flush=True)

    for marker in FATAL_MARKERS:
        if marker in output:
            return False, f"died fatally: output contains {marker!r}"

    if process.returncode < 0:
        return False, f"killed by signal {-process.returncode}"

    if scenario.expect_success and process.returncode != 0:
        return False, f"expected a clean exit, got {process.returncode}"
    if not scenario.expect_success and process.returncode == 0:
        return False, "expected a non-zero exit, got 0"

    missing = [text for text in scenario.expect_text if text not in output]
    if missing:
        return False, f"output did not mention {missing}"

    return True, ""


def main() -> int:
    """Run every scenario and report a one-line verdict per scenario."""
    if len(sys.argv) != 4:
        print("Usage: failureModes.py <mpirun> <interpreter> <repo-root>", file=sys.stderr)
        return 2

    mpirun, interpreter, repo_root = sys.argv[1], sys.argv[2], Path(sys.argv[3]).resolve()
    source_dir = Path(__file__).resolve().parent

    results = []
    for scenario in SCENARIOS:
        print(f"\n=== {scenario.name} (-np {scenario.ranks}) : {scenario.summary}", flush=True)
        ok, reason = run_scenario(scenario, mpirun, interpreter, source_dir, repo_root)
        results.append((scenario, ok, reason))
        print(f"    {'PASS' if ok else 'FAIL'} {reason}", flush=True)

    print("\n=== summary", flush=True)
    for scenario, ok, reason in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {scenario.name:<26} {reason}", flush=True)

    failures = [scenario.name for scenario, ok, _ in results if not ok]
    if failures:
        print(f"\n{len(failures)} scenario(s) failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"\nAll {len(results)} failure-mode scenarios behaved.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
