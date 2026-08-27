# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Runnable wrapper around ``two_rank_replica.py``.

Deliberately NOT under ``tests/unit``: it needs ``mpirun`` and the native
extension (``ninja -C platform/build`` with ``-DbuildPythonBindings=true``), so
it is skipped rather than failed anywhere those are missing -- including the
``platform-build`` CI runner, which has neither torch nor a checkpoint. Run it
with ``pytest tests/platform -q``.

Two of these cases are hang regressions. They assert termination *and* a
non-zero status, and each carries a timeout well above the bounded wait it is
testing, so a genuine wedge fails the test instead of hanging the suite.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).with_name("two_rank_replica.py")

pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("pypto_serving.platform._native") is None,
        reason="native platform extension not built (-DbuildPythonBindings=true)",
    ),
    pytest.mark.skipif(shutil.which("mpirun") is None, reason="mpirun not available"),
]


def _run(extra_env: dict[str, str], *, timeout: float = 600.0):
    env = dict(os.environ)
    env["PYPTO_SERVING_TRANSPORT"] = "platform"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPOSITORY_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env.update(extra_env)
    forwarded: list[str] = []
    for name in ("PYTHONPATH", "PYPTO_SERVING_TRANSPORT", *extra_env):
        forwarded += ["-x", name]
    started = time.monotonic()
    result = subprocess.run(
        ["mpirun", "-np", "2", "--oversubscribe", *forwarded, sys.executable, str(SCRIPT)],
        cwd=REPOSITORY_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result, time.monotonic() - started


def test_commands_and_results_cross_the_platform_channels():
    result, _ = _run({})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "commands and results crossed the platform channels" in result.stdout


def test_backpressure_refuses_dispatch_without_losing_a_token():
    """More steps in flight than the ring has slots: the loop must fall through
    to _await_and_apply_oldest() and still produce exactly the right tokens."""
    result, _ = _run({"TWO_RANK_FORCE_BACKPRESSURE": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "the command channel was full 0 time(s)" not in result.stdout
    assert "commands and results crossed the platform channels" in result.stdout


def test_the_production_pipelined_worker_loop_runs_over_the_channels():
    """The serial loop is not the path async scheduling takes in production; the
    pipelined one pushes results from a different thread."""
    result, _ = _run({"TWO_RANK_PIPELINED_WORKER": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
    assert "pipelined busy loop selected" in result.stderr + result.stdout
    assert "commands and results crossed the platform channels" in result.stdout


def test_a_dying_engine_loop_takes_the_replica_down_with_a_failing_status():
    result, _ = _run({"TWO_RANK_INJECT_LOOP_FAILURE": "1"})
    assert result.returncode != 0, (
        "a replica whose engine loop died must not exit 0: a supervisor would "
        "see a clean shutdown and never restart it\n" + result.stdout + result.stderr
    )
    assert "a dying engine loop took the replica down" in result.stdout


def test_a_wedged_blocking_push_now_gives_up_instead_of_hanging():
    """Regression for the reviewed F2 hang (previously SIGKILL after 90s)."""
    result, elapsed = _run(
        {"TWO_RANK_REPRO_F2": "1", "PYPTO_SERVING_PLATFORM_PUT_TIMEOUT": "5"},
        timeout=180.0,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "wedged pushing a result nobody will read" in result.stdout, (
        "the scenario did not reach the wedge\n" + result.stdout + result.stderr
    )
    assert elapsed < 60.0, (
        f"took {elapsed:.1f}s: the harness idles for 60s once the worker is "
        "wedged, so anything at or above that means the bounded put did not fire"
    )
    assert "still full after" in result.stderr


def test_an_engine_that_returns_without_shutting_the_worker_down_fails_the_job():
    """Regression for the reviewed F3 hang (previously SIGKILL after 90s)."""
    result, elapsed = _run({"TWO_RANK_REPRO_F3": "1"}, timeout=180.0)
    assert result.returncode != 0, result.stdout + result.stderr
    assert elapsed < 60.0, f"took {elapsed:.1f}s; expected a prompt failure"
    assert "without sending the worker rank a" in result.stderr


def test_an_unfittable_step_fails_its_requests_not_the_replica():
    """A StepCommand larger than the edge can carry must not kill the replica."""
    result, _ = _run(
        {
            "TWO_RANK_REPRO_OVERSIZED": "1",
            "PYPTO_SERVING_PLATFORM_MAX_MESSAGE_BYTES": "128",
        },
        timeout=180.0,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "failed its requests, not the replica" in result.stdout
    assert "does not fit the command channel" in result.stderr
