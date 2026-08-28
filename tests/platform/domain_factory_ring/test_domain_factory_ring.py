# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Runnable wrapper around ``main.py``, the platform-allocated comm-window proof.

Deliberately NOT under ``tests/unit``: it compiles a distributed pypto program and needs two
real NPUs, so it is skipped unless the broker has granted them -- which means it only runs
inside a ``task-submit`` job. The seam itself is unit-tested without hardware in
``tests/unit/platform/test_domain_factory.py``.

    task-submit --device auto --device-num 2 --run "... pytest tests/platform/domain_factory_ring -q"
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = Path(__file__).with_name("main.py")
DEVICES_NEEDED = 2


def granted_devices() -> list[int]:
    """Devices the broker actually handed us.

    Both variables are consulted because the broker has been seen passing the literal request
    string through in one of them; anything that is not a plain comma-separated list of digits
    is rejected rather than parsed optimistically.
    """
    for name in ("TASK_PHYS_DEVICE", "TASK_DEVICE"):
        raw = os.environ.get(name, "").strip()
        if raw and all(part.isdigit() for part in raw.split(",") if part != ""):
            return [int(part) for part in raw.split(",") if part != ""]
    return []


pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("pypto") is None,
        reason="pypto is not installed",
    ),
    pytest.mark.skipif(
        len(granted_devices()) < DEVICES_NEEDED,
        reason=f"needs {DEVICES_NEEDED} broker-granted NPUs (TASK_PHYS_DEVICE / TASK_DEVICE)",
    ),
]


def test_the_platform_allocates_the_windows_a_compiled_program_asks_for():
    devices = granted_devices()[:DEVICES_NEEDED]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPOSITORY_ROOT), env.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "-p", "a2a3", "-d", ",".join(str(d) for d in devices)],
        cwd=REPOSITORY_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "platform domain-factory checks PASSED" in result.stdout
    # The control run must have gone through the runtime, and the other two through us.
    assert "no factory supplied" in result.stdout
    assert "[domain-factory:factory] factory allocated" in result.stdout
    assert "[domain-factory:factory-transient] factory allocated" in result.stdout
