# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The engine rank publishes its pid so shutdown can signal it instead of mpirun.

Signalling mpirun kills both ranks before the engine can send the worker its
ShutdownCommand: measured on OpenMPI 4.1.2, that exits 1 and leaves both ranks
unreaped, while signalling the engine rank exits 0 on both. The pid is therefore
load-bearing, and it cannot be derived from outside -- mpirun's children carry no
rank label.
"""

import os

from pypto_serving.serving.transport import platform_launch


def test_pidfile_holds_this_pid_while_the_engine_runs(tmp_path):
    path = tmp_path / "engine-rank.pid"
    os.environ[platform_launch.PIDFILE_ENV_VAR] = str(path)
    try:
        with platform_launch._engine_pidfile():
            assert path.read_text(encoding="utf-8").strip() == str(os.getpid())
    finally:
        os.environ.pop(platform_launch.PIDFILE_ENV_VAR, None)
    # Removed on the way out: a stale pid could later be signalled, and by then it
    # belongs to whatever process the OS has since given that number to.
    assert not path.exists()


def test_no_pidfile_is_written_when_the_environment_names_none():
    os.environ.pop(platform_launch.PIDFILE_ENV_VAR, None)
    with platform_launch._engine_pidfile():
        pass  # nothing to assert beyond not raising


def test_an_unwritable_path_warns_and_still_serves(tmp_path, caplog):
    path = tmp_path / "missing-dir" / "engine-rank.pid"
    os.environ[platform_launch.PIDFILE_ENV_VAR] = str(path)
    try:
        with caplog.at_level("WARNING"):
            with platform_launch._engine_pidfile():
                pass
    finally:
        os.environ.pop(platform_launch.PIDFILE_ENV_VAR, None)
    assert not path.exists()
    # A replica that cannot publish its pid still serves; it just cannot be shut down
    # gracefully, so the failure has to be visible rather than silent.
    assert any(platform_launch.PIDFILE_ENV_VAR in record.message for record in caplog.records)
