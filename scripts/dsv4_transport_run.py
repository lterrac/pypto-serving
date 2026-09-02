# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run one DeepSeek V4 completion over either engine<->worker transport.

The point of this script is that the *only* difference between the two runs is
how the server is launched:

* ``--transport queue``    -- ``python launch_server.py ...`` and
  ``PYPTO_SERVING_TRANSPORT`` unset: the engine spawns a worker process and
  creates its own ``multiprocessing.Queue`` pair, exactly as serving does today.
* ``--transport platform`` -- ``mpirun -np 2 python launch_server.py ...`` with
  ``PYPTO_SERVING_TRANSPORT=platform``: the platform creates the channels, rank 0
  is the engine/API and rank 1 is the worker that owns the NPUs.

Everything else -- the model, the CLI flags, the prompt, the sampling, the
shutdown sequence -- is one code path, so a difference in the generated text is
a difference in the transport and not in how the run was driven.

Why the HTTP request is driven from *here* rather than from inside the server
process: under ``mpirun`` every rank runs the same argv, so a driver that spawns
the server as a subprocess (as
``.agents/skills/profile-dsv4-serving-strace/scripts/run_profile.py`` does) would
be spawned twice and would try to serve twice. This process is not an MPI rank;
it launches ``mpirun`` and talks to the resulting server over HTTP.

The server entry point is the profiling skill's ``launch_server.py``, unchanged,
because it is what pins ``total_kv_pages`` -- the one knob with no CLI flag, and
the one the known-good reference run used.

Profiling is deliberately not requested: the platform transport has two edges and
refuses profile commands (``ProfilingUnsupportedError`` -> HTTP 400), so asking
for it would make the two runs differ for a reason that has nothing to do with
the tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SERVER_ENTRY = (
    REPOSITORY_ROOT
    / ".agents"
    / "skills"
    / "profile-dsv4-serving-strace"
    / "scripts"
    / "launch_server.py"
)

STARTUP_TIMEOUT_SECONDS = 1800
REQUEST_TIMEOUT_SECONDS = 1800
POLL_SECONDS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=("queue", "platform"), required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--served-model-name", default="dsv4-flash-w8a8")
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--prompt", default="Huawei is")
    parser.add_argument("--use-compile-cache", action="store_true")
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=5.0,
        help=(
            "Idle time between the completion and the shutdown. The replica reports "
            "per-request metrics (DeepSeek's MTP acceptance among them) from "
            "release_finished_requests, which runs on a LATER engine step than the one "
            "that produced the last token -- the engine has to send the finished ids to "
            "the worker and get a result back. Stopping the server the instant the HTTP "
            "response is written races that round trip and the metrics never appear."
        ),
    )
    parser.add_argument(
        "--shutdown-timeout",
        type=float,
        default=180.0,
        help="Seconds to wait for a SIGINT shutdown before escalating.",
    )
    return parser.parse_args()


def unused_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def server_command(args: argparse.Namespace, port: int, devices: str) -> list[str]:
    """Build the launch command; the transport only changes its prefix."""
    parallel_size = len(devices.split(","))
    command = [
        args.python,
        str(SERVER_ENTRY),
        "--model",
        str(args.model_dir),
        "--served-model-name",
        args.served_model_name,
        "--backend",
        "npu",
        "--platform",
        "a2a3",
        "--devices",
        devices,
        "--dp",
        str(parallel_size),
        "--ep",
        str(parallel_size),
        "--block-size",
        "128",
        "--max-model-len",
        "260",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "512",
        "--long-prefill-token-threshold",
        "2048",
        "--enable-mtp",
        "--no-enable-prefix-caching",
        "--port",
        str(port),
        "--show-startup-logs",
    ]
    if args.use_compile_cache:
        command.append("--use-compile-cache")
    if args.transport == "queue":
        return command
    # --bind-to none matters: OpenMPI binds a 2-process job to one core per rank
    # by default, and the worker rank forks eight chip processes plus the pypto
    # scheduler threads. Inheriting a single-core affinity mask would serialise
    # all of them onto that core.
    return [
        "mpirun",
        "-np",
        "2",
        "--bind-to",
        "none",
        "--oversubscribe",
        "-x",
        "PYPTO_SERVING_TRANSPORT",
        "-x",
        "PYTHONPATH",
        *command,
    ]


def wait_for_health(process: subprocess.Popen, port: int) -> None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    next_heartbeat = 0.0
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"server exited during startup with code {return_code}")
        try:
            with opener.open(f"http://127.0.0.1:{port}/health", timeout=5) as response:
                payload = json.loads(response.read())
            if response.status == 200 and payload == {"status": "ok"}:
                print("DeepSeek server is healthy", flush=True)
                return
        except (OSError, TimeoutError, ValueError, urllib.error.URLError) as exc:
            last_error = exc
        now = time.monotonic()
        if now >= next_heartbeat:
            print("Waiting for DeepSeek server startup...", flush=True)
            next_heartbeat = now + 30
        time.sleep(POLL_SECONDS)
    raise TimeoutError(f"server did not become healthy: {last_error}")


def request_completion(port: int, model_name: str, prompt: str, max_tokens: int) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(
            {
                "model": model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with opener.open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return json.loads(response.read())


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="replace").strip()
    except OSError:
        return ""


def _parent_pid(pid: str) -> str:
    for line in _read(Path("/proc") / pid / "status").splitlines():
        if line.startswith("PPid:"):
            return line.split()[1]
    return ""


def _mpi_rank(pid: str) -> str:
    """This process's MPI rank, from its environment, or "" if it is not a rank.

    ``OMPI_COMM_WORLD_RANK`` is exported into each rank by ``mpirun``, so
    ``/proc/<pid>/environ`` identifies the rank without the process having to
    cooperate. Nothing else distinguishes them: under SPMD both ranks have
    byte-identical argv.
    """
    try:
        raw = (Path("/proc") / pid / "environ").read_bytes()
    except OSError:
        return ""
    for item in raw.decode(errors="replace").split("\x00"):
        name, _, value = item.partition("=")
        if name == "OMPI_COMM_WORLD_RANK":
            return value
    return ""


def _owning_rank(pid: str) -> str:
    """The MPI rank this process belongs to, following ppid links upwards.

    The eight per-card chip processes are children of whichever rank forked
    them, and they inherit its environment, so in practice the first lookup
    already answers it; the walk is there for any layer that scrubs the
    environment.
    """
    seen: set[str] = set()
    current = pid
    while current and current != "0" and current not in seen:
        seen.add(current)
        rank = _mpi_rank(current)
        if rank:
            return rank
        current = _parent_pid(current)
    return ""


def _process_record(pid: str) -> dict[str, str]:
    return {
        "pid": pid,
        "ppid": _parent_pid(pid),
        "rank": _mpi_rank(pid),
        "owning_rank": _owning_rank(pid),
        "cmdline": _read(Path("/proc") / pid / "cmdline").replace("\x00", " ").strip(),
    }


def device_owners() -> list[dict[str, object]]:
    """Report which live processes hold an Ascend device node open.

    The question this answers is "did rank 0 open an NPU?". Under the platform
    transport only the worker rank may: the engine rank runs the API and the
    scheduler and must never touch a device. Reading ``/proc/<pid>/fd`` is the
    direct evidence -- ``npu-smi`` reports per-card processes but not their
    ancestry, and ancestry is what identifies the rank.
    """
    owners: list[dict[str, object]] = []
    for entry in sorted(Path("/proc").iterdir()):
        if not entry.name.isdigit():
            continue
        try:
            targets = [os.readlink(str(item)) for item in (entry / "fd").iterdir()]
        except OSError:
            continue
        devices = sorted({t for t in targets if "davinci" in t or "devmm" in t})
        if not devices:
            continue
        owners.append({**_process_record(entry.name), "devices": devices})
    return owners


def process_tree() -> list[dict[str, str]]:
    """Flat pid/ppid/rank/cmdline listing, so the artifact reads after the fact."""
    return [
        _process_record(entry.name)
        for entry in sorted(Path("/proc").iterdir())
        if entry.name.isdigit()
    ]


def _engine_rank_pid(pidfile: Path | None) -> int | None:
    """The engine rank's own pid, if it published one."""
    if pidfile is None:
        return None
    try:
        return int(pidfile.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def stop_server(
    process: subprocess.Popen, timeout: float, pidfile: Path | None = None
) -> None:
    """Signal the ENGINE RANK, not ``mpirun``, then escalate.

    Which process gets the signal decides whether this job exits cleanly. Measured on
    the deployment container (OpenMPI 4.1.2), with a two-rank job whose ranks both
    return 0:

        SIGTERM to mpirun           -> mpirun exits 1, and both ranks are left unreaped
        SIGTERM to the engine rank  -> both ranks exit 0 and mpirun exits 0

    The reason is that mpirun forwards the signal to every rank and kills them, so the
    engine never runs its own shutdown and never sends the worker a ShutdownCommand.
    Signalling the engine rank instead lets that chain run: uvicorn's shutdown ->
    ``AsyncLLMEngine.stop()`` -> ``ShutdownCommand``, which is also what releases rank 1
    from ``get(timeout=None)``.

    On the queue transport ``process`` IS the server, so there is no pidfile and nothing
    changes. Falling back to ``process`` when the pid is unavailable keeps the old
    behaviour rather than leaving the job running.
    """
    if process.poll() is not None:
        return
    target = _engine_rank_pid(pidfile)
    if target is None:
        target = process.pid
    for signal_number, wait in ((signal.SIGINT, timeout), (signal.SIGTERM, 30.0)):
        try:
            os.kill(target, signal_number)
            process.wait(timeout=wait)
            return
        except (OSError, subprocess.TimeoutExpired):
            continue
    try:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        pass


def main(args: argparse.Namespace) -> int:
    artifact_dir = args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    server_log = artifact_dir / "server.log"
    if server_log.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {server_log}")
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {args.model_dir}")
    if not SERVER_ENTRY.is_file():
        raise FileNotFoundError(f"server entry point missing: {SERVER_ENTRY}")

    device_list = [item.strip() for item in args.devices.split(",") if item.strip()]
    if len(device_list) != 8 or len(set(device_list)) != 8:
        raise ValueError(f"--devices must contain exactly eight unique IDs, got {args.devices!r}")
    if any(not item.isdigit() for item in device_list):
        raise ValueError(f"--devices must contain non-negative integer IDs, got {args.devices!r}")
    devices = ",".join(device_list)

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPOSITORY_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    engine_pidfile: Path | None = None
    if args.transport == "platform":
        env["PYPTO_SERVING_TRANSPORT"] = "platform"
        # mpirun refuses to run as root without this, and the container is root.
        env.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
        env.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
        # Ask the engine rank to publish its pid, so shutdown can signal IT rather than
        # mpirun. mpirun forwards a signal to every rank and kills them, which skips the
        # engine's own shutdown, exits 1 and leaves the ranks unreaped -- see stop_server.
        engine_pidfile = artifact_dir / "engine-rank.pid"
        engine_pidfile.unlink(missing_ok=True)
        env["PYPTO_SERVING_PLATFORM_PIDFILE"] = str(engine_pidfile)
    else:
        env.pop("PYPTO_SERVING_TRANSPORT", None)
        env.pop("PYPTO_SERVING_PLATFORM_PIDFILE", None)

    port = unused_local_port()
    command = server_command(args, port, devices)
    print(f"Transport: {args.transport}", flush=True)
    print(f"Server command: {' '.join(command)}", flush=True)
    print(f"Server log: {server_log}", flush=True)

    exit_code = 0
    with server_log.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY_ROOT,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
            env=env,
        )
        try:
            wait_for_health(process, port)
            (artifact_dir / "device-owners.json").write_text(
                json.dumps(
                    {"transport": args.transport, "owners": device_owners()}, indent=2
                )
                + "\n",
                encoding="utf-8",
            )
            (artifact_dir / "process-tree.json").write_text(
                json.dumps(process_tree(), indent=2) + "\n",
                encoding="utf-8",
            )
            started = time.perf_counter()
            response = request_completion(
                port, args.served_model_name, args.prompt, args.max_tokens
            )
            elapsed = time.perf_counter() - started
            print(f"Completion elapsed_s={elapsed:.6f}", flush=True)
            print(
                f"Completion response: {json.dumps(response, ensure_ascii=False)}",
                flush=True,
            )
            (artifact_dir / "completion-response.json").write_text(
                json.dumps(response, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            choices = response.get("choices", [])
            if len(choices) != 1:
                raise AssertionError(f"expected one choice, got {choices!r}")
            (artifact_dir / "completion.txt").write_text(
                choices[0].get("text", ""), encoding="utf-8"
            )
            usage = response.get("usage") or {}
            if usage.get("completion_tokens") != args.max_tokens:
                raise AssertionError(
                    f"expected {args.max_tokens} completion tokens, got usage={usage!r}"
                )
        finally:
            if args.settle_seconds > 0 and process.poll() is None:
                print(f"Letting the replica settle for {args.settle_seconds:g}s...", flush=True)
                time.sleep(args.settle_seconds)
            print("Stopping server gracefully...", flush=True)
            stop_server(process, args.shutdown_timeout, engine_pidfile)
            exit_code = process.returncode if process.returncode is not None else 1
            print(f"Server exit code: {exit_code}", flush=True)
    # Recorded, not used as the verdict -- the completion assertions above are.
    # On the platform transport this is expected to be 1 even for a perfectly clean
    # run: the signal goes to `mpirun`, and OpenMPI 4.1.2's mpirun exits 1 when it is
    # itself interrupted, whatever its ranks returned (measured directly with a
    # two-rank job whose ranks both exit 0). The queue transport exits 0 because the
    # signal goes to uvicorn, which handles it.
    (artifact_dir / "server-exit-code.txt").write_text(f"{exit_code}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    parsed = parse_args()
    parsed.artifact_dir = parsed.artifact_dir.resolve()
    raise SystemExit(main(parsed))
