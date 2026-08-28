#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Job payload for `task-submit --device auto --device-num 8`: run one DeepSeek V4
# completion over the queue transport, the platform transport, or both.
#
#   bash scripts/dsv4_transport_run.sh <artifact-root> [queue|platform|both]
#
# With `both`, the queue run goes first and the platform run is only attempted if
# it produced the expected number of tokens. That ordering is the whole point: a
# platform-transport failure can only be attributed to the transport if the same
# checkout has just served the same prompt without it.
#
# Runtime environment is the one the known-good reference run used
# (.agents/skills/profile-dsv4-serving-strace/scripts/run_profile.sh), minus the
# profiling: PYPTO_RUNTIME_LOG is left at its default because host STRACE is not
# needed to compare generated text, and SA profiling is refused by the platform
# transport by design.

set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

ARTIFACT_ROOT=${1:?usage: dsv4_transport_run.sh <artifact-root> [queue|platform|both]}
MODE=${2:-both}

MODEL_DIR=${PYPTO_DSV4_MODEL_DIR:-/mounted_home/models/DeepSeek-V4-Flash-w8a8}
# Deliberately NOT `${PTOAS_ROOT:-...}`: the container image already exports
# PTOAS_ROOT=/opt/ptoas-bin, which is a different (older) assembler and has no
# Python beside it. Honouring the inherited value silently ran the wrong stack --
# it failed loudly here only because the interpreter path is derived from it. The
# override knob is a distinct name so an inherited PTOAS_ROOT can never win.
PTOAS_ROOT=${PYPTO_DSV4_PTOAS_ROOT:-/mounted_home/pypto-private/venv}
PYTHON_BIN=${PYPTO_PROFILE_PYTHON:-$PTOAS_ROOT/bin/python}
MAX_TOKENS=${PYPTO_DSV4_MAX_TOKENS:-20}
PROMPT=${PYPTO_DSV4_PROMPT:-"Huawei is"}
# Reuse the kernels a previous run left in build_output. Only safe with the same
# devices, config and kernel sources -- the cache is not fingerprinted -- but it
# turns a ~9-minute startup into a ~2-minute one, which is the difference between
# holding eight cards for twenty minutes and for eight.
USE_COMPILE_CACHE=${PYPTO_USE_COMPILE_CACHE:-0}
SETTLE_SECONDS=${PYPTO_DSV4_SETTLE_SECONDS:-5}

# The broker has been seen passing the literal request string rather than a
# device list, so take the first candidate that is actually a list of digits.
DEVICES=""
for candidate in "${PYPTO_PROFILE_DEVICES:-}" "${TASK_PHYS_DEVICE:-}" "${TASK_DEVICE:-}" \
                 "${ASCEND_RT_VISIBLE_DEVICES:-}"; do
  case "$candidate" in
    ""|*[!0-9,]*) continue ;;
    *) DEVICES="$candidate"; break ;;
  esac
done
if [ -z "$DEVICES" ]; then
  echo "FATAL_NO_DEVICES: none of PYPTO_PROFILE_DEVICES/TASK_PHYS_DEVICE/TASK_DEVICE/ASCEND_RT_VISIBLE_DEVICES held a device list" >&2
  exit 1
fi

export PTOAS_ROOT
export LD_LIBRARY_PATH="$PTOAS_ROOT/lib:${LD_LIBRARY_PATH:-}"

# The container image exports PYTHONPATH=/opt/pypto/runtime/python:... , which
# shadows the `simpler` package installed in $PTOAS_ROOT with the image's shared
# simpler *source tree*. That tree moves independently of this venv, and
# simpler.task_interface refuses to import when the compiled `_task_interface`
# was built from a different revision of the tree it is being read from:
#
#   ImportError: _task_interface was built from 93adc3865e65,
#                but this source tree is at 3165cc89b6ea.
#
# The venv's own copy carries no .git, so the same guard passes there -- and it
# is the copy the extension was actually built against. Drop just that one entry
# and leave the CANN entries alone.
_kept_pythonpath=""
IFS=':' read -r -a _pythonpath_parts <<< "${PYTHONPATH:-}"
for _part in "${_pythonpath_parts[@]:-}"; do
  case "$_part" in
    ""|/opt/pypto/runtime/python) continue ;;
    *) _kept_pythonpath="${_kept_pythonpath:+$_kept_pythonpath:}$_part" ;;
  esac
done
export PYTHONPATH="$REPO_ROOT${_kept_pythonpath:+:$_kept_pythonpath}"
export SIMPLER_DEVICE_STRACE_ENABLE=0
export PTO2_RING_DEP_POOL=${PTO2_RING_DEP_POOL:-131072}
export PTO2_RING_TASK_WINDOW=${PTO2_RING_TASK_WINDOW:-131072}
export PTO2_RING_HEAP=${PTO2_RING_HEAP:-2147483648}
export SIMPLER_OP_EXECUTE_TIMEOUT_US=${SIMPLER_OP_EXECUTE_TIMEOUT_US:-400000000}
export SIMPLER_STREAM_SYNC_TIMEOUT_MS=${SIMPLER_STREAM_SYNC_TIMEOUT_MS:-440000}
export SIMPLER_SCHEDULER_TIMEOUT_MS=${SIMPLER_SCHEDULER_TIMEOUT_MS:-320000}
export SERVING_WORKER_STEP_TIMEOUT=${SERVING_WORKER_STEP_TIMEOUT:-1800}
export NO_PROXY=${NO_PROXY:-127.0.0.1,localhost}
export no_proxy=${no_proxy:-127.0.0.1,localhost}
# mpirun refuses to run as root; the container is root.
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

if [ ! -x "$PYTHON_BIN" ]; then
  echo "FATAL: interpreter $PYTHON_BIN is missing or not executable (PTOAS_ROOT=$PTOAS_ROOT)" >&2
  exit 1
fi
if [ ! -d "$MODEL_DIR" ]; then
  echo "FATAL: model directory $MODEL_DIR does not exist" >&2
  exit 1
fi

mkdir -p "$ARTIFACT_ROOT"

run_one() {
  transport=$1
  out="$ARTIFACT_ROOT/$transport"
  echo "=================================================================="
  echo "=== transport=$transport devices=$DEVICES artifacts=$out"
  echo "=================================================================="
  rm -rf "$out"
  extra=()
  case "${USE_COMPILE_CACHE,,}" in
    1|true|yes|on) extra+=(--use-compile-cache) ;;
  esac
  "$PYTHON_BIN" "$SCRIPT_DIR/dsv4_transport_run.py" \
    --transport "$transport" \
    --artifact-dir "$out" \
    --model-dir "$MODEL_DIR" \
    --devices "$DEVICES" \
    --python "$PYTHON_BIN" \
    --max-tokens "$MAX_TOKENS" \
    --prompt "$PROMPT" \
    --settle-seconds "$SETTLE_SECONDS" \
    "${extra[@]}"
  rc=$?
  echo "--- transport=$transport driver exit=$rc"
  if [ -f "$out/completion.txt" ]; then
    echo "--- transport=$transport completion.txt:"
    cat "$out/completion.txt"
    echo
  fi
  if [ -f "$out/server.log" ]; then
    echo "--- transport=$transport MTP acceptance:"
    grep -a "MTP acceptance" "$out/server.log" || echo "(none)"
    echo "--- transport=$transport server.log tail:"
    tail -c 20000 "$out/server.log"
  fi
  return $rc
}

case "$MODE" in
  queue|platform)
    run_one "$MODE"
    exit $?
    ;;
  both)
    if ! run_one queue; then
      echo "BASELINE_FAILED: the queue transport did not serve the prompt on this checkout;" >&2
      echo "not attempting the platform transport, because a failure there could not be" >&2
      echo "distinguished from this one." >&2
      exit 1
    fi
    run_one platform
    exit $?
    ;;
  *)
    echo "unknown mode: $MODE (expected queue, platform or both)" >&2
    exit 2
    ;;
esac
