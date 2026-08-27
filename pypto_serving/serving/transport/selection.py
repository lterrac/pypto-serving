# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Environment-driven selection between the queue and platform transports.

Reads ``PYPTO_SERVING_TRANSPORT`` (``queue`` | ``platform``), defaulting to
``queue`` so existing ``multiprocessing.Queue``-based behaviour is untouched
unless a caller opts in.

The native extension (``pypto_serving.platform._native``) is never imported
at module import time here -- only inside ``is_platform_extension_available``,
lazily, on demand -- so importing this module never fails on an environment
that lacks it. Whether the extension is actually present is environment- and
build-dependent (it may be built in one worktree/container and absent in
another); code that needs a specific answer must call
``is_platform_extension_available()`` itself rather than assume either way.
"""

from __future__ import annotations

import enum
import os
from collections.abc import Mapping

TRANSPORT_ENV_VAR = "PYPTO_SERVING_TRANSPORT"
_DEFAULT_TRANSPORT = "queue"


class TransportKind(enum.Enum):
    """Which underlying transport an engine replica should use."""

    QUEUE = "queue"
    PLATFORM = "platform"


def is_platform_extension_available() -> bool:
    """Return whether the native platform extension can be imported.

    Import is attempted lazily, right here, not at module load time: the
    extension is being built in parallel against the same contract and does
    not exist in this tree yet. This function is the only place that probes
    for it.
    """
    try:
        import pypto_serving.platform._native  # noqa: F401
    except ImportError:
        return False
    return True


def require_platform_extension_available() -> None:
    """Raise if the native platform extension is not importable.

    Deliberately raises rather than silently continuing: a caller that asked
    for the platform transport and silently got the queue transport instead
    would report a platform run that never happened.
    """
    if not is_platform_extension_available():
        raise RuntimeError(
            "PYPTO_SERVING_TRANSPORT=platform was requested but "
            "pypto_serving.platform._native is not importable (native "
            "extension not built/installed in this environment). Refusing "
            "to silently fall back to the queue transport: falling back "
            "here would report a platform run that never happened."
        )


def transport_kind_from_env(env: Mapping[str, str] | None = None) -> TransportKind:
    """Parse ``PYPTO_SERVING_TRANSPORT`` from ``env`` (default ``os.environ``).

    Does not check availability -- see ``resolve_transport_kind`` for the
    variant that also enforces the platform extension is present.
    """
    source = os.environ if env is None else env
    raw = source.get(TRANSPORT_ENV_VAR, _DEFAULT_TRANSPORT)
    normalized = raw.strip().lower()
    try:
        return TransportKind(normalized)
    except ValueError as exc:
        valid = ", ".join(kind.value for kind in TransportKind)
        raise ValueError(
            f"Invalid {TRANSPORT_ENV_VAR}={raw!r}; expected one of: {valid}"
        ) from exc


def resolve_transport_kind(env: Mapping[str, str] | None = None) -> TransportKind:
    """Resolve the transport to use, enforcing platform availability.

    Raises ``RuntimeError`` (via ``require_platform_extension_available``) if
    ``platform`` was requested but the native extension cannot be imported,
    instead of silently downgrading to ``queue``.
    """
    kind = transport_kind_from_env(env)
    if kind is TransportKind.PLATFORM:
        require_platform_extension_available()
    return kind
