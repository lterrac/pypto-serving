# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for PYPTO_SERVING_TRANSPORT selection.

Whether the native extension is actually importable is environment- and
build-dependent (it may be built in one worktree/container and not another),
so these tests do not assert on the ambient environment. Tests of the
availability-dependent behaviour monkeypatch
``selection.is_platform_extension_available`` to pin the case under test.
"""

from __future__ import annotations

import pytest

from pypto_serving.serving.transport import selection
from pypto_serving.serving.transport.selection import (
    TRANSPORT_ENV_VAR,
    TransportKind,
    require_platform_extension_available,
    resolve_transport_kind,
    transport_kind_from_env,
)


def test_transport_env_var_name_is_stable() -> None:
    assert TRANSPORT_ENV_VAR == "PYPTO_SERVING_TRANSPORT"


def test_default_transport_is_queue_when_env_absent() -> None:
    assert transport_kind_from_env({}) is TransportKind.QUEUE


def test_default_transport_is_queue_via_real_os_environ(monkeypatch) -> None:
    monkeypatch.delenv(TRANSPORT_ENV_VAR, raising=False)
    assert transport_kind_from_env() is TransportKind.QUEUE


def test_explicit_queue_override() -> None:
    assert transport_kind_from_env({TRANSPORT_ENV_VAR: "queue"}) is TransportKind.QUEUE


def test_explicit_platform_override() -> None:
    assert transport_kind_from_env({TRANSPORT_ENV_VAR: "platform"}) is TransportKind.PLATFORM


def test_env_value_is_trimmed_and_case_insensitive() -> None:
    assert transport_kind_from_env({TRANSPORT_ENV_VAR: "  PLATFORM  "}) is TransportKind.PLATFORM


def test_invalid_env_value_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Invalid"):
        transport_kind_from_env({TRANSPORT_ENV_VAR: "raw_pipe"})


def test_require_platform_extension_available_raises_when_extension_missing(monkeypatch) -> None:
    monkeypatch.setattr(selection, "is_platform_extension_available", lambda: False)
    with pytest.raises(RuntimeError, match="native"):
        require_platform_extension_available()


def test_require_platform_extension_available_is_a_noop_when_extension_present(monkeypatch) -> None:
    monkeypatch.setattr(selection, "is_platform_extension_available", lambda: True)
    require_platform_extension_available()  # must not raise


def test_resolve_transport_kind_defaults_to_queue() -> None:
    assert resolve_transport_kind({}) is TransportKind.QUEUE


def test_resolve_transport_kind_raises_for_platform_when_extension_missing(monkeypatch) -> None:
    # The load-bearing behaviour: requesting `platform` without the native
    # extension must raise, never silently fall back to `queue` -- a silent
    # fallback would report a platform run that never happened.
    monkeypatch.setattr(selection, "is_platform_extension_available", lambda: False)
    with pytest.raises(RuntimeError, match="PYPTO_SERVING_TRANSPORT=platform"):
        resolve_transport_kind({TRANSPORT_ENV_VAR: "platform"})


def test_resolve_transport_kind_returns_platform_when_extension_present(monkeypatch) -> None:
    monkeypatch.setattr(selection, "is_platform_extension_available", lambda: True)
    assert resolve_transport_kind({TRANSPORT_ENV_VAR: "platform"}) is TransportKind.PLATFORM


def test_resolve_transport_kind_does_not_probe_extension_for_queue(monkeypatch) -> None:
    # Must not raise (or even consult availability) just because the
    # extension happens to be unavailable -- only requesting `platform`
    # triggers the availability check.
    def _boom() -> bool:
        raise AssertionError("availability must not be probed for the queue transport")

    monkeypatch.setattr(selection, "is_platform_extension_available", _boom)
    assert resolve_transport_kind({TRANSPORT_ENV_VAR: "queue"}) is TransportKind.QUEUE
