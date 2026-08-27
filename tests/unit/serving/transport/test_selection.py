# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for PYPTO_SERVING_TRANSPORT selection.

This workstation has neither torch nor the native platform extension, which
is exactly the condition the availability check exists to handle -- so these
tests exercise the real (not mocked) unavailable-extension path.
"""

from __future__ import annotations

import pytest

from pypto_serving.serving.transport.selection import (
    TRANSPORT_ENV_VAR,
    TransportKind,
    is_platform_extension_available,
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


def test_platform_extension_is_not_available_in_this_environment() -> None:
    # This workstation has no torch and no native extension built: the
    # availability check must report False rather than raising ImportError
    # itself, and must not have imported anything at module load time.
    assert is_platform_extension_available() is False


def test_require_platform_extension_available_raises_when_missing() -> None:
    with pytest.raises(RuntimeError, match="native"):
        require_platform_extension_available()


def test_resolve_transport_kind_defaults_to_queue() -> None:
    assert resolve_transport_kind({}) is TransportKind.QUEUE


def test_resolve_transport_kind_raises_for_platform_when_extension_missing() -> None:
    # The load-bearing behaviour: requesting `platform` without the native
    # extension must raise, never silently fall back to `queue` -- a silent
    # fallback would report a platform run that never happened.
    with pytest.raises(RuntimeError, match="PYPTO_SERVING_TRANSPORT=platform"):
        resolve_transport_kind({TRANSPORT_ENV_VAR: "platform"})


def test_resolve_transport_kind_does_not_probe_extension_for_queue() -> None:
    # Must not raise just because the extension happens to be unavailable --
    # only requesting `platform` triggers the availability check.
    assert resolve_transport_kind({TRANSPORT_ENV_VAR: "queue"}) is TransportKind.QUEUE
