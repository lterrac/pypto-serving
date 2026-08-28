# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Which launch path ``pypto_serving.cli.main`` takes, and when."""

from __future__ import annotations

import argparse

import pytest

from pypto_serving.cli import main as cli_main


@pytest.fixture
def stub_config(monkeypatch):
    """Skip config building (it reads a model directory) but keep main()'s flow."""
    config = object()
    monkeypatch.setattr(cli_main, "build_serving_engine_config", lambda args: config)
    monkeypatch.setattr(cli_main, "_warn_deprecated_serving_profile_env", lambda args: None)
    monkeypatch.setattr(
        cli_main,
        "build_parser",
        lambda: _parser_returning(
            argparse.Namespace(
            host="0.0.0.0",
            port=8000,
            show_startup_logs=True,
            # main() builds a GenerateConfig from this and passes it to run_serve;
            # prompt selects the one-shot generate path instead of serving.
            generate_config=None,
            prompt=None,
        )
        ),
    )
    return config


def _parser_returning(namespace):
    class _Parser:
        def parse_args(self, argv):
            return namespace

    return _Parser()


def test_unset_transport_goes_straight_to_run_serve(monkeypatch, stub_config):
    """The default path must not touch the platform launcher at all."""
    monkeypatch.delenv("PYPTO_SERVING_TRANSPORT", raising=False)
    served: list[tuple] = []

    def _serve(config, generate_config=None, **kwargs) -> int:
        served.append((config, kwargs))
        return 0

    monkeypatch.setattr(cli_main, "run_serve", _serve)

    assert cli_main.main([]) == 0
    assert served == [(stub_config, {"host": "0.0.0.0", "port": 8000})]


def test_platform_transport_is_refused_when_the_extension_is_missing(monkeypatch, stub_config):
    """Never a silent downgrade: a platform run that never happened must not be
    reported as one."""
    monkeypatch.setenv("PYPTO_SERVING_TRANSPORT", "platform")
    monkeypatch.setattr(
        "pypto_serving.serving.transport.selection.is_platform_extension_available",
        lambda: False,
    )
    monkeypatch.setattr(
        cli_main, "run_serve", lambda *a, **k: pytest.fail("must not serve")
    )

    with pytest.raises(RuntimeError, match="Refusing to silently fall back"):
        cli_main.main([])


def test_an_unknown_transport_name_is_rejected(monkeypatch, stub_config):
    monkeypatch.setenv("PYPTO_SERVING_TRANSPORT", "sockets")
    monkeypatch.setattr(
        cli_main, "run_serve", lambda *a, **k: pytest.fail("must not serve")
    )

    with pytest.raises(ValueError, match="Invalid PYPTO_SERVING_TRANSPORT"):
        cli_main.main([])


def test_main_propagates_a_failing_exit_code(monkeypatch, stub_config):
    """A replica whose engine loop died must not exit 0: under systemd or
    Kubernetes a clean exit means no restart, no backoff and no alert."""
    monkeypatch.delenv("PYPTO_SERVING_TRANSPORT", raising=False)
    monkeypatch.setattr(cli_main, "run_serve", lambda config, generate_config, **kwargs: 1)
    assert cli_main.main([]) == 1
