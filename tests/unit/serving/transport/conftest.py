# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Test-collection shim so this directory's tests do not require torch.

``pypto_serving/serving/transport`` (the module under test) has no torch
dependency. But Python always executes a package's ``__init__.py`` before any
of its submodules, and the top-level ``pypto_serving/__init__.py`` eagerly
imports ``pypto_serving.config.types`` -> ``torch``, plus the model loader
and async engine. On a workstation without torch installed (this one, by
design -- see the task's environment note), a plain
``import pypto_serving.serving.transport.channel_queue`` fails inside that
unrelated chain before this package's own code ever runs.

Registering a placeholder for ``pypto_serving`` in ``sys.modules`` ahead of
collection sidesteps that: it is a namespace-package stand-in that never
executes the real ``pypto_serving/__init__.py`` body, but still resolves
``pypto_serving.serving...`` submodule imports via ``__path__`` exactly as
the real package layout would. Nothing here modifies production code, and it
only affects tests collected under this directory (pytest scopes
``conftest.py`` to its own directory and below). Any test that legitimately
needs the real ``pypto_serving`` package (e.g. because torch is actually
installed, or because ``import pypto_serving`` already ran first) is
unaffected: this only fires when ``pypto_serving`` is not already in
``sys.modules``.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

if "pypto_serving" not in sys.modules:
    _repo_root = Path(__file__).resolve().parents[4]
    _stub = types.ModuleType("pypto_serving")
    _stub.__path__ = [str(_repo_root / "pypto_serving")]
    sys.modules["pypto_serving"] = _stub
