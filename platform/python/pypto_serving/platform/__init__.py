# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Python surface of the serving platform: the MPI/HiCR runtime and its channels.

The Python process is the MPI rank; the platform runs inside it as a library. Launch
with ``mpirun -np N python -m ...``. See ``platform/docs/python-channel-contract.md``.

The native extension is built by the platform meson project with
``-DbuildPythonBindings=true``; it is not built by ``pip install``.
"""

from pypto_serving.platform._native import Deployment, Input, Output, Platform, Runtime

__all__ = [
    "Deployment",
    "Input",
    "Output",
    "Platform",
    "Runtime",
]
