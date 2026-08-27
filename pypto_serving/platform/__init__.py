# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Python surface of the serving platform: the MPI/HiCR runtime and its channels.

The Python process is the MPI rank; the platform runs inside it as a library, so this
package is only usable under ``mpirun -np N python -m pypto_serving....``. See
``platform/docs/python-channel-contract.md``.

``_native`` is a compiled extension. It is produced by the platform meson project with
``-DbuildPythonBindings=true``, which stages the built ``.so`` into this directory; it is
gitignored and is not built by ``pip install``. If the import below fails, that build step
has not been run.
"""

from pypto_serving.platform._native import Deployment, Input, Output, Platform, Runtime

__all__ = [
    "Deployment",
    "Input",
    "Output",
    "Platform",
    "Runtime",
]
