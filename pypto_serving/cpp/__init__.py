# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The C++ serving engine from Python.

``python -m pypto_serving.cpp --model DIR`` serves; ``--prompt`` generates
offline. :class:`Runtime` is the same assembly as an object.

``_pypto_serving`` must be importable: put the ``serving/`` build directory, or
its install prefix, on ``PYTHONPATH``.
"""

from pypto_serving.cpp.executor import PyptoModelExecutor
from pypto_serving.cpp.runtime import GenerationResult, Runtime, ServingOptions

__all__ = ["GenerationResult", "PyptoModelExecutor", "Runtime", "ServingOptions"]
