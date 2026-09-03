# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Python surface of the serving platform.

Two independent layers live here.

**Host-side channels** -- ``Deployment``, ``Input``, ``Output``, ``Platform``, ``Runtime``.
The Python process is the MPI rank; the platform runs inside it as a library, so these are
only usable under ``mpirun -np N python -m pypto_serving....``. See
``platform/docs/python-channel-contract.md``.

**Device-payload channels** -- :mod:`pypto_serving.platform.device_channels`. Symmetric NPU
windows that a pypto kernel reads and writes directly, with the host out of the data path.
Unrelated to the host-side channels above: no MPI, no HiCR, no native extension.

``_native`` is a compiled extension backing the host-side channels only. It is produced by the
platform meson project with ``-DbuildPythonBindings=true``, which stages the built ``.so`` into
this directory; it is gitignored and is not built by ``pip install``.

Its import is **deferred to first attribute access** rather than run at package import. The
device-channel layer has nothing to do with MPI and must stay importable -- and unit-testable --
in a checkout where the extension was never built. Touching ``Runtime`` and friends still raises
the same ``ImportError`` it always did, just at the point of use. That also makes
``importlib.util.find_spec("pypto_serving.platform._native")`` answer the question it was asked,
instead of raising while importing this package on its way to the submodule.
"""

from pypto_serving.platform.device_channels import (
    DeviceBufferSpec,
    DeviceChannelRequest,
    DeviceChannelSet,
    open_device_channels,
)

_NATIVE_EXPORTS = ("Deployment", "Input", "Output", "Platform", "Runtime")

__all__ = [
    "DeviceBufferSpec",
    "DeviceChannelRequest",
    "DeviceChannelSet",
    "Deployment",
    "Input",
    "Output",
    "Platform",
    "Runtime",
    "open_device_channels",
]


def __getattr__(name: str):
    """Resolve the native host-channel exports on first use (PEP 562)."""
    if name in _NATIVE_EXPORTS:
        from pypto_serving.platform import _native

        return getattr(_native, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_NATIVE_EXPORTS})
