# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Queue-shaped adapter over platform channels (Layer 2 of the channel contract).

See ``platform/docs/python-channel-contract.md`` for the full design. This
package makes a platform ``Output``/``Input`` pair usable wherever
``multiprocessing.Queue`` is used today (``pypto_serving.serving.engine.async_engine``,
``pypto_serving.serving.server.serving_worker``), plus the environment-driven
transport selection that picks between them.

Wiring this into ``async_engine.py`` / ``serving_worker.py`` is a separate
subproblem; this package is self-contained and independently testable.
"""

from __future__ import annotations

from pypto_serving.serving.transport.channel_queue import (
    DEFAULT_GET_POLL_INTERVAL_SECONDS,
    ChannelNotReadyError,
    MessageTooLargeError,
    PlatformInputQueue,
    PlatformOutputQueue,
)
from pypto_serving.serving.transport.protocols import ChannelInput, ChannelOutput
from pypto_serving.serving.transport.selection import (
    TRANSPORT_ENV_VAR,
    TransportKind,
    is_platform_extension_available,
    require_platform_extension_available,
    resolve_transport_kind,
    transport_kind_from_env,
)

__all__ = [
    "DEFAULT_GET_POLL_INTERVAL_SECONDS",
    "ChannelNotReadyError",
    "MessageTooLargeError",
    "PlatformInputQueue",
    "PlatformOutputQueue",
    "ChannelInput",
    "ChannelOutput",
    "TRANSPORT_ENV_VAR",
    "TransportKind",
    "is_platform_extension_available",
    "require_platform_extension_available",
    "resolve_transport_kind",
    "transport_kind_from_env",
]
