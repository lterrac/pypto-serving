# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Structural types for platform channel endpoints (Layer 1 of the contract).

These ``Protocol`` classes mirror ``Output``/``Input`` from
``platform/docs/python-channel-contract.md`` exactly. They exist so this
package can be written and tested against the *shape* of the native
extension without importing it: ``pypto_serving.platform._native`` does not
exist yet (it is being built in parallel against the same contract), and
this module must stay importable -- without torch, without pypto, without
the extension -- on a plain Python environment.

Anything duck-typed to this shape works here: the real extension once it
lands, or a plain in-memory test double (see
``tests/unit/serving/transport``).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ChannelOutput(Protocol):
    """Structural match for ``platform._native.Output``."""

    def is_full(self, message_size: int) -> bool:
        """Return whether a push of ``message_size`` bytes would not fit."""
        ...

    def push(
        self,
        payload: bytes,
        message_type: int = 0,
        group_id: int = 0,
        sequence_id: int = 0,
    ) -> None:
        """Push ``payload`` (``pushMessageLocking``: waits until there is room)."""
        ...

    def is_ready(self) -> bool:
        """Return whether the channel has completed its handshake."""
        ...


@runtime_checkable
class ChannelInput(Protocol):
    """Structural match for ``platform._native.Input``."""

    def has_message(self) -> bool:
        """Return whether a message is currently available to read."""
        ...

    def read(self) -> bytes:
        """Copy out and pop the next message (``getMessage`` + copy + ``popMessage``)."""
        ...

    def is_ready(self) -> bool:
        """Return whether the channel has completed its handshake."""
        ...
