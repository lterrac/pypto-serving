# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Make the platform the *creator* of the comm windows a compiled model uses.

``pypto_serving.platform.device_channels`` already lets serving open a device channel it
asked for by hand. That covers windows serving invents. It does not cover the windows a
compiled model needs, which are not serving's idea at all: the pypto compiler reads them off
the model's ``@pl.jit.host`` orchestration and emits ``orch.allocate_domain(...)`` calls into
generated Python. Those are the windows DeepSeek's MoE dispatch actually moves tokens
through, and until now the platform had nothing to do with them.

``DistributedWorker(..., domain_factory=...)`` -- the pypto-side hook this module exists to
fill -- moves exactly one step of that chain and no more::

    model's @pl.jit.host orch function   USES the windows      (unchanged)
    pypto's internal domain provider     MANAGES the windows   (unchanged)
    this factory                         CREATES the windows   (was orch.allocate_domain)

The three-way split is the point. The **model** still declares what it needs, in its own
source, compiled the same way. **pypto** still decides when a domain is created, reuses it by
name across dispatches, refuses a specification that changes between runs, and releases it at
close -- that bookkeeping is real and stays where it is. What moves here is the single act of
materialising the memory, which is a resource decision, and resources are the platform's.

What this module must NOT become
--------------------------------
A place where the window is reinterpreted. The factory does not resize, rename, pad, reorder,
cache or substitute anything: it receives the compiler's parameters, restates them as a
:class:`~pypto_serving.platform.device_channels.DeviceChannelRequest`, and allocates that.
A platform that adjusted a model's window would be making a model-execution decision, which
is what the platform layer must not do (hw-native-sys/pypto-serving#32).

Why it hands back the raw runtime handle
----------------------------------------
:func:`~pypto_serving.platform.device_channels.open_device_channels` returns a
:class:`~pypto_serving.platform.device_channels.DeviceChannelSet` wrapper, which is right for
a caller that *uses* a window. pypto is not that caller: it is the manager, and its per-name
reuse, spec-immutability check and release bookkeeping are keyed on the identity of the object
``allocate_domain`` returned. So the factory returns ``channels.runtime_handle`` and lets the
wrapper go: what it keeps for its own observability is a value-copied
:class:`DomainCreation`, never the handle.

Usage::

    factory = PlatformDomainFactory()
    worker = DistributedWorker(compiled, persistent=True, domain_factory=factory)
    ...
    for record in factory.records:
        print(record.name, record.workers, record.window_size)
"""

from __future__ import annotations

import inspect
import logging
import os
from dataclasses import dataclass
from typing import Any

from pypto_serving.platform.device_channels import (
    DeviceBufferSpec,
    DeviceChannelRequest,
    open_device_channels,
)

__all__ = [
    "DomainCreation",
    "PlatformDomainFactory",
    "RankWindow",
    "platform_domain_factory",
]

#: Set to ``0``/``false``/``no``/``off`` to leave comm-window creation with the runtime.
ENABLE_ENV_VAR = "PYPTO_SERVING_PLATFORM_COMM_WINDOWS"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RankWindow:
    """Where one participating chip's window landed, as read at creation time."""

    worker_id: int
    domain_rank: int
    device_ctx: int
    window_base: int

    def __str__(self) -> str:
        return (
            f"worker {self.worker_id}: rank={self.domain_rank} "
            f"device_ctx=0x{self.device_ctx:x} window_base=0x{self.window_base:x}"
        )


@dataclass(frozen=True)
class DomainCreation:
    """One window this factory materialised, as it was asked for.

    Kept so a caller can prove *which* party allocated a program's windows -- otherwise the
    only evidence that the seam is live is that nothing broke, which is equally consistent
    with the factory never having been called.

    A record is **pure data**: it holds no reference to the channel set or the runtime handle,
    only values copied out of them at creation. Two reasons, and both bite.

    *Lifetime.* pypto releases every retained domain when the worker closes, and a released
    handle refuses indexing -- so a record that read addresses back on demand would raise
    ``RuntimeError`` at exactly the moment a caller wants the evidence.

    *Ownership.* The handle is a nanobind object owned by the runtime, and a serving replica
    calls ``gc.freeze()`` on its long-lived state. A factory reachable from the model runner
    would therefore hold those instances past interpreter shutdown and the runtime would
    report them leaked -- observed as ``nanobind: leaked 4975 instances`` on a DeepSeek V4
    run that kept them, and absent from the same run once the record became a value copy.
    """

    name: str
    workers: tuple[int, ...]
    window_size: int
    buffers: tuple[tuple[str, str, int, int], ...]
    ranks: tuple[RankWindow, ...]
    allocation_id: int

    def __str__(self) -> str:
        buffers = ", ".join(f"{name}({nbytes}B)" for name, _dtype, _count, nbytes in self.buffers)
        return (
            f"{self.name}: workers={list(self.workers)} window_size={self.window_size} "
            f"allocation_id={self.allocation_id} buffers=[{buffers}]"
        )


class PlatformDomainFactory:
    """The platform, standing in for ``orch.allocate_domain``.

    Pass an instance as ``DistributedWorker(..., domain_factory=...)``. pypto then calls it
    once per CommDomain it needs, with the run's orchestrator and exactly the keyword
    arguments it would have passed to the runtime, and takes ownership of what comes back.

    Instances are reusable across dispatches and across programs; :attr:`records` accumulates
    one :class:`DomainCreation` per window actually created (pypto reuses a retained window
    rather than asking again, so a persistent program yields one record per domain, not one
    per dispatch).
    """

    __slots__ = ("_records", "_log_level")

    def __init__(self, *, log_level: int = logging.INFO) -> None:
        self._records: list[DomainCreation] = []
        self._log_level = log_level

    @property
    def records(self) -> tuple[DomainCreation, ...]:
        """Every window this factory created, in creation order."""
        return tuple(self._records)

    def __call__(
        self,
        orch: Any,
        *,
        name: str,
        workers: Any,
        window_size: int,
        buffers: Any = (),
    ) -> Any:
        """Create the window pypto asked for and hand the runtime handle back.

        The signature is pypto's ``allocate_domain`` signature with the orchestrator moved to
        the front, so a mismatch between what the compiler emits and what the platform accepts
        is a ``TypeError`` here rather than a silently dropped field.
        """
        request = DeviceChannelRequest(
            name=name,
            workers=tuple(int(worker) for worker in workers),
            window_size=int(window_size),
            buffers=tuple(
                DeviceBufferSpec(
                    name=buffer.name,
                    dtype=buffer.dtype,
                    count=int(buffer.count),
                    nbytes=int(buffer.nbytes),
                    load_from_host=bool(getattr(buffer, "load_from_host", False)),
                    store_to_host=bool(getattr(buffer, "store_to_host", False)),
                )
                for buffer in buffers
            ),
        )
        channels = open_device_channels(orch, request)
        # Everything the record keeps is read here, while the set is live, and copied by
        # value: see DomainCreation on why it must not hold the set itself.
        record = DomainCreation(
            name=request.name,
            workers=request.workers,
            window_size=request.window_size,
            buffers=tuple(
                (buffer.name, buffer.dtype, buffer.count, buffer.nbytes) for buffer in request.buffers
            ),
            ranks=tuple(
                RankWindow(
                    worker_id=worker_id,
                    domain_rank=channels.domain_rank(worker_id),
                    device_ctx=channels.device_ctx(worker_id),
                    window_base=channels.window_base(worker_id),
                )
                for worker_id in channels.workers
            ),
            allocation_id=channels.allocation_id,
        )
        self._records.append(record)
        logger.log(self._log_level, "platform created comm window %s", record)
        # The manager's handle, not our wrapper -- see the module docstring.
        return channels.runtime_handle


def platform_domain_factory() -> PlatformDomainFactory | None:
    """The factory to hand ``DistributedWorker``, or ``None`` to leave the runtime allocating.

    Returns ``None`` -- with a log line saying so, never silently -- in two cases:

    * ``PYPTO_SERVING_PLATFORM_COMM_WINDOWS`` is switched off, which is how a run is put back
      on the runtime's own allocator to tell a platform-side fault from a model-side one;
    * the installed pypto has no ``domain_factory`` parameter. The hook is a PyPTO change that
      serving does not control, so a serving that hard-required it would simply fail to start
      against a stock runtime. Capability is read off the constructor's own signature rather
      than a version number, because the signature is the thing that has to be true.

    The caller therefore spreads the result rather than passing it positionally::

        kwargs = {} if factory is None else {"domain_factory": factory}
    """
    if os.environ.get(ENABLE_ENV_VAR, "1").strip().lower() in {"0", "false", "no", "off"}:
        logger.info(
            "%s is off; the runtime will create this program's comm windows", ENABLE_ENV_VAR
        )
        return None
    from pypto.runtime import DistributedWorker  # noqa: PLC0415

    if "domain_factory" not in inspect.signature(DistributedWorker.__init__).parameters:
        logger.info(
            "installed pypto has no DistributedWorker(domain_factory=...) hook; "
            "the runtime will create this program's comm windows"
        )
        return None
    return PlatformDomainFactory()
