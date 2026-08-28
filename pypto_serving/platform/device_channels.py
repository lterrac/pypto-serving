# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Device-payload channels: symmetric NPU windows the serving layer can hand to a kernel.

This is a **separate layer** from the host-side channels in
``platform/docs/python-channel-contract.md``. Those carry msgpack bytes between MPI ranks
through process heap memory. These carry nothing by themselves: they are device HBM windows,
symmetric across a set of chip workers, that a pypto kernel reads and writes directly. The
host is not in the data path once the window exists.

The division of labour is the whole point of this module, and it runs one way only:

* **Serving decides.** Which chip workers take part, how large the per-rank window is, which
  named buffers are carved out of it, their dtypes and byte counts. All of that arrives here
  as a fully-specified :class:`DeviceChannelRequest`.
* **Platform materialises.** :func:`open_device_channels` translates that request into the
  runtime call that allocates it, and wraps the result so the caller can read back the
  per-rank handles. It chooses nothing. There is no default window size, no sizing rule, no
  inspection of what the buffers are for, and no awareness that a model exists.

That asymmetry is deliberate. A platform that sized a window would be making a model-execution
decision, which is exactly what this layer must not do.

Lifetime
--------
A channel set is allocated from **inside an orchestration function**, because the underlying
runtime call is only valid while a run's graph is being built, and it is collective across the
participating chips. Release is two-stage and the distinction matters:

``released``
    :meth:`DeviceChannelSet.release` was called. The handle refuses further use and must not be
    passed to any new task submission. The device memory is still alive.
``freed``
    The backend window has actually been torn down. This happens after the owning ``Worker.run``
    fence, never inside the orchestration function, because tasks already submitted captured the
    window's addresses and must see live memory until they finish.

So the caller's teardown assertion belongs *after* ``Worker.run`` returns:
``channels.freed`` is True and the worker reports no live domains. See
:mod:`tests.platform.device_channel_ring.main` for that check driven end to end.

Use it as a context manager and the ``released`` mark is lexical:

.. code-block:: python

    def orch_fn(orch, args, cfg):
        with open_device_channels(orch, request) as channels:
            for worker_id in channels.workers:
                task_args.add_scalar(channels.device_ctx(worker_id))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from simpler.task_interface import ChipDomainContext

__all__ = [
    "DeviceBufferSpec",
    "DeviceChannelRequest",
    "DeviceChannelSet",
    "open_device_channels",
]


@dataclass(frozen=True)
class DeviceBufferSpec:
    """One named slice of the per-rank window, as the serving layer specifies it.

    Buffers are carved sequentially out of the window in declaration order, so a spec's
    position in :attr:`DeviceChannelRequest.buffers` determines its offset. ``nbytes`` is
    what is actually reserved; ``dtype`` and ``count`` describe how the caller intends to
    read it and are carried through unexamined.

    ``load_from_host`` / ``store_to_host`` are passed straight to the runtime's buffer spec.
    """

    name: str
    dtype: str
    count: int
    nbytes: int
    load_from_host: bool = False
    store_to_host: bool = False


@dataclass(frozen=True)
class DeviceChannelRequest:
    """A complete description of the device channels the serving layer wants.

    Every field is supplied by the caller. In particular ``window_size`` is **required** and
    is never derived here: choosing how much device memory to commit is a serving decision,
    and a platform-side default would quietly become a policy. The runtime rejects a request
    whose buffers do not fit the window, so an inconsistent request fails loudly at allocation
    rather than being silently corrected.

    ``name`` is a process-local label; participating ranks do not have to agree on it. It must
    be unique among the channel sets currently live on the owning worker.

    ``workers`` are indices into the owning ``Worker``'s ``device_ids``, and their order fixes
    the dense rank each chip holds inside this channel set.
    """

    name: str
    workers: tuple[int, ...]
    window_size: int
    buffers: tuple[DeviceBufferSpec, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # Normalise the sequence fields so a caller may pass lists. Frozen dataclass, so the
        # rewrite goes through object.__setattr__.
        object.__setattr__(self, "workers", tuple(int(worker) for worker in self.workers))
        object.__setattr__(self, "buffers", tuple(self.buffers))


class DeviceChannelSet:
    """Handle for one allocated set of device channels, with per-rank getters.

    Wraps the runtime's domain handle. The getters exist so the serving layer can retrieve
    what a kernel needs -- the device context pointer and the address of each named buffer --
    without reaching into runtime internals or learning its vocabulary.

    Not constructed directly; :func:`open_device_channels` returns it.
    """

    __slots__ = ("_request", "_handle")

    def __init__(self, request: DeviceChannelRequest, handle: Any) -> None:
        self._request = request
        self._handle = handle

    # -- identity ---------------------------------------------------------------------

    @property
    def request(self) -> DeviceChannelRequest:
        """The request this set was allocated from, unmodified."""
        return self._request

    @property
    def runtime_handle(self) -> Any:
        """The runtime's own domain handle, for a caller that must manage its lifetime.

        Ordinary callers should not need this: the getters above are the whole surface, and
        :meth:`release` is the whole teardown. It exists for the one caller that is not a
        consumer of the window but its *manager* -- ``pypto``'s ``DistributedWorker``, which
        keys its per-name reuse, its across-dispatch spec check and its release bookkeeping on
        the identity of the object ``allocate_domain`` returned. Handing that party a wrapper
        instead of the handle would silently break those checks, so the seam hands back what
        it was given. See :mod:`pypto_serving.platform.domain_factory`.
        """
        return self._handle

    @property
    def name(self) -> str:
        return self._request.name

    @property
    def workers(self) -> tuple[int, ...]:
        """Participating chip worker indices, in the order that fixed their ranks."""
        return tuple(self._handle.workers)

    @property
    def allocation_id(self) -> int:
        """The runtime's identifier for this allocation. Useful for correlating logs."""
        return int(self._handle.allocation_id)

    # -- per-rank getters -------------------------------------------------------------

    def context(self, worker_id: int) -> ChipDomainContext:
        """The full per-rank context for *worker_id*.

        This is what a pypto kernel consumes as its ``CommCtxType`` tail. Raises ``KeyError``
        for a chip that is not a member, and ``RuntimeError`` once the set is released.
        """
        return self._handle[worker_id]

    def device_ctx(self, worker_id: int) -> int:
        """Device pointer to *worker_id*'s ``CommContext``, to pass as a kernel scalar."""
        return int(self.context(worker_id).device_ctx)

    def buffer_ptr(self, worker_id: int, buffer_name: str) -> int:
        """Device address of the named buffer within *worker_id*'s window."""
        context = self.context(worker_id)
        try:
            return int(context.buffer_ptrs[buffer_name])
        except KeyError:
            available = sorted(context.buffer_ptrs)
            raise KeyError(
                f"device channel {self.name!r} has no buffer named {buffer_name!r} on worker "
                f"{worker_id}; it carries {available}"
            ) from None

    def window_base(self, worker_id: int) -> int:
        """Base device address of *worker_id*'s window."""
        return int(self.context(worker_id).local_window_base)

    def domain_rank(self, worker_id: int) -> int:
        """*worker_id*'s dense rank within this channel set."""
        return int(self.context(worker_id).domain_rank)

    def size(self) -> int:
        """Number of ranks participating."""
        return len(self.workers)

    # -- teardown ---------------------------------------------------------------------

    @property
    def released(self) -> bool:
        """True once :meth:`release` ran. Device memory may still be alive; see :attr:`freed`."""
        return bool(self._handle.released)

    @property
    def freed(self) -> bool:
        """True once the backend window is actually gone.

        Only flips after the owning ``Worker.run`` completes its fence, so an orchestration
        function never observes True for a set it released itself.
        """
        return bool(self._handle.freed)

    def release(self) -> None:
        """Mark this set for release. Idempotent.

        Inside an orchestration function this is a non-blocking mark: the backend free is
        deferred until the owning run's fence, so tasks already submitted with these addresses
        still execute against live memory.
        """
        self._handle.release()

    def __enter__(self) -> DeviceChannelSet:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()

    def __repr__(self) -> str:
        if self.freed:
            state = "freed"
        elif self.released:
            state = "released-pending-free"
        else:
            state = "live"
        return f"DeviceChannelSet(name={self.name!r}, workers={self.workers}, {state})"


def open_device_channels(orch: Any, request: DeviceChannelRequest) -> DeviceChannelSet:
    """Allocate exactly the device channels *request* describes. Purely mechanical.

    Must be called from inside an orchestration function: the underlying allocation is only
    valid while a run's graph is being built, and it is collective across ``request.workers``,
    so it blocks until every participating chip has completed the handshake.

    This function makes no decisions. It translates the request field for field, calls the
    runtime, and wraps the result. Anything the runtime rejects -- buffers that overflow the
    window, a duplicate name, a chip index out of range -- surfaces as the runtime's own
    exception, unwrapped, because the caller chose those values and is the only party that can
    fix them.

    :param orch: the ``simpler`` orchestrator handed to the orchestration function.
    :param request: what the serving layer wants allocated.
    :returns: a :class:`DeviceChannelSet` over the allocated windows.
    """
    # Imported here rather than at module scope so this module can be imported -- and its
    # request/handle types exercised -- on a host with no pypto runtime installed.
    from simpler.task_interface import CommBufferSpec

    handle = orch.allocate_domain(
        name=request.name,
        workers=list(request.workers),
        window_size=request.window_size,
        buffers=[
            CommBufferSpec(
                name=buffer.name,
                dtype=buffer.dtype,
                count=buffer.count,
                nbytes=buffer.nbytes,
                load_from_host=buffer.load_from_host,
                store_to_host=buffer.store_to_host,
            )
            for buffer in request.buffers
        ],
    )
    return DeviceChannelSet(request, handle)
