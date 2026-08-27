# The Python channel contract

Work-in-progress design note for making `platform/` create the channels that the Python
serving layer consumes. It fixes the interface so the pieces can be built in parallel.

## Why the Python process must be the MPI rank

A channel here is not a handle. `Input`/`Output` are built over nine globally-exchanged
HiCR memory slots, and with the MPI backend those slots are **MPI RMA windows over ordinary
process heap memory** (`MPI_Alloc_mem` + `MPI_Win_allocate`, transferred with `MPI_Put`/
`MPI_Get`). There is no file descriptor, no shared-memory segment, nothing with an OS-level
handle that could be inherited or sent over a socket.

Membership is established by **participating in the collective**
`exchangeGlobalMemorySlots(tag, slots)` + `fence(tag)` that `channelController::reconcile()`
performs (`platform/include/modules/channelController/module.hpp:170-171`). A process that is
not a rank in the communicator cannot join it.

And the platform cannot create those ranks itself: `Engine::createInstance()` forwards to
`HiCR::InstanceManager::createInstance()`, whose MPI backend does not override the default and
throws `HICR_THROW_LOGIC("This backend does not currently support the launching of new
instances during runtime")`.

Both facts point the same way, and it is the only shape that works:

    mpirun -np N python -m pypto_serving.cli ...

The platform runs **inside** the Python process as a library. Python is the rank. `platform`
initializes, creates the channels, and hands them to the serving layer.

This is the opposite of "a platform binary launches Python children", which cannot work here.

## Layer 1 — the extension module

Native module, importable as `pypto_serving.platform._native`. Names are snake_case on the
Python side; the C++ they wrap is named in each entry.

    class Runtime:
        """MPI + HiCR bootstrap; mirrors examples/include/runtime/helpers.hpp:makeRuntime."""
        def __init__(self, compute_resource_count: int = 2) -> None: ...
        @property
        def instance_id(self) -> int: ...
        @property
        def is_root(self) -> bool: ...
        @property
        def instance_count(self) -> int: ...
        def finalize(self) -> None: ...          # InstanceManager::finalize

    class Deployment:
        @staticmethod
        def from_json_file(path: str) -> "Deployment": ...
        def assign_edge_managers(self, runtime: Runtime) -> None: ...
        def assign_instances(self, runtime: Runtime) -> None: ...   # readAndParseConfiguration's rank->partition step
        def edge_names(self) -> list[str]: ...

    class Input:
        def has_message(self) -> bool: ...
        def read(self) -> bytes: ...     # getMessage + COPY + popMessage, as one atomic step
        def is_ready(self) -> bool: ...

    class Output:
        def is_full(self, message_size: int) -> bool: ...
        def push(self, payload: bytes, message_type: int = 0,
                 group_id: int = 0, sequence_id: int = 0) -> None: ...   # pushMessageLocking
        def is_ready(self) -> bool: ...

    class Platform:
        """Engine + channelController + service, wired as examples/modules/channelController does."""
        def __init__(self, runtime: Runtime, deployment: Deployment) -> None: ...
        def open_output(self, edge_name: str) -> Output: ...
        def open_input(self, edge_name: str) -> Input: ...
        def start(self) -> None: ...     # addModule, initialize, run  -- channels become ready here
        def wait_until_ready(self, timeout_s: float = 30.0) -> None: ...
        def stop(self) -> None: ...      # terminate (root only) + await

### Two rules the binding must obey, both load-bearing

1. **`Input.read()` copies.** `Message::getData()` returns a pointer *into the consumer ring
   buffer*, valid only until `popMessage()`. Exposing it as a Python buffer would hand out a
   dangling view. `read()` therefore copies into `bytes` and pops, in one call.
2. **Release the GIL around anything that blocks.** `Output::pushMessageLocking` spins with a
   1 us sleep until the ring has room, and `reconcile()` runs a collective. Holding the GIL
   there deadlocks the process against its own asyncio loop.

## Layer 2 — the queue-shaped adapter

`ReplicaEngineCore` already moves opaque msgpack bytes and never inspects them
(`serving/server/ipc.py`: "Switching the queue to a raw Pipe is a drop-in swap at the
encode_command / decode_command call sites"). So the adapter only has to satisfy the surface
the call sites actually use:

    put(payload: bytes) -> None          # async_engine.py:509, :709, :315, :883 -- called
                                         # SYNCHRONOUSLY on the asyncio loop, must not block
    get(timeout: float | None) -> bytes  # async_engine.py:575, :317 -- always inside
                                         # asyncio.to_thread; MUST raise queue.Empty on timeout
                                         # (caught at :321, :531, :714)

FIFO with exactly one result per command is assumed structurally: a `step_id` mismatch is
treated as fatal (`async_engine.py:539-547`). An SPSC channel gives that by construction.

`ready_event` and `num_pages_value` stay on their current mechanism for now. They are one-shot
startup handshakes, consumed only in `start()` and never stored on the object
(`async_engine.py:245`, `:259`). `num_pages_value` in particular is a device-derived quantity
that sizes a model-support structure; carrying it semantically through the platform is how the
platform turns into another model-execution abstraction layer, which issue #32 forbids.

## Layer 3 — the topology

One edge is one unidirectional SPSC channel between two partitions' coordinator instances,
identified by its index in the `Edges` array. For a single replica:

    partition "engine"  -- task with output "commands", input "results"
    partition "worker"  -- task with input  "commands", output "results"
    edge "commands": producer "engine", consumer "worker"
    edge "results":  producer "worker", consumer "engine"

`Deployment::verify()` requires every edge to be used exactly once as an input and once as an
output, producer partition != consumer partition, and every task to have at least one input
and one output. The two-edge cycle above satisfies all of it.

Rank assignment follows `readAndParseConfiguration`: partitions take instances in declaration
order, so rank 0 is `engine` and rank 1 is `worker`.

## What is deliberately not in scope

Moving per-step token traffic onto device-side channels; heartbeat; multiple replicas;
dynamic scale-up. The first three are deferred modules that do not exist in this tree; the
last one the MPI backend cannot do at all.
