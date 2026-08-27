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
        def finalize(self) -> None: ...          # InstanceManager::finalize; stops live Platforms first
        def abort(self, exit_code: int = -1) -> None: ...  # InstanceManager::abort -- does not return
        @property
        def is_finalized(self) -> bool: ...
        def __enter__(self) -> "Runtime": ...
        def __exit__(self, *exc) -> bool: ...     # finalize

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
        edge_name: str
        buffer_size: int        # the edge's payload ring, in bytes
        max_message_size: int   # == buffer_size; the largest payload this edge can carry
        capacity: int           # the edge's token capacity

    class Output:
        def is_full(self, message_size: int) -> bool: ...
        def push(self, payload: bytes, message_type: int = 0,
                 group_id: int = 0, sequence_id: int = 0) -> None: ...   # pushMessageLocking
        def is_ready(self) -> bool: ...
        edge_name: str
        buffer_size: int
        max_message_size: int   # push() raises ValueError above this rather than spinning forever
        capacity: int

    class Platform:
        """Engine + channelController + service, wired as examples/modules/channelController does."""
        def __init__(self, runtime: Runtime, deployment: Deployment) -> None: ...
        def open_output(self, edge_name: str) -> Output: ...
        def open_input(self, edge_name: str) -> Input: ...
        def start(self) -> None: ...     # addModule, initialize, run  -- channels become ready here
        def wait_until_ready(self, timeout_s: float = 30.0) -> None: ...
        def stop(self) -> None: ...      # terminate (root only) + await; idempotent, also valid before start
        @property
        def is_stopped(self) -> bool: ...
        def __enter__(self) -> "Platform": ...
        def __exit__(self, *exc) -> bool: ...     # stop

### Two rules the binding must obey, both load-bearing

1. **`Input.read()` copies.** `Message::getData()` returns a pointer *into the consumer ring
   buffer*, valid only until `popMessage()`. Exposing it as a Python buffer would hand out a
   dangling view. `read()` therefore copies into `bytes` and pops, in one call.
2. **Release the GIL around anything that blocks *or that a caller may poll*.**
   `Output::pushMessageLocking` spins with a 1 us sleep until the ring has room, and
   `reconcile()` runs a collective. Holding the GIL there deadlocks the process against its
   own asyncio loop.

   "Blocks" is not the whole rule. `Input::hasMessage()` and `Output::isFull()` each drive two
   `updateDepth()` calls -- MPI RMA progress, ~4.5 us apiece -- and both are *polled*: the
   Layer 2 adapter calls `has_message()` ~66 times per 36 ms serving step from an
   `asyncio.to_thread` worker, and calls `is_full()` on the event-loop thread on every `put`.
   A native method that never drops the GIL cannot be preempted mid-call, so a poll loop over
   one pins the interpreter for whole `sys.setswitchinterval` quanta. Measured on hg-atlas-01,
   a competing thread's 1 ms wakeup lands at p50 **10.0 ms** late when the polled method holds
   the GIL, against **2.9 ms** when it releases it and **0.06 ms** with no native calls at all.
   The release pays for itself: it adds under 1 us to a ~5 us call and cuts the competing
   thread's p50 wakeup latency by ~3.5x.

   So the per-method rule is:

   * release -- `Runtime()`, `Runtime.finalize`, `Platform.start`, `Platform.stop`,
     `Platform.wait_until_ready`, `Output.push`, `Output.is_full`, `Input.has_message`;
   * hold -- `Input.read` (it must build the `bytes` object, and the copy needs the GIL) and
     `is_ready` (a plain atomic load; the release would cost more than the call).

   Never touch the Python C-API -- including refcounts, `py::bytes` construction and raising --
   inside a released region.

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

With a worker *rank* rather than a worker *process* there is no shared memory to keep them in,
so they ride the `results` edge as one **opaque msgpack payload** -- bytes the platform cannot
tell apart from a `StepResult` (`serving/transport/handshake.py`). That keeps `num_pages` out of
the platform's semantic surface, as #32 requires. It is safe to overload that edge because the
message is strictly ordered outside the request stream: the worker pushes it as the first thing
it ever puts on `results`, before its busy loop starts, and the engine consumes it inside
`start()`, before `_engine_loop` exists. One object (`PlatformStartupHandshake`) satisfies both
`ready_event.wait(timeout=...) -> bool` and `num_pages_value.value -> int`, so `start()`'s
six-tuple unpacking is unchanged and the queue path keeps exactly the code it had.

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

`serving/transport/platform_launch.py` generates exactly this policy (rather than shipping it
as a file) so the edge sizing can follow one rule: **`Buffer Size` = `Buffer Capacity` x the
largest message the adapter will accept.** That is what makes a producer's "is there room?"
question answerable *before* it knows the payload -- and the engine has to ask it before
`scheduler.schedule()` runs, because `schedule()` allocates KV blocks and promotes requests and
cannot be rolled back if the dispatch is then refused. With a full worst-case message's worth of
ring free, whatever the step turns out to be will fit. Note that `Output::isFull` is
two-dimensional -- a metadata slot count *and* the payload ring's free bytes -- so the rule has
to hold in bytes, not just in slots.

Two sizing constraints follow, and both are enforced rather than documented:

* **`Buffer Capacity` >= `ReplicaEngineCore._max_in_flight`** (2 under async scheduling). At
  capacity 1 the engine can hold the only slot while the worker is inside a blocking push on
  `results`, and the only thing that would free the worker is the engine reading -- which it
  has stopped doing. That is a hang no signal can break, because the native push spins with
  the GIL released.
* **`Buffer Size` must cover a real worst-case `StepCommand`.** ipc.py's "~1 KB steady state"
  counts only the per-request deltas; `_build_step_command` also ships `block_ids` *and*
  `block_ids_by_group`, and DeepSeek V4 declares six cache groups, so the block tables are sent
  seven times over. Measured, at page_size 16: 0.74 MiB for 32 requests x 8192 blocks on a
  single generic pool, **5.17 MiB** for the same batch with DeepSeek V4's groups, and 21.16 MiB
  if 32 fresh 128K-token prompts are admitted in one step. The default per-message limit is
  16 MiB. A step that still does not fit fails *its requests* through `_handle_step_error`, not
  the replica.

Profiling has no place in this topology. `_set_profile_active` needs a third worker->engine
channel for its acknowledgement; multiplexing acks onto `results` would break the engine's
one-result-per-command invariant (`async_engine.py:539-547`), so the platform transport raises
`ProfilingUnsupportedError`, which the HTTP layer answers with 400.

Two lifecycle rules the launcher owns, because nothing else can:

* **The worker rank must be told to stop, and the launcher checks.** A worker blocked in
  `get(timeout=None)` looks identical whether it is idle or abandoned, so `_run_role` refuses to
  return normally unless a `ShutdownCommand` actually reached the channel. Any path where
  `run_serve` *returns* rather than raises without its lifespan shutdown having run -- a uvicorn
  startup-event failure being the obvious one -- would otherwise leave rank 1 blocked forever.
* **A dead engine loop must not exit 0.** `_engine_loop` runs under a guard that logs, fails
  the in-flight requests and SIGTERMs the process; `run_serve` then returns 1 and
  `run_platform_replica` propagates it. Without that the process exits cleanly and no supervisor
  restarts it.

## What is deliberately not in scope

Moving per-step token traffic onto device-side channels; heartbeat; multiple replicas;
dynamic scale-up. The first three are deferred modules that do not exist in this tree; the
last one the MPI backend cannot do at all.

## Building and running the extension

The extension is opt-in; the default `platform-build` configuration does not build it.

```bash
meson setup platform/build platform/ -Dengines=mpi -DbuildTests=true -DbuildPythonBindings=true
ninja -C platform/build
meson test -C platform/build --suite examples
```

`-DpythonInterpreter=<name-or-path>` selects the interpreter to build against (default
`python3`); its development headers must be installed. pybind11 is used for the binding
because it is already vendored in this tree, at
`platform/extern/TaskR/extern/tracr/extern/pybind11`.

`platform/examples/python/roundTrip.py` is the worked example: `mpirun -np 2` over a
two-partition, two-edge policy, rank 0 = `engine`, rank 1 = `worker`.
`tests/platform/two_rank_replica.py` is the same shape carrying a real `ReplicaEngineCore` and a
real serving worker busy loop over those channels (`pytest tests/platform -q`, or under `mpirun`
directly); only `init_device_and_model`, `_execute_step` and the tokenizer are stubbed, because
nothing there may touch an NPU.
`platform/examples/python/failureModes.py` is the counter-example: it drives every misuse
below under mpirun and asserts each one fails loudly rather than fatally.

### Where the `.so` has to live

`ninja` stages the built extension into `pypto_serving/platform/_native.<abi>.so` in the
source tree (gitignored; declared as setuptools package data). That is not a convenience --
it is the only layout that works. `pypto_serving` has an `__init__.py`, so it is a *regular*
package, and a regular package always wins over an implicit namespace portion no matter how
`sys.path` is ordered. A build-tree mirror at `<builddir>/python/pypto_serving/platform/` is
therefore unimportable the moment the repository itself is importable -- which is exactly the
shape the contract mandates, `mpirun ... python -m pypto_serving.cli` from the repository
root. An editable install resolves to the same directory, so staging *is* the install.

The cost of that layout: importing `pypto_serving.platform` executes
`pypto_serving/__init__.py`, which pulls in torch and the model loader. Measured on
hg-atlas-01, `import pypto_serving.platform` takes ~8.5 s, of which the extension itself is
~6 ms. Inside the real serving process this is free, because torch is imported anyway.

## What the binding enforces, and why it has to

Everything in this section is a rule the object model already had; the binding turns each
one from a segmentation fault or a silent hang into an exception.

* **Channels are opened before `start()`, and the set is closed there.**
  `channelController::reconcile()` only enters `exchangeGlobalMemorySlots`/`fence` when *it*
  has channels to create, and those calls are collective over the whole communicator.
  Opening an edge on one rank after the others have moved on would hang them.

* **`start()` agrees the channel *claim set* across ranks before committing to the
  collective.** Counting is not enough. A rank that opened nothing skips a collective its
  peers are in and the job wedges; a rank that opened a *different set* of edges passes any
  count-based check, enters the exchange, and then asks for a global key nobody registered --
  a segmentation fault, not an exception. Since every rank walks the same edge list in the
  same order, one allreduce over a producer/consumer claim vector settles both: an edge must
  be opened by exactly one producer and exactly one consumer, or by neither end. Skipping an
  edge is legitimate as long as both ends skip it, which is what lets the serving layer open
  only the edges it needs. `Platform.__init__` separately rejects a deployment that has not
  had both `assign_edge_managers()` and `assign_instances()` called on it, since skipping the
  latter leaves every partition on coordinator id 0 and produces the same asymmetry.

* **Teardown is ordered, and the binding orders it.** The channels are MPI RMA windows and
  the channel controller runs on TaskR fibers. Unwinding into `~Engine` while a service
  worker is still inside `reconcile()` is a segmentation fault, and freeing a channel after
  `MPI_Finalize` is undefined. So: `Platform.stop()` is idempotent and valid before `start()`
  (opening a channel already allocated memory slots); `Platform` and `Runtime` are both
  context managers; both call `stop()`/`finalize()` from their destructors as a last resort;
  and `Runtime.finalize()` stops any live `Platform` before `MPI_Finalize` rather than
  assuming the caller did. `Input`/`Output` hold their channel weakly, so a handle that
  outlives `stop()` raises instead of freeing MPI memory late.

* **A rank that fails alone must abort the job.** Its peers are otherwise blocked in
  `Engine::await()` waiting for a STOP RPC, in the claim agreement's allreduce, or in the
  collective `MPI_Finalize`. `Runtime.abort(code)` is `InstanceManager::abort`; `roundTrip.py`
  calls it from its exception handler, and that is the pattern to copy.

  `with Runtime() as runtime:` alone is *not* that pattern and is not sufficient on its own.
  It does abort rather than finalize when the block exits on an exception and a `Platform`
  was created -- because finalizing there parks the peers -- but it cannot see a failure you
  catch yourself, and it cannot help before the first `Platform` exists. Keep the explicit
  `except Exception: runtime.abort(1)` around anything a single rank can fail at.

* **`push()` rejects a payload that can never fit.** `pushMessageLocking` spins while
  `isFull` is true, and for `len(payload) > max_message_size` that is true forever -- with the
  GIL released, so not even SIGINT gets the caller out of it. Above `max_message_size`,
  `push()` raises `ValueError`.

* **`push()` takes `bytes` only.** `bytearray` and `memoryview` raise `TypeError`. The
  producer reads the buffer after the GIL is released, so it must not be something the caller
  can mutate underneath it.

* **`is_full()` and `push()` on one edge must run on the same thread.** Not merely one
  handle per edge per thread: `Output::pushMessageLocking` holds the channel mutex across its
  own `isFull` check and re-takes it every microsecond while it spins, and `std::mutex` is not
  fair. In Layer 2, `put()` calls `is_full()` on the asyncio event-loop thread; if a worker
  thread is spinning inside `push()` on that same edge, the loop thread queues behind it. The
  channels are SPSC by construction, so keep both sides of one edge on one thread. The binding
  takes the channel mutex around the polled and blocking calls, so a violation degrades to
  contention rather than corruption -- but nothing detects it.
