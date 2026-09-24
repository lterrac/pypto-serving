# Serving in C++

## Boundary

pypto's L3 entry point is generated Python (`<output_dir>/orchestration/host_orch.py`,
regenerated per compile, invoked on every prefill and decode step through
`DistributedCompiledProgram.__call__` → `distributed_runner._load_orch_entry` → simpler
`Worker._submit_l3_locked`). A C++ process cannot dispatch a pypto program without Python
in the loop, so the split is:

| | Language |
| --- | --- |
| Model layer: `pypto_serving/model/`, `worker/worker.py`, sampler | Python, unchanged |
| Engine: config, KV cache, scheduler, engine loop, tokenizer, HTTP, router, coordinator | C++ (`serving/`) |
| Bridge between them | `pypto_serving/bridge/` (Python) + `serving/engine/bridge_executor` (C++) |
| Process owner | Python (`pypto_serving/cpp/`), through the `_pypto_serving` bindings; C++ mains for development |

The boundary is `ModelExecutor` (`pypto_serving/model/common/executor/executor.py`):
`register_model`, `run_prefill`, `run_decode`, `close`, plus capability flags read once at
startup. The KV cache manager does not cross it (the Qwen executor is constructed with
`kv_cache_manager=None`), and Qwen's per-step tensors are `[B]`/`[B,1]` integer tensors
(it embeds on device), so the C++ side carries no libtorch: a step crosses the bridge as
JSON — per request, its token ids, block ids, lengths and sampling parameters — and the
Python shim builds the torch tensors.

### Process layout

The user runs Python. `python -m pypto_serving.cpp --model DIR` (or `Runtime` in
`pypto_serving/cpp/runtime.py`) builds tokenizer, chat template, executor, engine and HTTP
server through the bindings and owns them; the engine and HTTP threads are C++ threads that
call back into Python only through the executor and, if it is a Python one, the tokenizer.
The executor is `pypto_serving/cpp/executor.py`: a Python subclass of `ModelExecutor` that
drives the existing `Qwen314BPyptoExecutor`. Every binding that can block on the engine
thread releases the GIL.

simpler forks one child per chip from Python (`worker.py:_forked_child_main`) and installs
`pthread_atfork` in the parent before doing so (`simpler/src/common/hierarchical/worker.h`).
Forking a process that already has C++ threads corrupts that. `Engine.start()` therefore
runs `register_model` -- the model load -- on the calling thread and starts the engine
thread only afterwards.

`pypto-serving-cpp` is the same assembly with C++ owning the process and embedding Python
(`bridge_executor`, JSON over the boundary); it is kept for development. There
`Py_Initialize` leaves the GIL held by the calling thread and `PyEval_SaveThread` releases
it once the model is up.

### Threads

One process:

* **HTTP** — cpp-httplib's pool: request parsing, chat template, tokenization.
* **Engine** — scheduler → step → reconciliation. Requests in on an MPSC queue; tokens out
  on per-request streams the HTTP layer drains for SSE.
* **Device** — one thread holding the GIL for the duration of a bridge call. Qwen takes the
  serial path (`supports_async_decode_prepare` is false); the pipelined three-lane loop is
  DeepSeek-only and not ported.

## Build

`serving/` is a top-level meson project, sibling to `platform/`. `platform/` is an optional
subproject (`-Dplatform=`), reached through the `serving/subprojects/platform` symlink;
nothing outside the coordinator needs it.

```
serving/
  meson.build  meson_options.txt  .clang-format      # .clang-format from platform/
  include/serving/{config,memory,sched,engine,model,server,router,coordinator,bindings}/
  source/  tests/  examples/  extern/
```

Namespace `serving::`, macros `__SERVING_`, LLVM-based `.clang-format`. Headers plus
sources.

| Dependency | Mechanism |
| --- | --- |
| minja, nlohmann/json, cpp-httplib, pybind11 | submodules under `serving/extern/` (see its README) |
| tokenizers-cpp | pkg-config `tokenizers_cpp`; `PKG_CONFIG_PATH` at the install tree (`-Dtokenizers=`). Built with `CMAKE_POSITION_INDEPENDENT_CODE=ON`: it is linked into the extension module |
| CPython | `import('python')`; embed dependency for the bridge (`-DpythonBridge=`), extension dependency for the bindings (`-DpythonBindings=`), `-DpythonPath=` selects the interpreter |
| googletest | `dependency('gtest_main')` |
| libtorch | not linked |

Features are `auto`: each is on where its dependency is found, and the core builds with all
of them off. pybind11 requires exceptions and hidden visibility; both are set on every
target that includes it.

## Components

Each is a transliteration of the Python module it names, same names, so the two diff
against each other. Tests are the Python tests ported to gtest, plus golden fixtures
generated from the Python stack (`serving/tests/fixtures/`).

| C++ | Python |
| --- | --- |
| `config/types.hpp` | the two types the engine reads from `config/types.py`: `GenerateConfig`, `RuntimeConfig` |
| `memory/kv_cache` | `serving/memory/kv_cache.py`, single-pool half; grouped caches are DeepSeek-only |
| `sched/scheduler` | `serving/sched/scheduler.py`; `SchedulerConfig::validate()` replaces `__post_init__`, same messages, run when the scheduler is built |
| `model/tokenizer`, `model/hf_tokenizer` | `model/tokenizer.py` over tokenizers-cpp |
| `model/chat_template.hpp` | `apply_chat_template` over minja, `apply_polyfills = false` |
| `engine/detokenizer` | `_detokenize_incrementally`; withholds a trailing partial UTF-8 sequence |
| `engine/engine`, `engine/executor.hpp` | `ReplicaEngineCore`; `ModelExecutor` |
| `engine/bridge_executor` | the executor that calls `pypto_serving.bridge` |
| `server/http_server` | OpenAI `/v1/completions`, `/v1/chat/completions`, `/v1/models`; `/health` is 503 until the engine is ready; `chat_template_kwargs` and `reasoning_effort` reach the template in the Python server's order |
| `router/routing`, `router/proxy`, `router/router_server` | `pypto_serving/router/` (from `feat/router-ssh-launcher`; not on `main`) — standalone router, kept as a reference |
| `router/strategy` | routing for a partitioned deployment, below |
| `coordinator/` | the partition coordinator, below |
| `bindings/` | `_pypto_serving`, below |

Binaries: `pypto-serving-cpp` (needs `pythonBridge` and `tokenizers`), `pypto-serving-router-cpp`.

## Sampling

A request's `temperature`, `top_p`, `top_k` and `seed` travel on every `PrefillItem` and
`DecodeItem` (`engine/executor.hpp`), per item rather than per step: a batch is not
homogeneous. The engine neither samples nor interprets them; the executor does.

`pypto_serving/bridge/sampling.py` chooses between the executor's three paths, mirroring
`WorkerProcess._allow_device_sampled_ids`, `._allow_device_topk_sampling` and
`._sample_result_row`:

| Path | Condition | Comes back |
| --- | --- | --- |
| device sampled ids | every request greedy, or the executor samples stochastically on device | one token id per row |
| device top-k candidates | every request names a `top_k` within `device_topk_sampling_k` (32 for Qwen3) | the k best values and ids per row |
| host logits | anything else | `[B, vocab]`, sampled by `model/common/executor/sampler.py` |

The third path does not batch. Qwen3-14B on one 910B2, identical prompts, 64 tokens each:

| requests in flight | greedy | `top_k=32` | no `top_k` |
| ---: | ---: | ---: | ---: |
| 1 | 34.1 tok/s | 32.4 | 19.1 |
| 4 | 128.7 | 118.0 | 31.1 |
| 8 | 245.5 | 222.4 | 35.4 |
| 16 | 444.7 | 381.2 | 37.9 |
| scaling 1 → 16 | 13.0x | 11.8x | **2.0x** |

Decode step time on the logits path is linear in the batch — 51.7, 126, 222, 413 ms at
1/4/8/16 — because the whole `[B, vocab]` logits block comes back to the host and
`Sampler.sample` then runs over 152k entries per row. The other two paths stay flat at
roughly 29 and 35 ms. At sixteen requests the fallback is 12x slower than greedy.

This is the Python worker's behaviour, not something the port introduced: the bridge runs
the same predicates and the same `Sampler`. But it means an OpenAI client that sets
`temperature` and leaves `top_k` unset — the common case — costs an order of magnitude of
throughput, and `top_k=32` recovers nearly all of it. Prefill returns logits on every
path, so the cost is decode's alone.

The flags go onto the batch, so one request that does not qualify moves the whole step: a
greedy request sharing a step with a sampled one puts both on logits, and reverts to
device sampling on the steps where it runs alone.

That module imports no torch, which is why the rules live apart from the batch building:
the predicates and the row dispatch are then tested on any machine
(`serving/tests/python/test_bridge_sampling.py`).

`top_k <= 0` and `top_p` outside `(0, 1]` are refused by the HTTP layer, where the caller
can be told which field was wrong.

## Routing

A model split across nodes runs as a chain of partitions, each with a coordinator and
replicas; a request follows a path of one replica per partition. Two options
(`scaling-pypto-serving.pptx`):

* **`ingress`** — the coordinator receives every request, routes it, and forwards to the
  next partition. Fully informed balancing, fewest channels; a coordinator hop per partition
  on the critical path.
* **`rules`** — the coordinator decides once per session and publishes a forwarding rule to
  each replica on the path; replicas forward directly. Viable because cached data must be
  reused: a conversation follows the same path every turn, so routing happens once per
  session.

Both use `RoutingPlanner` over a per-partition `ReplicaRegistry`; they differ in where the
decision is applied. Affinity is keyed per partition (that is where the KV lives). A path
carries a generation, bumped on replan, so a replica holding an older rule can recognise it
as stale; losing a replica revokes only the paths through it.

The option is chosen in the coordinator's configuration file (`coordinator/config_file.hpp`,
JSON):

```json
{ "routing_mode": "rules", "partition": 1,
  "replicas": [ { "name": "p0a", "host": "10.0.0.1", "port": 8000, "partition": 0, "instance": 10 } ] }
```

Replica entries use the fleet file's keys (`name`, `host`, `port`, `scheme`) plus
`partition` and `instance`. `routing_mode` is required; unknown keys and the launcher's
`hosts` key are rejected; every replica in the pipeline is listed.

### Coordinator

`coordinator/module.hpp` implements `serving::modules::Module`; the platform's engine owns
its lifecycle and drives `service()` on a timer. Replica losses are queued and applied on
the service tick. Rules travel over platform channels (`coordinator/channel_transport.hpp`,
`system::channels::Output`, message types `serving.routing.rule.publish` / `.revoke`, wire
format in `coordinator/rule_message.hpp`). Channels carry control-plane traffic only;
tensors between kernels are the kernels' own channels and the control plane never writes to
them.

`platform/include/system/channels/output.hpp` includes `<modules/configuration/edge.hpp>`,
which is not on `platform`'s `main`; the `havePlatformChannels` gate skips the transport when
that header is absent.

## Python bindings

`_pypto_serving` (`serving/bindings/`, pybind11) is the serving API as Python uses it.

* Config: `GenerateConfig`, `RuntimeConfig`, `SchedulerConfig`, `EngineConfig`,
  `KvCacheManager`, under the Python field names and with the dataclass defaults; frozen
  types are read-only. `std::invalid_argument` surfaces as `ValueError`, so a scheduler
  configuration or a prompt the scheduler cannot admit fails with the Python message.
* Runtime: `TokenizerAdapter` (implementable in Python), `HfTokenizer.from_model_dir`,
  `ChatTemplate` and `load_chat_template`, `ModelExecutor` (implementable in Python;
  `StepCommand` of `PrefillItem`/`DecodeItem` in — each carrying its `SamplingParams` —
  `StepResult` out, an exception becoming `StepResult.error`), `Engine`, `RequestStream`
  (iterable), `TokenOutput`, `ServerConfig`, `HttpServer`. The engine keeps its tokenizer
  and executor alive.

Built as an extension module and, through `PYBIND11_EMBEDDED_MODULE`, into
`pypto-serving-cpp` so the interpreter it embeds can import it too. Tested from pytest
(`serving/tests/python/`): the config contract, and the runtime driven end to end with a
Python tokenizer and a Python executor. `pypto_serving.config.types` does not yet re-export
the C++ types.

Not yet exposed: the coordinator and router, and anything from `platform/`.

## Verification

CPU:

```
meson setup serving/build serving -DbuildTests=true
meson test -C serving/build
```

Device (hg-atlas-01, through `task-submit`, `--device 4`), Python owning the process:

```
task-submit --device 4 --max-time 0 --run \
  "PTO2_RING_HEAP=536870912 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072 \
   PYTHONPATH=serving/build python -m pypto_serving.cpp --model /home/terra/models/Qwen3-14B \
     --prompt 'The capital of France is' --platform a2a3 --max-model-len 512 \
     --generate-config '{\"max_new_tokens\": 8}'"
```

`pypto-serving-cpp` takes the same flags.

Parity: the same prompt through the Python entry point must yield the same token ids
(`serving/tests/fixtures/qwen3_generation.json`). `SERVING_TEST_MODEL_DIR` enables the
tokenizer tests that need a model directory.

## Gaps

What the Python stack serves and this one does not, measured against `pypto_serving/` at
`3f1dbd9`, the revision the port was made from. No request field is accepted and ignored:
everything the HTTP layer parses reaches something that acts on it.

Ported but unreachable — the code and its tests are here, nothing turns them on:

| | |
| --- | --- |
| async scheduling | `SchedulerConfig::asyncScheduling`; no entry point sets it |
| speculative tokens | `numSpeculativeTokens`; the budget logic is ported, no executor implements it |

Not ported:

| | |
| --- | --- |
| grouped KV caches | `KvCacheManager::hasGroups()` is `false` |
| reasoning output parser | `serving/reasoning/parser.py`; `include_reasoning`, the `reasoning` response field and its stream deltas, the parser-aware detokenization in `async_engine.py` |
| observability | `observability/`: `/metrics`, `/metrics/json`, the access log, `set_stat_logger`, `KvCacheManager.usage()` |
| parallelism | `--devices`, `--tensor-parallel-size`, `--data-parallel-size`, `--expert-parallel-size`, `--data-parallel-routing`. One card, one rank |
| `--npu-memory-utilization`, `total_kv_pages` | `bridge.open_model` takes no such parameter; the page count comes back from the executor unmodulated |
| `--dtype`, `--kv-cache-dtype` | the bridge builds its `RuntimeConfig` with `bfloat16` for both |
| `--use-compile-cache`, `--backend`, `--show-startup-logs` | |
| profiling | `/start_profile`, `/stop_profile`, `ProfileCommand`, `--profile*` |
| `--ring-heap`, `--ring-task-window`, `--ring-dep-pool` | set through the environment instead |

## Out of scope

* DeepSeek V4 and DSpark: grouped KV caches, the pipelined decode loop, MTP.
* Replacing the Python model layer with a C++ orchestration path against simpler.
* Code from hllm / deepseek-hllm; those are references for library choice only.
