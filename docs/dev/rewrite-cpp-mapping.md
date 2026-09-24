# Serving in C++: what maps to what

Per Python module, the C++ that replaces it, the commit that introduced it, and the tests
on each side. Companion to `rewrite-cpp-plan.md`, which says why the cut is where it is.

## Revisions

The port was made from `pypto_serving/` at `d6179e9`: upstream `main` as of `3f1dbd9`
plus the Python router (`78ab0d9..d6179e9`, `feat/multi-node-router`). The branch is based
on `c9e1a5c`, an older `main`. So the `pypto_serving/` checked out next to `serving/` is
not the Python the C++ was read from; for a side-by-side, take the Python from the source
revision:

```
git show d6179e9:pypto_serving/serving/sched/scheduler.py | diff - serving/source/sched/scheduler.cpp
```

Line counts below are `wc -l` at those revisions. C++ counts are header plus source.
Commits are named by subject, which survives a rebase; `git log --oneline --grep='<subject>'`
finds one.

## Engine

| Python | lines | C++ | lines | introduced by | |
| --- | ---: | --- | ---: | --- | --- |
| `config/types.py` | 369 | `config/types.hpp` | 47 | meson project, config/types.hpp, and its Python face | `GenerateConfig` and `RuntimeConfig` only; the other nine types had no C++ consumer |
| `serving/sched/scheduler.py` | 1088 | `sched/scheduler.{hpp,cpp}` | 932 | port the continuous-batching scheduler | `SchedulerConfig` lives here, with `validate()`; grouped-cache phase not ported |
| `serving/memory/kv_cache.py` | 1391 | `memory/kv_cache.{hpp,cpp}` | 383 | port the paged KV cache and prefix cache | single-pool half; `hasGroups()` is `false`; no `usage()` |
| `model/tokenizer.py` | 251 | `model/tokenizer.hpp`, `model/hf_tokenizer.{hpp,cpp}`, `model/model_files.{hpp,cpp}` | 242 | port the tokenizer and incremental detokenization, stop spelling the same thing three times | `TokenizerAdapter` is the interface, `HfTokenizer` the tokenizers-cpp implementation |
| `apply_chat_template` (transformers) | — | `model/chat_template.{hpp,cpp}` | 84 | render chat templates with minja | minja; byte-equal golden fixture |
| `serving/engine/async_engine.py` — `_detokenize_incrementally`, `_advance_detokenization` | | `engine/detokenizer.{hpp,cpp}` | 125 | port the tokenizer and incremental detokenization | |
| `serving/engine/async_engine.py` — `ReplicaEngineCore` | 1522 total | `engine/engine.{hpp,cpp}` | 483 | run the engine loop on a thread, driven from Python | one thread; per-request `BlockingQueue` replaces asyncio streams |
| `serving/engine/async_engine.py` — parser detokenization, `set_stat_logger` | | not ported | | | reasoning and observability, see Gaps |
| `serving/server/server.py` | 548 | `server/http_server.{hpp,cpp}` | 438 | port the OpenAI HTTP surface with SSE streaming | `/health` gates on engine readiness; `/metrics`, `/start_profile`, `/stop_profile` not ported |
| `serving/server/serving_worker.py`, `ipc.py`, `streamer.py` | 1384 | `engine/executor.hpp` | 97 | run the engine loop on a thread, driven from Python | the process boundary becomes an interface: `StepCommand` in, `StepResult` out |
| | | `pypto_serving/bridge/__init__.py` | 317 | run Qwen from Python on the C++ engine | Python; builds the batches `serving_worker` built |
| | | `pypto_serving/bridge/sampling.py` | 146 | sample with the parameters the request asked for | Python; `_allow_device_*` and `_sample_result_row` |
| | | `engine/bridge_executor.{hpp,cpp}` | 247 | pypto-serving-cpp, a C++ binary embedding Python | the embedded-CPython caller of the bridge, development only |
| `model/common/executor/sampler.py` | 195 | stays Python | | | called from the bridge |
| `serving/utils/prefill.py` | 64 | stays Python | | | called from the bridge |
| `worker/worker.py`, `model/` | | stays Python | | | the model layer |
| `cli/main.py` | 1035 | `pypto_serving/cpp/__main__.py`, `runtime.py`, `executor.py` | 323 | run Qwen from Python on the C++ engine | Python; the user-facing entry point. `source/main.cpp` (278) is the same assembly in C++, development only |
| `serving/utils/env.py`, `gc_utils.py` | 135 | not ported | | | |
| `serving/reasoning/parser.py` | 341 | not ported | | | |
| `observability/` | 708 | not ported | | | |

New in C++ with no Python counterpart: `util/blocking_queue.hpp` (80), `util/signals.hpp`
(40), `bindings/` (680, the `_pypto_serving` module).

## Router and coordinator

| Python | lines | C++ | lines | introduced by | |
| --- | ---: | --- | ---: | --- | --- |
| `router/routing.py` | 285 | `router/routing.{hpp,cpp}` | 503 | port session-affinity routing | `SessionDirectory`, `ReplicaRegistry`, the affinity rule |
| `router/proxy.py` | 161 | `router/proxy.{hpp,cpp}` | 310 | streaming proxy, health polling and the router binary | |
| `router/app.py`, `router/config.py` | 328 | `router/router_server.{hpp,cpp}`, `source/router_main.cpp` | 397 | streaming proxy, health polling and the router binary | the standalone router binary |
| — | | `router/strategy.{hpp,cpp}` | 429 | route a partitioned pipeline with the coordinator as ingress, publish routing rules once per session | partitioned routing: `ingress` and `rules` |
| — | | `coordinator/module.{hpp,cpp}`, `platform_module.hpp` | 272 | run the routing strategy as a platform module | the `serving::modules::Module` |
| — | | `coordinator/rule_message.{hpp,cpp}`, `channel_transport.hpp` | 264 | publish routing rules over platform channels | the only files that touch platform channels |
| — | | `coordinator/config_file.{hpp,cpp}` | 291 | choose the routing strategy from a config file | |

The Python router was never on `main`; it is the port's source and the C++ router's
reference. The partitioned strategy and the coordinator have no Python at all.

## Tests

Python tests at `d6179e9` and the C++ or pytest file covering the same component. Counts
are test functions as defined.

| Python | lines / tests | C++ or pytest | lines / tests | |
| --- | ---: | --- | ---: | --- |
| `sched/test_async_scheduler.py` | 1036 / 31 | `sched/test_scheduler.cpp` | 663 / 29 | |
| `engine/test_output_delivery.py` | 583 / 11 | `engine/test_detokenizer.cpp`, `engine/test_engine.cpp` | 213 / 6, 420 / 15 | |
| `engine/test_request_cleanup.py` | 135 / 4 | `engine/test_engine.cpp` | | abort and failure paths |
| `engine/test_async_engine_replicas.py`, `test_async_pipeline.py` | 717 / 14 | not ported | | async and pipelined paths |
| `server/test_health_readiness.py` | 112 / 9 | `server/test_http_server.cpp`, `python/test_http_server.py` | 349 / 14, 32 / 1 | |
| `server/test_streaming_usage.py` | 279 / 5 | `server/test_http_server.cpp` | | the terminal usage chunk; the reasoning cases are not ported |
| `server/test_worker_sampling.py` | 235 / 4 | `python/test_bridge_sampling.py` | 212 / 13 | torch-free |
| `server/test_worker_step_protocol.py` | 609 / 16 | `engine/test_engine.cpp`, `python/test_engine.py` | , 141 / 9 | the step shape, through the trampoline |
| `server/test_metrics.py`, `test_profiling.py` | 310 / 9 | not ported | | |
| `reasoning/test_deepseek_v4_parser.py` | 160 / 9 | not ported | | |
| `router/test_routing.py` | 359 / 26 | `router/test_routing.cpp` | 416 / 25 | |
| `router/test_proxy.py`, `test_end_to_end.py` | 727 / 31 | `router/test_router_server.cpp` | 390 / 13 | |
| `cli/test_parallel_options.py`, `config/test_parallel_config.py` | 218 / 12 | not ported | | parallelism |
| — | | `memory/test_kv_cache.cpp`, `python/test_kv_cache.py` | 321 / 22, 64 / 4 | the Python KV cache has no unit test of its own; it is covered through the scheduler's |
| — | | `model/test_chat_template.cpp`, `python/test_chat_template.py` | 55 / 2, 42 / 3 | golden against transformers, `fixtures/qwen3_chat_template.json` |
| — | | `model/test_hf_tokenizer.cpp`, `python/test_tokenizer.py` | 93 / 4, 43 / 3 | need `SERVING_TEST_MODEL_DIR` |
| — | | `router/test_strategy.cpp` | 409 / 18 | |
| — | | `coordinator/test_*.cpp` | 700 / 39 | against fake transports |
| — | | `python/test_config_types.py`, `test_scheduler_config.py` | 72 / 7, 18 / 1 | the binding surface |

`serving/tests/fixtures/qwen3_generation.json` holds the token ids the Python stack
produces for the parity prompt; the device run must reproduce them.

## Reading the series

One component per commit, in dependency order: design doc and submodules, then config,
chat template, KV cache, scheduler, tokenizer, engine, HTTP, Qwen end to end; then the
router in three steps and the coordinator in three; then the development binary; then the
review fixes, each its own commit. Docs are folded into the first commit.

```
git log --reverse --oneline c9e1a5c..feat/serving-cpp-rewrite
git show --stat <commit>
```
