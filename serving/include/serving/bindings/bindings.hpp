#pragma once

/**
 * Python bindings: the `_pypto_serving` module.
 *
 * Python field names, keyword constructors with the dataclass defaults, frozen
 * types read-only, validate() failures raised as ValueError. Built as an
 * extension module (module.cpp) and registered on an embedding interpreter's
 * inittab (embedded.cpp); both call bindServing.
 */

#include <pybind11/pybind11.h>

namespace serving::bindings
{

/// Deleter for bound types whose destructor joins a thread that may be waiting
/// for the GIL. pybind11 runs dealloc with the GIL held, so joining under it
/// deadlocks against the engine thread inside a Python executor.
struct GilReleasingDelete
{
  template <typename T>
  void operator()(T *object) const
  {
    pybind11::gil_scoped_release release;
    delete object;
  }
};

/// Everything below, in dependency order. Both module faces call this.
void bindServing(pybind11::module_ &m);

void bindConfigTypes(pybind11::module_ &m);  // config/types.hpp
void bindChatTemplate(pybind11::module_ &m); // model/chat_template.hpp
void bindKvCache(pybind11::module_ &m);      // memory/kv_cache.hpp
void bindScheduler(pybind11::module_ &m);    // sched/scheduler.hpp (SchedulerConfig)
void bindTokenizer(pybind11::module_ &m);    // model/tokenizer.hpp, model/hf_tokenizer.hpp
void bindEngine(pybind11::module_ &m);       // engine/executor.hpp, engine/engine.hpp
void bindServer(pybind11::module_ &m);       // server/http_server.hpp

} // namespace serving::bindings
