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

/// Everything below, in dependency order. Both module faces call this.
void bindServing(pybind11::module_ &m);

void bindConfigTypes(pybind11::module_ &m);  // config/types.hpp
void bindChatTemplate(pybind11::module_ &m); // model/chat_template.hpp
void bindKvCache(pybind11::module_ &m);      // memory/kv_cache.hpp
void bindScheduler(pybind11::module_ &m);    // sched/scheduler.hpp (SchedulerConfig)

} // namespace serving::bindings
