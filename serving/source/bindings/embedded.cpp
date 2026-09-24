// Registers _pypto_serving on the inittab so an embedding interpreter can
// import it. Compiled into the binary directly: a static library would drop
// this unreferenced object.
#include <pybind11/embed.h>

#include <serving/bindings/bindings.hpp>

PYBIND11_EMBEDDED_MODULE(_pypto_serving, m) { serving::bindings::bindServing(m); }
