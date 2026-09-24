// The _pypto_serving extension module.
#include <serving/bindings/bindings.hpp>

PYBIND11_MODULE(_pypto_serving, m) { serving::bindings::bindServing(m); }
