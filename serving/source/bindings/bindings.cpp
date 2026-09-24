#include <serving/bindings/bindings.hpp>

namespace serving::bindings
{

void bindServing(pybind11::module_ &m)
{
  m.doc() = "pypto-serving: the C++ serving engine.";
  bindConfigTypes(m);
}

} // namespace serving::bindings
