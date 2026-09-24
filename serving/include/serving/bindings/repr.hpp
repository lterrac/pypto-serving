#pragma once

/**
 * Dataclass-style __repr__ for bound types: `Name(field=value, ...)`.
 *
 * `show` renders one value as Python prints it. Overloads for a bound struct
 * live in that struct's namespace, so the optional/vector templates find them
 * by argument-dependent lookup.
 */

#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace serving::bindings
{

inline std::string show(int v) { return std::to_string(v); }
inline std::string show(uint64_t v) { return std::to_string(v); }
inline std::string show(bool v) { return v ? "True" : "False"; }
inline std::string show(double v)
{
  std::ostringstream out;
  out << v;
  const std::string s = out.str();
  return s.find_first_of(".e") == std::string::npos ? s + ".0" : s;
}
inline std::string show(const std::string &v) { return "'" + v + "'"; }
template <typename T>
inline std::string show(const std::optional<T> &v)
{
  return v ? show(*v) : "None";
}
template <typename T>
inline std::string show(const std::vector<T> &v)
{
  std::string out = "[";
  for (size_t i = 0; i < v.size(); ++i) { out += (i ? ", " : "") + show(v[i]); }
  return out + "]";
}

class Repr
{
  public:

  explicit Repr(const char *name)
    : _out(name)
  {
    _out += "(";
  }

  template <typename T>
  Repr &field(const char *name, const T &value)
  {
    _out += (_first ? "" : ", ") + std::string(name) + "=" + show(value);
    _first = false;
    return *this;
  }

  [[nodiscard]] std::string str() const { return _out + ")"; }

  private:

  std::string _out;
  bool        _first = true;
};

} // namespace serving::bindings
