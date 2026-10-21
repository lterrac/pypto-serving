#pragma once

/**
 * Reading a configuration document strictly, in one place.
 *
 * Three readers grew their own copy of this -- the infrastructure document, the
 * coordinator's configuration and the deployer's -- and the copies drifted on
 * the fields they share. `port` was bounded 1..65535 in one, 0..65535 in
 * another and not bounded at all in the third, so the same object was accepted
 * by one surface and rejected by the next. A config reader that disagrees with
 * itself is worse than a lax one, because which answer you get depends on which
 * file happened to parse it.
 *
 * The readers differ in two ways that matter and nothing else: the exception
 * they raise, and whether they hold an ordered or an unordered json. Both are
 * template parameters, so each keeps its own error type and its own message
 * shape while the parsing and the bounds are shared.
 */

#include <set>
#include <string>
#include <utility>
#include <vector>

namespace serving::util
{

/**
 * Where in a document a message is about, and what to throw when it is wrong.
 *
 * `source` names the file and may be empty for a document that has no name;
 * `where` is the path within it. `at` joins with a dot for a field of an
 * object, `in` with a colon for an element of a sequence -- the two shapes the
 * existing readers use, kept because the messages are what an operator sees.
 */
template <typename Error>
class Context
{
  public:

  explicit Context(std::string source = {}, std::string where = {})
    : _source(std::move(source)),
      _where(std::move(where))
  {}

  [[nodiscard]] Error error(const std::string &what) const
  {
    std::string message;
    if (!_source.empty()) { message += _source + ": "; }
    if (!_where.empty()) { message += _where + ": "; }
    return Error(message + what);
  }

  [[nodiscard]] Context at(const std::string &child) const { return Context(_source, _where.empty() ? child : _where + "." + child); }

  [[nodiscard]] Context in(const std::string &child) const { return Context(_source, _where.empty() ? child : _where + ": " + child); }

  private:

  std::string _source;
  std::string _where;
};

/// Every key must be one this reader knows. A configuration that is silently
/// half-read is how a deployment launches something nobody intended.
template <typename Json, typename Ctx>
void rejectUnknownKeys(const Json &object, const std::set<std::string> &allowed, const Ctx &ctx)
{
  for (const auto &[key, value] : object.items())
  {
    if (allowed.count(key) == 0) { throw ctx.error("unknown key '" + key + "'"); }
  }
}

template <typename Json, typename Ctx>
const Json &require(const Json &object, const char *key, const Ctx &ctx)
{
  const auto it = object.find(key);
  if (it == object.end()) { throw ctx.error(std::string("'") + key + "' is required"); }
  return *it;
}

template <typename Json, typename Ctx>
std::string requireString(const Json &object, const char *key, const Ctx &ctx)
{
  const Json &value = require(object, key, ctx);
  if (!value.is_string()) { throw ctx.error(std::string("'") + key + "' must be a string"); }
  return value.template get<std::string>();
}

template <typename Json, typename Ctx>
std::string optionalString(const Json &object, const char *key, const std::string &fallback, const Ctx &ctx)
{
  const auto it = object.find(key);
  if (it == object.end()) { return fallback; }
  if (!it->is_string()) { throw ctx.error(std::string("'") + key + "' must be a string"); }
  return it->template get<std::string>();
}

/// An integer within an inclusive range. The range is stated once per field so
/// two readers of the same field cannot disagree about what it allows.
template <typename Json, typename Ctx>
long long integerIn(const Json &value, const char *key, long long lo, long long hi, const Ctx &ctx)
{
  if (!value.is_number_integer()) { throw ctx.error(std::string("'") + key + "' must be an integer"); }
  const auto number = value.template get<long long>();
  if (number < lo || number > hi) { throw ctx.error(std::string("'") + key + "' must be between " + std::to_string(lo) + " and " + std::to_string(hi)); }
  return number;
}

template <typename Json, typename Ctx>
long long requireInteger(const Json &object, const char *key, long long lo, long long hi, const Ctx &ctx)
{
  return integerIn(require(object, key, ctx), key, lo, hi, ctx);
}

template <typename Json, typename Ctx>
long long optionalInteger(const Json &object, const char *key, long long fallback, long long lo, long long hi, const Ctx &ctx)
{
  const auto it = object.find(key);
  if (it == object.end()) { return fallback; }
  return integerIn(*it, key, lo, hi, ctx);
}

template <typename Json, typename Ctx>
double optionalNumber(const Json &object, const char *key, double fallback, double lo, double hi, const Ctx &ctx)
{
  const auto it = object.find(key);
  if (it == object.end()) { return fallback; }
  if (!it->is_number()) { throw ctx.error(std::string("'") + key + "' must be a number"); }
  const auto number = it->template get<double>();
  if (number < lo || number > hi) { throw ctx.error(std::string("'") + key + "' must be between " + std::to_string(lo) + " and " + std::to_string(hi)); }
  return number;
}

template <typename Json, typename Ctx>
bool optionalBoolean(const Json &object, const char *key, bool fallback, const Ctx &ctx)
{
  const auto it = object.find(key);
  if (it == object.end()) { return fallback; }
  if (!it->is_boolean()) { throw ctx.error(std::string("'") + key + "' must be a boolean"); }
  return it->template get<bool>();
}

/// The inclusive range a TCP port may take, shared so the three readers of this
/// field agree. Zero means "ask the kernel for one", which a replica that has
/// not been placed yet legitimately carries; a reader that must have a real
/// port asks for 1 as its lower bound.
inline constexpr int PORT_MIN = 0;
inline constexpr int PORT_MAX = 65535;

} // namespace serving::util
