#pragma once

/**
 * A bounded map that forgets what was used longest ago.
 *
 * Three of these were written out by hand -- the replica's rule table, the
 * router's session directory and the coordinator's standing paths -- each with
 * the same `std::list` of keys in use order, the same map from key to
 * {value, iterator}, and the same `splice` to move an entry to the back. All
 * three are keyed by a client-supplied session id, so all three need the bound
 * for the same reason: without one the map grows with the number of distinct
 * ids a caller chooses to send.
 *
 * The bookkeeping is what is shared, and it is the part that is easy to get
 * wrong -- erasing from the map without erasing from the list leaves a
 * dangling iterator that corrupts the next splice, and that bug would have had
 * to be found three times. What differs is policy, and policy stays with the
 * caller: the session directory expires entries by time before it evicts any,
 * the standing paths have to revoke a rule for whatever they drop, and the
 * rule table counts what it has lost. `trim` takes a callback for exactly that.
 *
 * Not thread-safe by itself. Each of the three already owns a mutex covering
 * more than this map, and a second lock inside would buy nothing.
 */

#include <cstddef>
#include <list>
#include <string>
#include <unordered_map>
#include <utility>

namespace serving::util
{

template <typename Value>
class LruMap
{
  public:

  /// `capacity` 0 is treated as 1: a map that can hold nothing would evict
  /// every entry as it was inserted, which no caller means.
  explicit LruMap(size_t capacity)
    : _capacity(capacity == 0 ? 1 : capacity)
  {}

  [[nodiscard]] size_t capacity() const { return _capacity; }
  [[nodiscard]] size_t size() const { return _index.size(); }
  [[nodiscard]] bool   empty() const { return _index.empty(); }

  /// The value for `key`, counting as a use. Null when absent.
  [[nodiscard]] Value *find(const std::string &key)
  {
    const auto it = _index.find(key);
    if (it == _index.end()) { return nullptr; }
    touch(it->second.second);
    return &it->second.first;
  }

  /// The value for `key` without counting as a use, for a caller that is
  /// reporting on the map rather than using it -- looking at one session must
  /// not make another the next evicted.
  [[nodiscard]] const Value *peek(const std::string &key) const
  {
    const auto it = _index.find(key);
    return it == _index.end() ? nullptr : &it->second.first;
  }

  /// Insert or replace, counting as a use. Does not evict: call `trim` after,
  /// so the caller decides what happens to what is dropped.
  Value &put(const std::string &key, Value value)
  {
    const auto it = _index.find(key);
    if (it != _index.end())
    {
      it->second.first = std::move(value);
      touch(it->second.second);
      return it->second.first;
    }
    _order.push_back(key);
    auto added = _index.emplace(key, std::pair<Value, Position>{std::move(value), std::prev(_order.end())});
    return added.first->second.first;
  }

  bool erase(const std::string &key)
  {
    const auto it = _index.find(key);
    if (it == _index.end()) { return false; }
    _order.erase(it->second.second);
    _index.erase(it);
    return true;
  }

  /**
   * Evict from the least recently used end until the map is within capacity,
   * handing each dropped entry to `onEvict(key, value)` first.
   *
   * The callback runs before the entry is gone, because a caller that has to
   * withdraw something published for it needs the value to do so.
   */
  template <typename OnEvict>
  void trim(OnEvict onEvict)
  {
    while (_index.size() > _capacity && !_order.empty())
    {
      const std::string oldest = _order.front();
      const auto        it     = _index.find(oldest);
      if (it == _index.end())
      {
        // Cannot happen while the two stay in step, and dropping the orphaned
        // key rather than looping forever is the safe reading if they ever do.
        _order.pop_front();
        continue;
      }
      onEvict(oldest, it->second.first);
      _order.pop_front();
      _index.erase(it);
      _evicted += 1;
    }
  }

  /// Entries dropped to stay within capacity, over the life of the map.
  [[nodiscard]] size_t evicted() const { return _evicted; }

  /// Every key, least recently used first. For reporting and for a caller that
  /// has to sweep its own policy across the map.
  [[nodiscard]] const std::list<std::string> &keysInUseOrder() const { return _order; }

  void clear()
  {
    _index.clear();
    _order.clear();
  }

  private:

  using Position = typename std::list<std::string>::iterator;

  void touch(Position position) { _order.splice(_order.end(), _order, position); }

  size_t                                                        _capacity;
  std::list<std::string>                                        _order;
  std::unordered_map<std::string, std::pair<Value, Position>>   _index;
  size_t                                                        _evicted = 0;
};

} // namespace serving::util
