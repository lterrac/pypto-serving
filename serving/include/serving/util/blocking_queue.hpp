#pragma once

/// A closable producer/consumer queue. `pop` blocks until there is a value or
/// the queue closes; with a capacity, `push` blocks until there is room.

#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <utility>

namespace serving::util
{

template <typename T>
class BlockingQueue
{
  public:

  /// `capacity` 0 is unbounded.
  explicit BlockingQueue(size_t capacity = 0)
    : _capacity(capacity)
  {}

  BlockingQueue(const BlockingQueue &)            = delete;
  BlockingQueue &operator=(const BlockingQueue &) = delete;

  /// Drops the value if the queue is already closed.
  void push(T value)
  {
    std::unique_lock<std::mutex> lock(_mutex);
    if (_capacity != 0)
    {
      _cv.wait(lock, [this] { return _queue.size() < _capacity || _closed; });
    }
    if (_closed) { return; }
    _queue.push_back(std::move(value));
    lock.unlock();
    _cv.notify_all();
  }

  /// The next value, or nullopt once the queue is closed and drained.
  [[nodiscard]] std::optional<T> pop()
  {
    std::unique_lock<std::mutex> lock(_mutex);
    _cv.wait(lock, [this] { return !_queue.empty() || _closed; });
    if (_queue.empty()) { return std::nullopt; }
    T value = std::move(_queue.front());
    _queue.pop_front();
    lock.unlock();
    _cv.notify_all();
    return value;
  }

  /// Wakes every waiter. Values already queued are still drained by `pop`.
  void close()
  {
    {
      const std::lock_guard<std::mutex> lock(_mutex);
      _closed = true;
    }
    _cv.notify_all();
  }

  [[nodiscard]] bool closed() const
  {
    const std::lock_guard<std::mutex> lock(_mutex);
    return _closed;
  }

  private:

  size_t                  _capacity;
  mutable std::mutex      _mutex;
  std::condition_variable _cv;
  std::deque<T>           _queue;
  bool                    _closed = false;
};

} // namespace serving::util
