#pragma once

/// Wait for SIGINT or SIGTERM.
///
/// A handler cannot do much safely -- notifying a condition variable from one
/// is not async-signal-safe, and doing it without the mutex loses a signal that
/// arrives between a waiter's predicate check and its wait. So the signals are
/// blocked in every thread and one thread calls sigwait, which needs no handler
/// and no shared state.

#include <csignal>

namespace serving::util
{

/// Block SIGINT and SIGTERM. Call before starting any thread: threads inherit
/// the mask, and a signal delivered to a thread that is not waiting would
/// otherwise terminate the process.
inline void blockShutdownSignals()
{
  sigset_t mask;
  sigemptyset(&mask);
  sigaddset(&mask, SIGINT);
  sigaddset(&mask, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &mask, nullptr);
}

/// Block until one of them arrives; returns the signal number.
inline int awaitShutdownSignal()
{
  sigset_t mask;
  sigemptyset(&mask);
  sigaddset(&mask, SIGINT);
  sigaddset(&mask, SIGTERM);
  int signalNumber = 0;
  sigwait(&mask, &signalNumber);
  return signalNumber;
}

} // namespace serving::util
