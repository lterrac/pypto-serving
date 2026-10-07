#pragma once

/**
 * The state map as a document on disk.
 *
 * Slide 12's declarative state: serving writes what the deployment should look
 * like, and the document is the single thing routing, scaling and monitoring
 * read. A replica carries both the phase it is observed in and the phase it is
 * wanted in, so one document holds spec and status rather than two.
 *
 * ```json
 * {
 *   "format": 1,
 *   "version": 12,
 *   "nodes": [
 *     { "name": "hg-atlas-01", "host": "192.168.150.11", "devices": [0, 1, 2, 3, 4, 5, 6, 7] }
 *   ],
 *   "replicas": [
 *     { "name": "qwen-0", "partition": 0, "node": "hg-atlas-01", "devices": [4],
 *       "port": 8000, "scheme": "http", "instance": 0,
 *       "phase": "active", "desired_phase": "active" }
 *   ],
 *   "channels": [ { "from": "qwen-0", "to": "qwen-1" } ]
 * }
 * ```
 *
 * Parsing is strict, like the coordinator's configuration file: an unknown key
 * anywhere is an error, a missing required field is an error, and every message
 * names where it happened. A document that is silently half-read is how a
 * deployment ends up routing somewhere nobody intended.
 */

#include <filesystem>
#include <stdexcept>
#include <string>

#include <serving/infra/state.hpp>

namespace serving::infra
{

/// The format this build writes and is the only one it reads.
inline constexpr int DOCUMENT_FORMAT = 1;

/// Anything wrong with a document: unreadable, not JSON, or not a deployment.
class DocumentError : public std::runtime_error
{
  public:

  using std::runtime_error::runtime_error;
};

/// The map as JSON, in a stable key order so two writes of one map compare equal.
[[nodiscard]] std::string writeDocument(const StateMap &map);

/// Parse a document. `source` only appears in error messages.
[[nodiscard]] StateMap readDocument(const std::string &text, const std::string &source = "<document>");

/// Read and parse. Errors name the path.
[[nodiscard]] StateMap loadDocument(const std::filesystem::path &path);

/**
 * Write it where a reader will only ever see it whole.
 *
 * Written to a sibling temporary and renamed over the target, because a reader
 * polling the document must not catch it half-written: rename within a
 * directory is atomic, a partial write is not.
 */
void saveDocument(const StateMap &map, const std::filesystem::path &path);

} // namespace serving::infra
