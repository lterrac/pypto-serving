#pragma once

/**
 * The coordinator's configuration file.
 *
 * Which routing option a deployment runs is decided here, not on the command
 * line: a strategy is a property of the deployment, and a deployment is what a
 * file describes. The file is JSON, with the same `replicas` entries the Python
 * router's fleet file uses (`name`, `host`, `port`, `scheme`) so one description
 * can eventually drive both, plus what a coordinator needs on top of that.
 *
 *   {
 *     "routing_mode": "rules",            // "ingress" (Option A) or "rules" (Option B)
 *     "partition": 1,                     // the partition this coordinator owns
 *     "routing": {                        // optional
 *       "session_ttl_seconds": 1800,
 *       "affinity_slack": 8
 *     },
 *     "replicas": [
 *       { "name": "p0a", "host": "10.0.0.1", "port": 8000, "partition": 0, "instance": 10 },
 *       { "name": "p1a", "host": "10.0.0.2", "port": 8000, "partition": 1, "instance": 20,
 *         "scheme": "https" }
 *     ]
 *   }
 *
 * Every replica in the pipeline is listed, across all partitions: the
 * coordinator owns one partition but plans the whole path.
 *
 * Parsing is strict. An unknown key anywhere is an error, a missing required
 * field is an error, and every message names where it happened ("replica 2:
 * ..."), because a configuration that is silently half-read is how a deployment
 * ends up routing to the wrong place with nothing in the log.
 */

#include <filesystem>
#include <stdexcept>
#include <string>

#include <serving/coordinator/module.hpp>

namespace serving::coordinator
{

/// Anything wrong with a configuration: unreadable, not JSON, or not a valid
/// deployment. The message names the source and the offending entry.
class ConfigError : public std::runtime_error
{
  public:

  using std::runtime_error::runtime_error;
};

/**
 * Parse a configuration from JSON text.
 *
 * `source` is only used in error messages, so a caller reading from a file can
 * name the file and a test can name itself.
 */
[[nodiscard]] CoordinatorConfig parseCoordinatorConfig(const std::string &json, const std::string &source = "<config>");

/// Read and parse a configuration file. Errors name the path.
[[nodiscard]] CoordinatorConfig loadCoordinatorConfig(const std::filesystem::path &path);

} // namespace serving::coordinator
