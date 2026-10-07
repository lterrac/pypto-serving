#pragma once

/**
 * What this machine actually has, for the state map.
 *
 * Only the devices are discovered. A node's name and address are configuration:
 * inside a container the hostname is the container's own id, so a machine
 * cannot name itself, and the deployment has to say who it is.
 */

#include <string>
#include <vector>

#include <serving/infra/state.hpp>

namespace serving::infra
{

/// Physical device ids this machine exposes, ascending.
///
/// Reads the `davinci<N>` character devices the driver creates. `davinci_manager`
/// and the other control nodes are not devices and are skipped. An empty result
/// means no accelerator, which is what a CPU-only build and a dev box both see.
[[nodiscard]] std::vector<int> discoverDevices(const std::string &deviceDir = "/dev");

/// This machine as a map entry, with the devices it was found to own.
[[nodiscard]] Node localNode(std::string name, std::string host, const std::string &deviceDir = "/dev");

} // namespace serving::infra
