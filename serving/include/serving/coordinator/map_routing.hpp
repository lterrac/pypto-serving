#pragma once

/**
 * The routing view of the infrastructure state map.
 *
 * The map says what exists and in which phase; routing needs endpoints and
 * whether each one may carry traffic. This is the one place that translates
 * between them, so there is a single answer to "does phase X route?".
 *
 * | phase | routing |
 * | --- | --- |
 * | `active` | routable |
 * | `draining` | kept, not routable, its sessions released |
 * | `pending` | kept, not routable: it cannot serve yet |
 * | `dead` | taken out, and its paths revoked |
 */

#include <string>
#include <vector>

#include <serving/coordinator/module.hpp>
#include <serving/infra/state.hpp>

namespace serving::coordinator
{

/// Where a replica's endpoint comes from: its own port and its node's address.
/// A replica whose node is not in the map has no address and is skipped.
[[nodiscard]] std::vector<ReplicaBinding> bindingsFrom(const infra::StateMap &map);

} // namespace serving::coordinator
