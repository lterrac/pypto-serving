# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Where this replica actually runs, for the infrastructure state map.

The map wants the *physical* card, because that is what the node owns and what
another replica would collide with. The process does not see it: the broker
pins the granted card through ``ASCEND_RT_VISIBLE_DEVICES``, so inside the job
it is always logical 0. ``TASK_PHYS_DEVICE`` carries the real one.

A node cannot name itself either -- inside a container the hostname is the
container's id -- so the name and address are configuration.
"""

from __future__ import annotations

import os

import _pypto_serving as cpp

#: What the task broker exports for the card it granted.
PHYSICAL_DEVICE_ENV = "TASK_PHYS_DEVICE"
#: The logical index the same card appears as inside the job.
LOGICAL_DEVICE_ENV = "TASK_DEVICE"


def granted_device(default: int | None = None) -> int | None:
    """The physical card this process was granted, or ``default`` off the broker."""
    raw = os.environ.get(PHYSICAL_DEVICE_ENV, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def describe_replica(
    name: str,
    *,
    node: str,
    port: int,
    partition: int = 0,
    device: int | None = None,
    phase: cpp.Phase = cpp.Phase.ACTIVE,
) -> cpp.Replica:
    """This process as a map entry.

    ``device`` defaults to the card the broker granted; pass it explicitly off
    the broker, where there is nothing to read.
    """
    card = granted_device(device)
    return cpp.Replica(
        name,
        partition=partition,
        node=node,
        devices=[] if card is None else [card],
        port=port,
        phase=phase,
    )
