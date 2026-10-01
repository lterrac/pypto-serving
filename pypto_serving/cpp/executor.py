# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The pypto model as the engine's ModelExecutor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import _pypto_serving as cpp

from pypto_serving import bridge

if TYPE_CHECKING:
    from pypto_serving.cpp.runtime import ServingOptions


class PyptoModelExecutor(cpp.ModelExecutor):
    """Runs ``Qwen314BPyptoExecutor`` through ``pypto_serving.bridge``.

    ``register_model`` runs on the thread calling ``Engine.start()``, before the
    engine thread exists.
    """

    def __init__(self, options: "ServingOptions") -> None:
        super().__init__()
        self._options = options

    def register_model(self) -> int:
        o = self._options
        return bridge.open_model(
            model_dir=o.model_dir,
            device_id=o.device,
            platform=o.platform,
            max_model_len=o.max_model_len,
            block_size=o.block_size,
            max_num_seqs=o.max_num_seqs,
            pypto_build_dir=o.pypto_build_dir,
        )

    @property
    def page_size(self) -> int:
        return bridge.page_size()

    @staticmethod
    def _sampling(item) -> dict:
        """The same block ``BridgeExecutor`` puts on the wire, so both entry
        points hand the bridge one shape."""
        sampling = item.sampling
        return {"temperature": sampling.temperature, "top_p": sampling.top_p, "top_k": sampling.top_k, "seed": sampling.seed}

    def execute_step(self, command: cpp.StepCommand) -> cpp.StepResult:
        prefill = [
            {
                "request_id": item.request_id,
                "tokens": list(item.chunk_tokens),
                "num_computed": item.num_computed_tokens,
                "sample_at_length": item.sample_at_length,
                "block_ids": list(item.block_ids),
                "sampling": self._sampling(item),
            }
            for item in command.prefill
        ]
        decode = [
            {
                "request_id": item.request_id,
                "last_token": item.last_token,
                "seq_len": item.seq_len,
                "block_ids": list(item.block_ids),
                "sampling": self._sampling(item),
            }
            for item in command.decode
        ]
        return cpp.StepResult(new_tokens=bridge.run_step(prefill, decode))

    def close(self) -> None:
        bridge.close()
