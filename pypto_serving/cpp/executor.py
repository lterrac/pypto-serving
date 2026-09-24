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

import logging

import _pypto_serving as cpp

from pypto_serving import bridge

logger = logging.getLogger(__name__)


class PyptoModelExecutor(cpp.ModelExecutor):
    """Runs ``Qwen314BPyptoExecutor`` through ``pypto_serving.bridge``.

    ``register_model`` runs on the thread calling ``Engine.start()``, before the
    engine thread exists.
    """

    def __init__(
        self,
        model_dir: str,
        device_id: int,
        *,
        platform: str = "a2a3",
        max_model_len: int = 512,
        block_size: int = 128,
        max_num_seqs: int = 16,
        pypto_build_dir: str = "build_output/bridge",
    ) -> None:
        super().__init__()
        self._open = dict(
            model_dir=model_dir,
            device_id=device_id,
            platform=platform,
            max_model_len=max_model_len,
            block_size=block_size,
            max_num_seqs=max_num_seqs,
            pypto_build_dir=pypto_build_dir,
        )

    def register_model(self) -> int:
        return bridge.open_model(**self._open)

    @property
    def page_size(self) -> int:
        return bridge.page_size()

    def execute_step(self, command: cpp.StepCommand) -> cpp.StepResult:
        try:
            prefill = [
                {
                    "request_id": item.request_id,
                    "tokens": list(item.chunk_tokens),
                    "num_computed": item.num_computed_tokens,
                    "sample_at_length": item.sample_at_length,
                    "block_ids": list(item.block_ids),
                }
                for item in command.prefill
            ]
            decode = [
                {
                    "request_id": item.request_id,
                    "last_token": item.last_token,
                    "seq_len": item.seq_len,
                    "block_ids": list(item.block_ids),
                }
                for item in command.decode
            ]
            return cpp.StepResult(new_tokens=bridge.run_step(prefill, decode))
        except Exception as exc:  # noqa: BLE001 -- reported as the step's error, so the engine keeps serving
            logger.exception("model step failed")
            return cpp.StepResult(error=f"{type(exc).__name__}: {exc}")

    def close(self) -> None:
        bridge.close()
