# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tokenizer, chat template, executor, engine and HTTP server, assembled from Python."""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from dataclasses import dataclass, field

import _pypto_serving as cpp

from pypto_serving.cpp.executor import PyptoModelExecutor

logger = logging.getLogger(__name__)


@dataclass
class ServingOptions:
    """What a serving process needs; the flags of ``python -m pypto_serving.cpp``."""

    model_dir: str
    served_model_name: str = ""
    platform: str = "a2a3"
    device: int = 0
    max_model_len: int = 1024
    block_size: int = 128
    max_num_seqs: int = 16
    max_num_batched_tokens: int = 4096
    long_prefill_token_threshold: int = 2048
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    pypto_build_dir: str = "build_output/bridge"

    def __post_init__(self) -> None:
        if not self.served_model_name:
            self.served_model_name = os.path.basename(os.path.normpath(self.model_dir))


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int] = field(default_factory=list)
    finish_reason: str = ""
    seconds: float = 0.0


def engine_config(options: ServingOptions) -> cpp.EngineConfig:
    config = cpp.EngineConfig()
    config.runtime = cpp.RuntimeConfig(
        page_size=options.block_size,
        max_batch_size=options.max_num_seqs,
        max_seq_len=options.max_model_len,
    )
    config.scheduler = cpp.SchedulerConfig(
        max_seq_len=options.max_model_len,
        max_num_running_reqs=options.max_num_seqs,
        max_num_scheduled_tokens=options.max_num_batched_tokens,
        long_prefill_token_threshold=options.long_prefill_token_threshold,
        enable_prefix_cache=options.enable_prefix_caching,
        enable_chunk_prefill=options.enable_chunked_prefill,
    )
    return config


class Runtime:
    """Owns the runtime.

    Blocking calls into the engine release the GIL; the engine and HTTP threads
    call back into Python through the executor and the tokenizer.
    """

    def __init__(
        self,
        options: ServingOptions,
        *,
        tokenizer: cpp.TokenizerAdapter | None = None,
        executor: cpp.ModelExecutor | None = None,
        chat_template: cpp.ChatTemplate | None = None,
    ) -> None:
        self.options = options
        if tokenizer is None:
            if not cpp.HAS_TOKENIZERS:
                raise RuntimeError("_pypto_serving was built without tokenizers-cpp; pass a tokenizer")
            tokenizer = cpp.HfTokenizer.from_model_dir(options.model_dir)
        self.tokenizer = tokenizer
        self.chat_template = chat_template if chat_template is not None else cpp.load_chat_template(options.model_dir)
        self.executor = executor or PyptoModelExecutor(
            options.model_dir,
            options.device,
            platform=options.platform,
            max_model_len=options.max_model_len,
            block_size=options.block_size,
            max_num_seqs=options.max_num_seqs,
            pypto_build_dir=options.pypto_build_dir,
        )
        self.engine = cpp.Engine(engine_config(options), self.tokenizer, self.executor)
        self.server: cpp.HttpServer | None = None

    def start(self) -> float:
        """Load the model and start the engine; returns the seconds it took."""
        began = time.monotonic()
        self.engine.start()
        return time.monotonic() - began

    def generate(self, prompt: str, generate: cpp.GenerateConfig | None = None) -> GenerationResult:
        began = time.monotonic()
        stream = self.engine.add_request(
            self.engine.generate_request_id(),
            self.tokenizer.encode(prompt),
            generate or cpp.GenerateConfig(),
        )
        result = GenerationResult(text="")
        for update in stream:
            result.text = update.text
            if update.token_id is not None:
                result.token_ids.append(update.token_id)
            if update.finished:
                result.finish_reason = update.finish_reason
        result.seconds = time.monotonic() - began
        return result

    def serve(self) -> None:
        """Serve HTTP until SIGINT or SIGTERM."""
        config = cpp.ServerConfig(host=self.options.host, port=self.options.port, model_id=self.options.served_model_name)
        self.server = cpp.HttpServer(config, self.engine, self.tokenizer, self.chat_template)
        self.server.start()
        logger.info("listening on %s:%d", self.options.host, self.server.bound_port)

        stop = threading.Event()
        previous = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            stop.wait()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            self.stop()

    def stop(self) -> None:
        if self.server is not None:
            self.server.stop()
            self.server = None
        self.engine.stop()
