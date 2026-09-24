# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""``python -m pypto_serving.cpp``: serve, or generate with ``--prompt``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence

import _pypto_serving as cpp

from pypto_serving.cpp.runtime import Runtime, ServingOptions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pypto_serving.cpp", description=__doc__)
    parser.add_argument("--model", required=True, help="model directory")
    parser.add_argument("--served-model-name", default="", help="name reported by the API (default: the directory name)")
    parser.add_argument("--platform", default="a2a3")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--long-prefill-token-threshold", type=int, default=2048)
    parser.add_argument("--no-prefix-caching", dest="enable_prefix_caching", action="store_false")
    parser.add_argument("--no-chunked-prefill", dest="enable_chunked_prefill", action="store_false")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--pypto-build-dir", default="build_output/bridge")
    parser.add_argument("--prompt", action="append", default=[], help="generate for this prompt and exit; repeatable")
    parser.add_argument("--generate-config", default="", help='JSON, e.g. \'{"max_new_tokens": 8}\'')
    return parser


def generate_config(text: str) -> cpp.GenerateConfig:
    if not text:
        return cpp.GenerateConfig()
    data = json.loads(text)
    known = {"max_new_tokens", "temperature", "top_p", "top_k", "seed", "stop", "stream", "ignore_eos"}
    unknown = sorted(set(data) - known)
    if unknown:
        raise SystemExit(f"--generate-config: unknown keys {unknown}")
    return cpp.GenerateConfig(**data)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    options = ServingOptions(
        model_dir=args.model,
        served_model_name=args.served_model_name,
        platform=args.platform,
        device=args.device,
        max_model_len=args.max_model_len,
        block_size=args.block_size,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        long_prefill_token_threshold=args.long_prefill_token_threshold,
        enable_prefix_caching=args.enable_prefix_caching,
        enable_chunked_prefill=args.enable_chunked_prefill,
        host=args.host,
        port=args.port,
        pypto_build_dir=args.pypto_build_dir,
    )

    runtime = Runtime(options)
    print(f"[serving] loading {options.model_dir} on device {options.device} ...", flush=True)
    print(f"[serving] ready in {runtime.start():.1f}s", flush=True)

    if args.prompt:
        generate = generate_config(args.generate_config)
        status = 0
        try:
            for index, prompt in enumerate(args.prompt, 1):
                result = runtime.generate(prompt, generate)
                rate = len(result.token_ids) / result.seconds if result.seconds and result.token_ids else 0.0
                print(f"\n-- prompt {index}/{len(args.prompt)} --")
                print(f"text: {result.text}")
                print(f"token_ids: {result.token_ids}")
                print(f"finish_reason: {result.finish_reason}")
                print(f"[generate] {len(result.token_ids)} tokens in {result.seconds:.2f}s ({rate:.1f} tok/s)")
                if not result.token_ids:
                    status = 1
        finally:
            runtime.stop()
        return status

    runtime.serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
