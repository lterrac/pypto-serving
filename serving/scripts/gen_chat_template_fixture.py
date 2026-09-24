# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Dumps transformers' apply_chat_template output for a set of conversations.

Run where the model and ``transformers`` are installed; redirect stdout into
``serving/tests/fixtures/``.
"""

from __future__ import annotations

import argparse
import json

CASES = [
    dict(name="single_user", messages=[{"role": "user", "content": "Huawei is"}], add_generation_prompt=True),
    dict(
        name="system_user",
        messages=[{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi"}],
        add_generation_prompt=True,
    ),
    dict(
        name="multi_turn",
        messages=[
            {"role": "user", "content": "A?"},
            {"role": "assistant", "content": "B."},
            {"role": "user", "content": "C?"},
        ],
        add_generation_prompt=True,
    ),
    dict(name="no_gen_prompt", messages=[{"role": "user", "content": "Hi"}], add_generation_prompt=False),
    # The case that catches a stale minja: the template strips <think> out of a
    # historical assistant turn via `'</think>' in content`.
    dict(
        name="thinking_block",
        messages=[
            {"role": "user", "content": "A?"},
            {"role": "assistant", "content": "<think>\nmull\n</think>\n\nB."},
            {"role": "user", "content": "C?"},
        ],
        add_generation_prompt=True,
    ),
    dict(
        name="no_thinking",
        messages=[{"role": "user", "content": "Hi"}],
        add_generation_prompt=True,
        enable_thinking=False,
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to the model directory.")
    parser.add_argument("--name", default=None, help="Fixture label. Defaults to the directory name.")
    args = parser.parse_args()

    from transformers import AutoTokenizer  # noqa: PLC0415 -- only needed where the fixture is generated

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, local_files_only=True)

    cases = []
    for case in CASES:
        kwargs = {key: value for key, value in case.items() if key not in ("name", "messages")}
        rendered = tokenizer.apply_chat_template(case["messages"], tokenize=False, **kwargs)
        cases.append(dict(name=case["name"], messages=case["messages"], kwargs=kwargs, expected=rendered))

    fixture = dict(
        model=args.name or args.model.rstrip("/").rsplit("/", 1)[-1],
        chat_template=tokenizer.chat_template,
        cases=cases,
    )
    print(json.dumps(fixture, indent=1))


if __name__ == "__main__":
    main()
