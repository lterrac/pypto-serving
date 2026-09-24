# Test fixtures

Golden output captured from the Python stack. A C++ component that replaces a
Python one is pinned against these rather than against hand-written expectations,
so a divergence shows up as a test failure instead of as a wrong prompt.

## `qwen3_chat_template.json`

The Qwen3-14B chat template plus six rendered conversations from
`transformers.AutoTokenizer.apply_chat_template`. Covers a bare user turn, a
system+user pair, a multi-turn history, `add_generation_prompt=False`,
`enable_thinking=False`, and -- the case that matters most -- a historical
assistant turn carrying a `<think>` block, which the template strips.

Regenerate with `scripts/gen_chat_template_fixture.py`, which must run where the
model and `transformers` live (the `terra-pypto` container on hg-atlas-01):

```bash
docker exec terra-pypto python3 /mounted_home/<path>/gen_chat_template_fixture.py \
  --model /mounted_home/models/Qwen3-14B > qwen3_chat_template.json
```

## `qwen3_generation.json`

The parity target for the port: what the Python stack generates for a fixed prompt, with
the exact stack that produced it. Captured by running the Python CLI through
`task-submit` (see the `project_qwen_serving_run` note for the launch). No C++ test
consumes it yet -- it is the acceptance criterion for the worker/bridge milestone, where
the C++ engine must reproduce these token ids exactly.
