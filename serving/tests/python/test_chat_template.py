"""ChatTemplate through the bindings: an inline template, and the golden fixture."""

from __future__ import annotations

import importlib
import json
import pathlib

import pytest

s = importlib.import_module("_pypto_serving")

FIXTURE = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "qwen3_chat_template.json"


def test_chat_template_renders_from_python():
    template = s.ChatTemplate(
        "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}"
    )
    out = template.apply([{"role": "user", "content": "hi"}], add_generation_prompt=True)
    assert out == "user: hi\nassistant:"
    assert template.apply([{"role": "user", "content": "hi"}], add_generation_prompt=False) == "user: hi\n"


def test_renders_the_golden_fixture_as_transformers_did():
    # The same cases the C++ test renders; the fixture was dumped from the
    # Python stack, so this pins Python -> C++ -> Python end to end.
    fixture = json.loads(FIXTURE.read_text())
    template = s.ChatTemplate(fixture["chat_template"], "", "<|im_end|>")
    for case in fixture["cases"]:
        kwargs = dict(case.get("kwargs", {}))
        add_generation_prompt = kwargs.pop("add_generation_prompt", True)
        rendered = template.apply(case["messages"], add_generation_prompt=add_generation_prompt, extra_context=kwargs or None)
        assert rendered == case["expected"], case["name"]


def test_load_chat_template_returns_none_without_a_config(tmp_path):
    assert s.load_chat_template(str(tmp_path)) is None
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages[0]['content'] }}", "eos_token": "<e>"}))
    loaded = s.load_chat_template(str(tmp_path))
    assert loaded is not None
    assert loaded.apply([{"role": "user", "content": "x"}]) == "x"
