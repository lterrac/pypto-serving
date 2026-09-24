"""TokenizerAdapter through the bindings: implementable in Python, HfTokenizer where built."""

from __future__ import annotations

import importlib
import os

import pytest

s = importlib.import_module("_pypto_serving")


class Reversing(s.TokenizerAdapter):
    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(i) for i in token_ids)

    def eos_token_id(self):
        return 0


def test_a_python_tokenizer_is_a_tokenizer_adapter():
    tok = Reversing()
    assert isinstance(tok, s.TokenizerAdapter)
    assert tok.decode(tok.encode("héllo")) == "héllo"
    assert tok.eos_token_id() == 0
    assert tok.bos_token_id() is None, "the base default"


def test_hf_tokenizer_is_bound_exactly_when_tokenizers_cpp_is_in():
    assert hasattr(s, "HfTokenizer") == s.HAS_TOKENIZERS


@pytest.mark.skipif(not os.environ.get("SERVING_TEST_MODEL_DIR"), reason="set SERVING_TEST_MODEL_DIR to run")
def test_hf_tokenizer_round_trips_the_golden_prompt():
    if not s.HAS_TOKENIZERS:
        pytest.skip("built without tokenizers-cpp")
    tok = s.HfTokenizer.from_model_dir(os.environ["SERVING_TEST_MODEL_DIR"])
    ids = tok.encode("The capital of France is")
    assert len(ids) == 5
    assert tok.decode([12095, 13, 3555, 374, 279, 6722, 315, 279]) == " Paris. What is the capital of the"
