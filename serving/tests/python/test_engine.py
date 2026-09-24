"""The engine from Python."""

from __future__ import annotations

import importlib

import pytest

from stubs import CountingExecutor, WordTokenizer, config, drain, running

s = importlib.import_module("_pypto_serving")


def test_engine_runs_with_a_python_tokenizer_and_a_python_executor():
    tok, ex = WordTokenizer(), CountingExecutor()
    engine = s.Engine(config(), tok, ex)
    assert not engine.is_ready

    with running(engine):  # start() calls ex.register_model on this thread, then starts the loop
        assert engine.is_ready
        assert ex.registered == 1

        prompt = tok.encode("one two three")
        tokens, last = drain(engine.add_request(engine.generate_request_id(), prompt, s.GenerateConfig(max_new_tokens=3)))
        assert tokens == [prompt[-1] + 1, prompt[-1] + 2, prompt[-1] + 3]
        assert last.finish_reason
        assert last.text == tok.decode(tokens)
    assert ex.closed


def test_engine_keeps_its_python_collaborators_alive():
    # Temporaries: only keep_alive on the Engine constructor keeps them valid.
    engine = s.Engine(config(), WordTokenizer(), CountingExecutor())
    with running(engine):
        tokens, _ = drain(engine.add_request("r", [1, 2, 3], s.GenerateConfig(max_new_tokens=2)))
        assert tokens == [4, 5]


def test_an_executor_exception_fails_the_request_not_the_engine():
    tok, ex = WordTokenizer(), CountingExecutor(fail_on_token=11)
    engine = s.Engine(config(), tok, ex)
    with running(engine):
        # 10 -> prefill emits 11 -> the next decode step raises.
        _, failed = drain(engine.add_request("bad", [10], s.GenerateConfig(max_new_tokens=5)))
        assert failed.finished
        assert "the model fell over" in failed.finish_reason

        # The engine thread is alive, and the failed request is gone from the
        # scheduler: were it still there, the executor would raise on it again
        # and take this request down with it.
        tokens, _ = drain(engine.add_request("good", [1], s.GenerateConfig(max_new_tokens=2)))
        assert tokens == [2, 3]


def test_a_rejected_prompt_is_a_value_error():
    engine = s.Engine(config(max_seq_len=4), WordTokenizer(), CountingExecutor())
    with running(engine), pytest.raises(ValueError, match="max_seq_len"):
        engine.add_request("long", [1, 2, 3, 4, 5], s.GenerateConfig())


def test_step_types_read_like_the_python_dicts():
    r = s.StepResult(new_tokens={"a": [1, 2]}, error="")
    assert r.new_tokens == {"a": [1, 2]}
    assert s.StepResult().error == ""
