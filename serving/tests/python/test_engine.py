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
        assert "the model fell over" in failed.error
        assert failed.finish_reason == "FINISHED_ERROR"

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


def test_dropping_the_engine_without_stopping_it_does_not_hang():
    # pybind11 runs dealloc with the GIL held, so a destructor that joins the
    # engine thread deadlocks against a Python executor waiting for the GIL.
    engine = s.Engine(config(), WordTokenizer(), CountingExecutor())
    engine.start()
    drain(engine.add_request("r", [1, 2, 3], s.GenerateConfig(max_new_tokens=2)))
    del engine  # no explicit stop()


def test_dropping_the_server_without_stopping_it_does_not_hang():
    tok, ex = WordTokenizer(), CountingExecutor()
    engine = s.Engine(config(), tok, ex)
    with running(engine):
        server = s.HttpServer(s.ServerConfig(host="127.0.0.1", port=0), engine, tok)
        server.start()
        del server


def test_preemption_replay_reproduces_the_unpreempted_output():
    """A request preempted mid-generation must resume to the same tokens.

    It keeps what it generated but loses its KV, so on resume it replays prompt
    plus generated. Feeding that replay as decode instead of prefill leaves the
    rows in between unwritten; with a scripted executor the tokens diverge.
    """

    # 1 prompt + 12 generated = 13 tokens = 4 pages of 4.
    def run(pages):
        tokenizer, executor = WordTokenizer(), CountingExecutor(pages=pages)
        engine = s.Engine(config(max_seq_len=256), tokenizer, executor)
        produced = {}
        with running(engine):
            streams = {name: engine.add_request(name, [n + 1], s.GenerateConfig(max_new_tokens=12)) for n, name in enumerate(("r0", "r1"))}
            for name, stream in streams.items():
                produced[name], _ = drain(stream)
            return produced, engine.preemptions, executor.kv_gaps

    reference, none_preempted, clean_gaps = run(64)  # room for both
    contended, some_preempted, gaps = run(5)         # room for one, not two

    assert none_preempted == 0
    assert clean_gaps == []
    assert some_preempted > 0, "the pool was meant to be too small to hold both"
    # The replay must have written every row the next decode attends.
    assert gaps == [], f"decode at a position with unwritten rows before it: {gaps}"
    assert contended == reference


def test_a_python_executor_sees_each_requests_sampling_params():
    tok, ex = WordTokenizer(), CountingExecutor()
    engine = s.Engine(config(), tok, ex)

    with running(engine):
        greedy = s.GenerateConfig(max_new_tokens=2, ignore_eos=True)
        sampled = s.GenerateConfig(max_new_tokens=2, ignore_eos=True, temperature=0.7, top_p=0.95, top_k=40, seed=1234)
        # Both in flight at once: a step batches them, and each item carries its
        # own parameters rather than one batch-wide setting.
        a = engine.add_request("greedy", tok.encode("one two"), greedy)
        b = engine.add_request("sampled", tok.encode("three four"), sampled)
        drain(a)
        drain(b)

    assert ex.sampling["greedy"].is_greedy
    assert ex.sampling["greedy"].temperature == 0.0
    assert ex.sampling["greedy"].top_k is None
    assert ex.sampling["greedy"].seed is None

    seen = ex.sampling["sampled"]
    assert not seen.is_greedy
    assert seen.temperature == pytest.approx(0.7)
    assert seen.top_p == pytest.approx(0.95)
    assert seen.top_k == 40
    assert seen.seed == 1234
    assert repr(seen) == "SamplingParams(temperature=0.7, top_p=0.95, top_k=40, seed=1234)"
