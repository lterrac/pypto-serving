"""The HTTP server over an engine, from Python."""

from __future__ import annotations

import importlib
import json
import urllib.request

from stubs import CountingExecutor, WordTokenizer, config, running

s = importlib.import_module("_pypto_serving")


def test_http_server_serves_from_a_python_owned_process():
    tok, ex = WordTokenizer(), CountingExecutor()
    engine = s.Engine(config(), tok, ex)
    server = s.HttpServer(s.ServerConfig(host="127.0.0.1", port=0, model_id="stub"), engine, tok)
    with running(engine, server):
        server.start()
        base = f"http://127.0.0.1:{server.bound_port}"

        with urllib.request.urlopen(f"{base}/health", timeout=10) as r:
            assert r.status == 200

        body = json.dumps({"model": "stub", "prompt": "alpha beta", "max_tokens": 2}).encode()
        req = urllib.request.Request(f"{base}/v1/completions", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            reply = json.loads(r.read())
        assert reply["choices"][0]["finish_reason"] == "length"
        assert reply["usage"]["completion_tokens"] == 2
        # "alpha beta" -> ids [0, 1]; prefill emits 2, decode emits 3; both unseen words.
        assert reply["choices"][0]["text"] == "<2> <3>"
