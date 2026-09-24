"""A whitespace tokenizer and a counting executor for testing the engine from Python."""

from __future__ import annotations

import contextlib
import faulthandler
import importlib

s = importlib.import_module("_pypto_serving")

# A GIL mistake in the bindings shows up as a hang, not a failure. This turns a
# hang into a traceback and a failed run; it fires from a C thread, so it works
# even when the main thread is stuck inside a binding.
faulthandler.dump_traceback_later(120, exit=True)


class WordTokenizer(s.TokenizerAdapter):
    """Whitespace tokens, ids assigned on first sight."""

    def __init__(self):
        super().__init__()
        self.words: list[str] = []
        self.ids: dict[str, int] = {}

    def encode(self, text):
        out = []
        for word in text.split():
            if word not in self.ids:
                self.ids[word] = len(self.words)
                self.words.append(word)
            out.append(self.ids[word])
        return out

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(self.words[i] if i < len(self.words) else f"<{i}>" for i in token_ids)


class CountingExecutor(s.ModelExecutor):
    """Prefill echoes the prompt's last token; decode emits last_token + 1."""

    def __init__(self, pages=64, fail_on_token=None):
        super().__init__()
        self.pages = pages
        self.fail_on_token = fail_on_token
        self.registered = 0
        self.steps = 0
        self.closed = False
        #: Decodes asked for at a position with unwritten rows before it.
        self.kv_gaps: list = []
        self._written: dict = {}
        #: Fail the step rather than let a livelocking schedule hang the suite.
        self.max_steps = 100_000

    def register_model(self):
        self.registered += 1
        return self.pages

    def execute_step(self, command):
        self.steps += 1
        if self.steps > self.max_steps:
            raise RuntimeError(f"executor gave up after {self.max_steps} steps: the schedule is not making progress")
        new = {}
        for item in command.prefill:
            # A prefill writes the KV rows for the positions it feeds.
            self._written.setdefault(item.request_id, set()).update(
                range(item.num_computed_tokens, item.num_computed_tokens + len(item.chunk_tokens))
            )
            if item.num_computed_tokens + len(item.chunk_tokens) >= item.sample_at_length:
                new[item.request_id] = [item.chunk_tokens[-1] + 1]
        for item in command.decode:
            if item.last_token == self.fail_on_token:
                raise RuntimeError("the model fell over")
            # A decode feeds one token at seq_len - 1 and attends everything
            # before it, so every earlier row must already be written. A gap is
            # the kernel reading stale pages, which no token comparison against
            # a stub would reveal.
            written = self._written.setdefault(item.request_id, set())
            position = item.seq_len - 1
            missing = set(range(position)) - written
            if missing:
                self.kv_gaps.append((item.request_id, position, sorted(missing)))
            written.add(position)
            new[item.request_id] = [item.last_token + 1]
        return s.StepResult(new_tokens=new)

    def close(self):
        self.closed = True


def config(max_seq_len=64):
    cfg = s.EngineConfig()
    cfg.runtime = s.RuntimeConfig(page_size=4, max_batch_size=4, max_seq_len=max_seq_len)
    cfg.scheduler = s.SchedulerConfig(max_seq_len=max_seq_len, max_num_running_reqs=4)
    return cfg


@contextlib.contextmanager
def running(engine, *servers):
    """Start the engine; stop it (and any servers) however the block exits."""
    engine.start()
    try:
        yield engine
    finally:
        for server in servers:
            server.stop()
        engine.stop()


def drain(stream):
    updates = list(stream)
    assert updates, "a request must produce at least its final update"
    assert updates[-1].finished
    return [u.token_id for u in updates if u.token_id is not None], updates[-1]
