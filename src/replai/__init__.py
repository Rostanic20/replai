"""replai — a local-first, framework-agnostic debugger for LLM agents.

See exactly what your agent did, step by step — then replay and diff runs.
"""
from __future__ import annotations

import contextlib
import functools
import inspect
import time
from collections import deque
from typing import Any, Optional

from . import context as _ctx
from .models import Run, Span
from .store import Store

__version__ = "0.1.0"
__all__ = ["init", "run", "span", "trace", "tool", "record_llm_call", "replay",
           "redact_keys", "Store", "Run", "Span"]


def init(db: Optional[str] = None, instrument: bool = True, redact=None) -> Store:
    """Set up replai. Call once at startup.

    db: path to the local SQLite file (defaults to ~/.replai/replai.db).
    instrument: auto-capture Anthropic & OpenAI client calls.
    redact: optional callable applied to each span's input and output before
        they're written to disk. Use replai.redact_keys(...) or your own.
    """
    store = Store(db, redact=redact)
    _ctx.set_store(store)
    if instrument:
        from .instrument import instrument as _do_instrument
        _do_instrument()
    return store


@contextlib.contextmanager
def run(name: str = "run", **metadata):
    """Open a run — the top-level container for everything an agent does."""
    store = _ctx.get_store()
    r = Run(name=name, metadata=metadata)
    store.save_run(r)
    token = _ctx.enter_run(r)
    try:
        yield r
    finally:
        r.end = time.time()
        store.save_run(r)
        _ctx.exit_run(token)


@contextlib.contextmanager
def span(name: str, type: str = "function", input: Any = None, **metadata):
    """Record a step. Nests automatically inside the current run/span."""
    store = _ctx.get_store()
    r = _ctx.ensure_run()
    sp = Span(
        run_id=r.id, name=name, type=type,
        parent_id=_ctx.current_span_id(), input=input, metadata=metadata,
    )
    store.save_span(sp)
    _ctx.push_span(sp.id)
    try:
        yield sp
    except Exception as exc:
        sp.error = repr(exc)
        raise
    finally:
        sp.end = time.time()
        store.save_span(sp)
        _ctx.pop_span()


def trace(fn=None, *, name: Optional[str] = None, type: str = "function"):
    """Decorator that records a function call as a span (sync or async).

    Under replay() a recorded call is played back instead of executed.
    """
    def deco(f):
        span_name = name or getattr(f, "__name__", "fn")

        if inspect.iscoroutinefunction(f):
            @functools.wraps(f)
            async def awrapper(*args, **kwargs):
                inp = {"args": args, "kwargs": kwargs}
                action, rec = _ctx.replay_lookup(span_name, type)
                if action == "play":
                    return _replay_emit(span_name, type, rec, inp)
                if action == "raise":
                    _replay_raise(span_name, type)
                with span(span_name, type=type, input=inp) as s:
                    out = await f(*args, **kwargs)
                    s.output = out
                    return out
            return awrapper

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            inp = {"args": args, "kwargs": kwargs}
            action, rec = _ctx.replay_lookup(span_name, type)
            if action == "play":
                return _replay_emit(span_name, type, rec, inp)
            if action == "raise":
                _replay_raise(span_name, type)
            with span(span_name, type=type, input=inp) as s:
                out = f(*args, **kwargs)
                s.output = out
                return out
        return wrapper

    return deco(fn) if callable(fn) else deco


def tool(fn=None, *, name: Optional[str] = None):
    """Decorator for tool calls — same as @trace but tagged as a tool."""
    return trace(fn, name=name, type="tool_call")


@contextlib.contextmanager
def replay(run_id: str, live=None):
    """Re-run your agent against a recorded run, played back step by step.

    Instrumented LLM calls and @tool/@trace calls return their recorded results
    instead of executing — deterministic, no network, no token cost. The replay
    is captured as a new run (diff it against the original to see what changed).

    live: a set of span types ("llm_call", "tool_call", "function") to run for
    real instead of playing back. A call with no recorded match (and not in
    `live`) raises rather than silently going live.
    """
    store = _ctx.get_store()
    spans = store.spans(run_id, with_raw=True)
    if not spans and store.run(run_id) is None:
        raise ValueError(f"No run found: {run_id}")
    queues: dict = {}
    for s in spans:
        queues.setdefault((s["name"], s["type"]), deque()).append(s)
    replay_token = _ctx.enter_replay(_ctx.Replay(queues, set(live or ())))

    r = Run(name=f"replay:{run_id[:8]}", metadata={"replay_of": run_id})
    store.save_run(r)
    run_token = _ctx.enter_run(r)
    try:
        yield r
    finally:
        r.end = time.time()
        store.save_run(r)
        _ctx.exit_run(run_token)
        _ctx.exit_replay(replay_token)


def _replay_emit(name: str, type: str, rec: dict, input: Any) -> Any:
    _ctx.record_span(name, type, input=input, output=rec["output"])
    return rec["output"]


def _replay_raise(name: str, type: str) -> None:
    raise RuntimeError(
        f"replay: no recorded response for {type} '{name}'. "
        f"Pass live={{'{type}'}} to run it live."
    )


def redact_keys(*keys: str, mask: str = "***"):
    """Build a redactor that replaces values under the given dict keys.

    Recurses through nested dicts and lists. Pass to replai.init(redact=...).
    """
    keyset = set(keys)

    def scrub(value):
        if isinstance(value, dict):
            return {k: (mask if k in keyset else scrub(v)) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [scrub(v) for v in value]
        return value

    return scrub


def record_llm_call(model: str, input: Any, output: Any, *,
                    tokens_in: Optional[int] = None, tokens_out: Optional[int] = None,
                    name: str = "llm_call") -> Span:
    """Manually record a single LLM call (when not using auto-instrumentation)."""
    return _ctx.record_span(
        name, "llm_call", input=input, output=output, model=model,
        tokens_in=tokens_in, tokens_out=tokens_out,
    )
