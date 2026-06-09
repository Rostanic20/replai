"""replai — a local-first, framework-agnostic debugger for LLM agents.

See exactly what your agent did, step by step — then replay and diff runs.
"""
from __future__ import annotations

import contextlib
import functools
import inspect
import time
from typing import Any, Optional

from . import context as _ctx
from .models import Run, Span
from .store import Store

__version__ = "0.1.0"
__all__ = ["init", "run", "span", "trace", "tool", "record_llm_call", "Store", "Run", "Span"]


def init(db: Optional[str] = None, instrument: bool = True) -> Store:
    """Set up replai. Call once at startup.

    db: path to the local SQLite file (defaults to ~/.replai/replai.db).
    instrument: auto-capture Anthropic & OpenAI client calls.
    """
    store = Store(db)
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
    """Decorator that records a function call as a span (sync or async)."""
    def deco(f):
        span_name = name or getattr(f, "__name__", "fn")

        if inspect.iscoroutinefunction(f):
            @functools.wraps(f)
            async def awrapper(*args, **kwargs):
                with span(span_name, type=type, input={"args": args, "kwargs": kwargs}) as s:
                    out = await f(*args, **kwargs)
                    s.output = out
                    return out
            return awrapper

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            with span(span_name, type=type, input={"args": args, "kwargs": kwargs}) as s:
                out = f(*args, **kwargs)
                s.output = out
                return out
        return wrapper

    return deco(fn) if callable(fn) else deco


def tool(fn=None, *, name: Optional[str] = None):
    """Decorator for tool calls — same as @trace but tagged as a tool."""
    return trace(fn, name=name, type="tool_call")


def record_llm_call(model: str, input: Any, output: Any, *,
                    tokens_in: Optional[int] = None, tokens_out: Optional[int] = None,
                    name: str = "llm_call") -> Span:
    """Manually record a single LLM call (when not using auto-instrumentation)."""
    return _ctx.record_span(
        name, "llm_call", input=input, output=output, model=model,
        tokens_in=tokens_in, tokens_out=tokens_out,
    )
