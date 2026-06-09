from __future__ import annotations

import contextvars
import time
from typing import Optional

from .models import Run, Span
from .store import Store

_store: Optional[Store] = None
_current_run: contextvars.ContextVar = contextvars.ContextVar("replai_run", default=None)
_span_stack: contextvars.ContextVar = contextvars.ContextVar("replai_span_stack", default=())
_last_run: Optional[Run] = None  # cross-thread fallback (ContextVars don't cross threads)


def set_store(store: Store) -> None:
    global _store
    _store = store


def get_store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store


def current_run() -> Optional[Run]:
    return _current_run.get()


def enter_run(run: Run):
    global _last_run
    _last_run = run
    return _current_run.set(run)


def exit_run(token) -> None:
    _current_run.reset(token)


def ensure_run() -> Run:
    run = current_run()
    if run is not None:
        return run
    # No run in this context — e.g. a worker thread that didn't inherit the
    # caller's ContextVars. Attach to the most recent still-open run rather
    # than silently starting an orphan "auto" run.
    if _last_run is not None and _last_run.end is None:
        return _last_run
    run = Run(name="auto")
    get_store().save_run(run)
    enter_run(run)
    return run


def current_span_id() -> Optional[str]:
    stack = _span_stack.get()
    return stack[-1] if stack else None


def push_span(span_id: str) -> None:
    _span_stack.set(_span_stack.get() + (span_id,))


def pop_span() -> None:
    stack = _span_stack.get()
    if stack:
        _span_stack.set(stack[:-1])


def record_span(name: str, type: str, start: Optional[float] = None, **fields) -> Span:
    """Record an already-completed span (used for one-shot LLM/tool calls).

    Pass `start` to preserve the real call latency; otherwise duration is ~0.
    """
    run = ensure_run()
    span = Span(run_id=run.id, name=name, type=type, parent_id=current_span_id(), **fields)
    if start is not None:
        span.start = start
    span.end = time.time()
    get_store().save_span(span)
    return span
