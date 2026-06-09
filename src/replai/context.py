from __future__ import annotations

import contextvars
import time
from typing import Optional

from .models import Run, Span
from .store import Store

_store: Optional[Store] = None
_current_run: contextvars.ContextVar = contextvars.ContextVar("replai_run", default=None)
_span_stack: contextvars.ContextVar = contextvars.ContextVar("replai_span_stack", default=())
_replay: contextvars.ContextVar = contextvars.ContextVar("replai_replay", default=None)
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


# --- replay ---------------------------------------------------------------

class Replay:
    """Playback source for a replayed run: recorded spans queued per (name, type)."""

    def __init__(self, queues: dict, live: set):
        self.queues = queues
        self.live = live


def enter_replay(replay: "Replay"):
    return _replay.set(replay)


def exit_replay(token) -> None:
    _replay.reset(token)


def replay_lookup(name: str, type: str):
    """Decide what a call should do under replay.

    Returns ("run", None) to execute normally, ("play", recorded_span) to play
    back a recorded result, or ("raise", None) when there's no recording and the
    type isn't allowed to run live.
    """
    replay = _replay.get()
    if replay is None or type in replay.live:
        return ("run", None)
    queue = replay.queues.get((name, type))
    if queue:
        return ("play", queue.popleft())
    return ("raise", None)
