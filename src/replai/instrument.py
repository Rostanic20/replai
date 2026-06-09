"""Best-effort auto-instrumentation of LLM client libraries.

Patches the Anthropic and OpenAI SDK call sites — sync and async, streaming and
non-streaming — so every model call is captured without changing your code,
including the tool calls the model requests. Safe no-op if a library isn't
installed, and the capture wrappers never raise into your actual call.
"""
from __future__ import annotations

import functools
import inspect
import json
import time

from . import context as _ctx

_patched = False


def instrument() -> None:
    global _patched
    if _patched:
        return
    _patch_anthropic()
    _patch_openai()
    _patched = True


def _patch_anthropic() -> None:
    try:
        from anthropic.resources.messages import AsyncMessages, Messages
    except Exception:
        return
    for cls in (Messages, AsyncMessages):
        _patch(cls, "create", "anthropic.messages.create",
               _extract_anthropic, _AnthropicAccum, _reconstruct_anthropic)


def _patch_openai() -> None:
    try:
        from openai.resources.chat.completions import AsyncCompletions, Completions
    except Exception:
        return
    for cls in (Completions, AsyncCompletions):
        _patch(cls, "create", "openai.chat.completions.create",
               _extract_openai, _OpenAIAccum, _reconstruct_openai)


def _patch(cls, attr: str, name: str, extract, make_accum, reconstruct) -> None:
    """Wrap cls.attr so each call is recorded. Handles sync/async + streaming."""
    original = getattr(cls, attr)
    if getattr(original, "_replai", False):
        return

    if inspect.iscoroutinefunction(original):
        @functools.wraps(original)
        async def wrapper(self, *args, **kwargs):
            played = _maybe_replay(name, kwargs, reconstruct)
            if played is not _MISS:
                return played
            start = time.time()
            resp = await original(self, *args, **kwargs)
            if kwargs.get("stream"):
                return _AsyncStreamProxy(resp, name, kwargs, make_accum, start)
            _record(name, kwargs, extract, resp, start)
            return resp
    else:
        @functools.wraps(original)
        def wrapper(self, *args, **kwargs):
            played = _maybe_replay(name, kwargs, reconstruct)
            if played is not _MISS:
                return played
            start = time.time()
            resp = original(self, *args, **kwargs)
            if kwargs.get("stream"):
                return _StreamProxy(resp, name, kwargs, make_accum, start)
            _record(name, kwargs, extract, resp, start)
            return resp

    wrapper._replai = True
    setattr(cls, attr, wrapper)


_MISS = object()


def _maybe_replay(name: str, kwargs: dict, reconstruct):
    """Return a played-back response under replay(), else the _MISS sentinel."""
    action, rec = _ctx.replay_lookup(name, "llm_call")
    if action == "run":
        return _MISS
    if action == "raise":
        raise RuntimeError(
            f"replay: no recorded response for llm_call '{name}'. "
            f"Pass live={{'llm_call'}} to run it live."
        )
    raw = rec.get("raw")
    if raw is None:
        raise RuntimeError(
            f"replay: recorded '{name}' has no raw response (streamed calls "
            f"aren't replayable). Pass live={{'llm_call'}} to run it live."
        )
    _ctx.record_span(
        name, "llm_call",
        input=kwargs.get("messages"), output=rec.get("output"),
        model=rec.get("model"), tokens_in=rec.get("tokens_in"),
        tokens_out=rec.get("tokens_out"), raw=raw,
    )
    return reconstruct(raw)


def _record(name: str, kwargs: dict, extract, resp, start=None) -> None:
    try:
        text, tokens_in, tokens_out = extract(resp)
        _ctx.record_span(
            name, "llm_call",
            start=start,
            input=kwargs.get("messages"),
            output=text,
            model=kwargs.get("model"),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            raw=_raw(resp),
        )
    except Exception:
        pass


def _raw(resp):
    """Serialize an SDK response (pydantic) so replay can reconstruct it."""
    dump = getattr(resp, "model_dump", None)
    if dump is None:
        return None
    try:
        return dump(mode="json")
    except Exception:
        return None


def _reconstruct_anthropic(raw):
    from anthropic.types import Message
    return Message.model_validate(raw)


def _reconstruct_openai(raw):
    from openai.types.chat import ChatCompletion
    return ChatCompletion.model_validate(raw)


# --- streaming -------------------------------------------------------------

class _StreamProxy:
    """Transparent wrapper over a sync stream that records once it's drained."""

    def __init__(self, stream, name, kwargs, make_accum, start=None):
        self._stream = stream
        self._name = name
        self._kwargs = kwargs
        self._accum = make_accum()
        self._start = start
        self._recorded = False

    def __iter__(self):
        try:
            for event in self._stream:
                try:
                    self._accum.feed(event)
                except Exception:
                    pass
                yield event
        finally:
            self._flush()

    def _flush(self):
        if self._recorded:
            return
        self._recorded = True
        _record_accum(self._name, self._kwargs, self._accum, self._start)

    def __enter__(self):
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *exc):
        if hasattr(self._stream, "__exit__"):
            return self._stream.__exit__(*exc)
        return False

    def __getattr__(self, item):
        stream = self.__dict__.get("_stream")
        if stream is None:
            raise AttributeError(item)
        return getattr(stream, item)


class _AsyncStreamProxy:
    """Transparent wrapper over an async stream that records once it's drained."""

    def __init__(self, stream, name, kwargs, make_accum, start=None):
        self._stream = stream
        self._name = name
        self._kwargs = kwargs
        self._accum = make_accum()
        self._start = start
        self._recorded = False

    async def __aiter__(self):
        try:
            async for event in self._stream:
                try:
                    self._accum.feed(event)
                except Exception:
                    pass
                yield event
        finally:
            self._flush()

    def _flush(self):
        if self._recorded:
            return
        self._recorded = True
        _record_accum(self._name, self._kwargs, self._accum, self._start)

    async def __aenter__(self):
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, *exc):
        if hasattr(self._stream, "__aexit__"):
            return await self._stream.__aexit__(*exc)
        return False

    def __getattr__(self, item):
        stream = self.__dict__.get("_stream")
        if stream is None:
            raise AttributeError(item)
        return getattr(stream, item)


def _record_accum(name: str, kwargs: dict, accum, start=None) -> None:
    try:
        text, tokens_in, tokens_out = accum.result()
        _ctx.record_span(
            name, "llm_call",
            start=start,
            input=kwargs.get("messages"),
            output=text,
            model=kwargs.get("model"),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
    except Exception:
        pass


class _AnthropicAccum:
    def __init__(self):
        self._text = []
        self._blocks = {}  # index -> {id, name, json: [partial_json, ...]}
        self.tokens_in = None
        self.tokens_out = None

    def feed(self, event) -> None:
        etype = getattr(event, "type", None)
        if etype == "content_block_start":
            block = getattr(event, "content_block", None)
            if getattr(block, "type", None) == "tool_use":
                self._blocks[getattr(event, "index", None)] = {
                    "id": getattr(block, "id", None),
                    "name": getattr(block, "name", None),
                    "json": [],
                }
        elif etype == "content_block_delta":
            delta = getattr(event, "delta", None)
            text = getattr(delta, "text", None)
            if text:
                self._text.append(text)
            partial = getattr(delta, "partial_json", None)
            if partial:
                block = self._blocks.get(getattr(event, "index", None))
                if block is not None:
                    block["json"].append(partial)
        elif etype == "message_start":
            usage = getattr(getattr(event, "message", None), "usage", None)
            if usage is not None:
                self.tokens_in = getattr(usage, "input_tokens", self.tokens_in)
        elif etype == "message_delta":
            usage = getattr(event, "usage", None)
            if usage is not None:
                self.tokens_out = getattr(usage, "output_tokens", self.tokens_out)

    def result(self):
        tool_calls = [
            {"id": b["id"], "name": b["name"], "input": _parse_json("".join(b["json"]))}
            for b in self._blocks.values()
        ]
        return _combine("".join(self._text), tool_calls), self.tokens_in, self.tokens_out


class _OpenAIAccum:
    def __init__(self):
        self._text = []
        self._tools = {}  # index -> {id, name, args: [fragment, ...]}
        self.tokens_in = None
        self.tokens_out = None

    def feed(self, chunk) -> None:
        choices = getattr(chunk, "choices", None)
        if choices:
            delta = getattr(choices[0], "delta", None)
            content = getattr(delta, "content", None)
            if content:
                self._text.append(content)
            for tc in (getattr(delta, "tool_calls", None) or []):
                slot = self._tools.setdefault(
                    getattr(tc, "index", 0), {"id": None, "name": None, "args": []}
                )
                if getattr(tc, "id", None):
                    slot["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        slot["args"].append(fn.arguments)
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            self.tokens_in = getattr(usage, "prompt_tokens", self.tokens_in)
            self.tokens_out = getattr(usage, "completion_tokens", self.tokens_out)

    def result(self):
        tool_calls = [
            {"id": s["id"], "name": s["name"], "input": _parse_json("".join(s["args"]))}
            for s in self._tools.values()
        ]
        return _combine("".join(self._text), tool_calls), self.tokens_in, self.tokens_out


def _extract_anthropic(resp):
    usage = getattr(resp, "usage", None)
    return (
        _anthropic_output(resp),
        getattr(usage, "input_tokens", None),
        getattr(usage, "output_tokens", None),
    )


def _extract_openai(resp):
    usage = getattr(resp, "usage", None)
    return (
        _openai_output(resp),
        getattr(usage, "prompt_tokens", None),
        getattr(usage, "completion_tokens", None),
    )


def _anthropic_output(resp):
    try:
        text_parts, tool_calls = [], []
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                tool_calls.append({
                    "id": getattr(block, "id", None),
                    "name": getattr(block, "name", None),
                    "input": getattr(block, "input", None),
                })
            else:
                text = getattr(block, "text", None)
                if text:
                    text_parts.append(text)
        return _combine("".join(text_parts), tool_calls)
    except Exception:
        return str(resp)


def _openai_output(resp):
    try:
        msg = resp.choices[0].message
        tool_calls = []
        for tc in (getattr(msg, "tool_calls", None) or []):
            fn = getattr(tc, "function", None)
            tool_calls.append({
                "id": getattr(tc, "id", None),
                "name": getattr(fn, "name", None),
                "input": _parse_json(getattr(fn, "arguments", None)),
            })
        return _combine(getattr(msg, "content", None) or "", tool_calls)
    except Exception:
        return str(resp)


def _combine(text: str, tool_calls: list):
    """A plain string when there are no tool calls, else a structured dict."""
    if tool_calls:
        out = {"tool_calls": tool_calls}
        if text:
            out["text"] = text
        return out
    return text


def _parse_json(raw):
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return raw
