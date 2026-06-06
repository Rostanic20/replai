"""Best-effort auto-instrumentation of LLM client libraries.

Patches the Anthropic and OpenAI SDK call sites so every model call is captured
without changing your code. Safe no-op if a library isn't installed, and the
capture wrappers never raise into your actual call.
"""
from __future__ import annotations

import functools

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
        from anthropic.resources.messages import Messages
    except Exception:
        return
    if getattr(Messages.create, "_replai", False):
        return
    original = Messages.create

    @functools.wraps(original)
    def create(self, *args, **kwargs):
        resp = original(self, *args, **kwargs)
        try:
            usage = getattr(resp, "usage", None)
            _ctx.record_span(
                "anthropic.messages.create", "llm_call",
                input=kwargs.get("messages"),
                output=_anthropic_text(resp),
                model=kwargs.get("model"),
                tokens_in=getattr(usage, "input_tokens", None),
                tokens_out=getattr(usage, "output_tokens", None),
            )
        except Exception:
            pass
        return resp

    create._replai = True
    Messages.create = create


def _patch_openai() -> None:
    try:
        from openai.resources.chat.completions import Completions
    except Exception:
        return
    if getattr(Completions.create, "_replai", False):
        return
    original = Completions.create

    @functools.wraps(original)
    def create(self, *args, **kwargs):
        resp = original(self, *args, **kwargs)
        try:
            usage = getattr(resp, "usage", None)
            _ctx.record_span(
                "openai.chat.completions.create", "llm_call",
                input=kwargs.get("messages"),
                output=_openai_text(resp),
                model=kwargs.get("model"),
                tokens_in=getattr(usage, "prompt_tokens", None),
                tokens_out=getattr(usage, "completion_tokens", None),
            )
        except Exception:
            pass
        return resp

    create._replai = True
    Completions.create = create


def _anthropic_text(resp) -> str:
    try:
        return "".join(getattr(block, "text", "") for block in resp.content)
    except Exception:
        return str(resp)


def _openai_text(resp):
    try:
        return resp.choices[0].message.content
    except Exception:
        return str(resp)
