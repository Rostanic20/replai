"""Export captured runs to OpenTelemetry (OTLP/JSON).

Maps replai spans onto the OTel GenAI semantic conventions so a run can be sent
to any OpenTelemetry backend. Emits the plain OTLP/JSON trace structure — no
opentelemetry SDK dependency required.
"""
from __future__ import annotations

# OTel SpanKind: INTERNAL=1, CLIENT=3
_KIND = {"llm_call": 3, "tool_call": 3, "function": 1}


def to_otlp(run: dict, spans: list[dict]) -> dict:
    """Build an OTLP/JSON trace payload for one run and its spans."""
    from . import __version__
    trace_id = _pad(run["id"], 32)
    return {
        "resourceSpans": [{
            "resource": {"attributes": [_attr("service.name", "replai")]},
            "scopeSpans": [{
                "scope": {"name": "replai", "version": __version__},
                "spans": [_span(trace_id, s) for s in spans],
            }],
        }],
    }


def _span(trace_id: str, s: dict) -> dict:
    span = {
        "traceId": trace_id,
        "spanId": _pad(s["id"], 16),
        "name": s["name"],
        "kind": _KIND.get(s["type"], 1),
        "startTimeUnixNano": _nanos(s.get("start")),
        "endTimeUnixNano": _nanos(s.get("end")),
        "attributes": _attributes(s),
    }
    if s.get("parent_id"):
        span["parentSpanId"] = _pad(s["parent_id"], 16)
    if s.get("error"):
        span["status"] = {"code": 2, "message": s["error"]}  # STATUS_CODE_ERROR
    return span


def _attributes(s: dict) -> list[dict]:
    attrs = [_attr("replai.span.type", s["type"])]
    if s["type"] == "llm_call":
        attrs.append(_attr("gen_ai.operation.name", "chat"))
        attrs.append(_attr("gen_ai.system", _system(s["name"])))
        if s.get("model"):
            attrs.append(_attr("gen_ai.request.model", s["model"]))
        if s.get("tokens_in") is not None:
            attrs.append(_attr("gen_ai.usage.input_tokens", s["tokens_in"]))
        if s.get("tokens_out") is not None:
            attrs.append(_attr("gen_ai.usage.output_tokens", s["tokens_out"]))
    elif s["type"] == "tool_call":
        attrs.append(_attr("gen_ai.operation.name", "execute_tool"))
        attrs.append(_attr("gen_ai.tool.name", s["name"]))
    return attrs


def _attr(key: str, value) -> dict:
    if isinstance(value, bool):
        v = {"boolValue": value}
    elif isinstance(value, int):
        v = {"intValue": value}
    else:
        v = {"stringValue": str(value)}
    return {"key": key, "value": v}


def _system(name: str) -> str:
    if "anthropic" in name:
        return "anthropic"
    if "openai" in name:
        return "openai"
    return "unknown"


def _pad(hex_id: str, width: int) -> str:
    return hex_id.rjust(width, "0")[:width]


def _nanos(ts) -> int:
    return int(ts * 1_000_000_000) if ts else 0
