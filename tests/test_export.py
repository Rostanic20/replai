import replai
from replai import context as _ctx
from replai.export import to_otlp
from replai.store import Store


def _attr_map(span):
    return {a["key"]: list(a["value"].values())[0] for a in span["attributes"]}


def _seed(db):
    replai.init(db=db, instrument=False)
    with replai.run("agent") as r:
        _ctx.record_span("anthropic.messages.create", "llm_call",
                         input="hi", output="yo", model="claude-opus-4-8",
                         tokens_in=10, tokens_out=4)
        with replai.span("get_weather", type="tool_call"):
            pass
        try:
            with replai.span("parse", type="function"):
                raise ValueError("bad")
        except ValueError:
            pass
    return r


def test_otlp_structure(tmp_path):
    db = str(tmp_path / "a.db")
    r = _seed(db)
    payload = to_otlp(Store(db).run(r.id), Store(db).spans(r.id))

    scope = payload["resourceSpans"][0]["scopeSpans"][0]
    assert scope["scope"]["name"] == "replai"
    spans = scope["spans"]
    assert len(spans) == 3
    # ids padded to OTel widths, shared trace id
    assert all(len(s["traceId"]) == 32 for s in spans)
    assert all(len(s["spanId"]) == 16 for s in spans)
    assert len({s["traceId"] for s in spans}) == 1


def test_llm_genai_attributes(tmp_path):
    db = str(tmp_path / "b.db")
    r = _seed(db)
    spans = to_otlp(Store(db).run(r.id), Store(db).spans(r.id))["resourceSpans"][0]["scopeSpans"][0]["spans"]
    llm = next(s for s in spans if s["name"] == "anthropic.messages.create")
    attrs = _attr_map(llm)
    assert llm["kind"] == 3  # CLIENT
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.system"] == "anthropic"
    assert attrs["gen_ai.request.model"] == "claude-opus-4-8"
    assert attrs["gen_ai.usage.input_tokens"] == 10
    assert attrs["gen_ai.usage.output_tokens"] == 4


def test_tool_and_error_spans(tmp_path):
    db = str(tmp_path / "c.db")
    r = _seed(db)
    spans = to_otlp(Store(db).run(r.id), Store(db).spans(r.id))["resourceSpans"][0]["scopeSpans"][0]["spans"]

    tool = next(s for s in spans if s["name"] == "get_weather")
    assert _attr_map(tool)["gen_ai.tool.name"] == "get_weather"

    err = next(s for s in spans if s["name"] == "parse")
    assert err["status"]["code"] == 2
    assert "bad" in err["status"]["message"]
    assert err["kind"] == 1  # INTERNAL for plain function


def test_cli_export(tmp_path, capsys):
    import json
    from replai.cli import main
    db = str(tmp_path / "d.db")
    r = _seed(db)
    main(["export", r.id, "--db", db])
    payload = json.loads(capsys.readouterr().out)
    assert payload["resourceSpans"][0]["scopeSpans"][0]["spans"]


def test_cli_export_unknown(tmp_path):
    import pytest
    from replai.cli import main
    db = str(tmp_path / "e.db")
    replai.init(db=db, instrument=False)
    with pytest.raises(SystemExit):
        main(["export", "nope", "--db", db])
