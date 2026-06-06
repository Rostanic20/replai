import replai
from replai.store import Store


def test_run_and_span(tmp_path):
    db = str(tmp_path / "t.db")
    replai.init(db=db, instrument=False)

    with replai.run("t") as r:
        with replai.span("step", type="tool_call", input={"x": 1}) as s:
            s.output = {"y": 2}

    store = Store(db)
    runs = store.runs()
    assert len(runs) == 1
    assert runs[0]["name"] == "t"

    spans = store.spans(r.id)
    assert len(spans) == 1
    assert spans[0]["type"] == "tool_call"
    assert spans[0]["input"] == {"x": 1}
    assert spans[0]["output"] == {"y": 2}
    assert spans[0]["end"] is not None


def test_nested_spans_and_parents(tmp_path):
    db = str(tmp_path / "n.db")
    replai.init(db=db, instrument=False)

    with replai.run("nested") as r:
        with replai.span("outer") as outer:
            with replai.span("inner"):
                pass

    spans = {s["name"]: s for s in Store(db).spans(r.id)}
    assert spans["inner"]["parent_id"] == outer.id
    assert spans["outer"]["parent_id"] is None


def test_error_is_recorded(tmp_path):
    db = str(tmp_path / "e.db")
    replai.init(db=db, instrument=False)

    with replai.run("boom") as r:
        try:
            with replai.span("explodes"):
                raise ValueError("nope")
        except ValueError:
            pass

    spans = Store(db).spans(r.id)
    assert "ValueError" in spans[0]["error"]


def test_record_llm_call(tmp_path):
    db = str(tmp_path / "l.db")
    replai.init(db=db, instrument=False)

    with replai.run("chat") as r:
        replai.record_llm_call(model="m", input="hi", output="hello",
                               tokens_in=2, tokens_out=3)

    span = Store(db).spans(r.id)[0]
    assert span["type"] == "llm_call"
    assert span["model"] == "m"
    assert span["tokens_out"] == 3
