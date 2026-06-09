import threading

import replai
from replai.models import Run, Span
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


def test_concurrent_writes(tmp_path):
    db = str(tmp_path / "c.db")
    store = Store(db)
    run = Run(name="threads")
    store.save_run(run)

    def writer(n):
        for i in range(20):
            store.save_span(Span(run_id=run.id, name=f"{n}-{i}"))

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(store.spans(run.id)) == 160


def test_migration_adds_raw_column(tmp_path):
    import sqlite3
    db = str(tmp_path / "old.db")
    conn = sqlite3.connect(db)
    conn.executescript(
        """CREATE TABLE runs (id TEXT PRIMARY KEY, name TEXT, start REAL, end REAL, metadata TEXT);
           CREATE TABLE spans (id TEXT PRIMARY KEY, run_id TEXT, parent_id TEXT, name TEXT,
             type TEXT, start REAL, end REAL, input TEXT, output TEXT, error TEXT, model TEXT,
             tokens_in INTEGER, tokens_out INTEGER, metadata TEXT);"""
    )
    conn.commit()
    conn.close()

    store = Store(db)  # should ALTER TABLE to add the raw column
    run = Run(name="x")
    store.save_run(run)
    store.save_span(Span(run_id=run.id, name="s", type="llm_call", raw={"k": 1}))
    assert store.spans(run.id, with_raw=True)[0]["raw"] == {"k": 1}


def test_span_in_worker_thread_attaches_to_open_run(tmp_path):
    db = str(tmp_path / "wt.db")
    replai.init(db=db, instrument=False)

    with replai.run("main") as r:
        def work():
            with replai.span("threaded-tool", type="tool_call"):
                pass
        t = threading.Thread(target=work)
        t.start()
        t.join()

    store = Store(db)
    assert len(store.runs()) == 1  # no orphan "auto" run
    spans = store.spans(r.id)
    assert [s["name"] for s in spans] == ["threaded-tool"]
