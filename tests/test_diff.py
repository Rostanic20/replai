from replai.diff import diff_runs, summarize


def span(name, type="llm_call", **fields):
    base = {"name": name, "type": type, "output": None, "model": None,
            "tokens_in": None, "tokens_out": None, "error": None}
    base.update(fields)
    return base


def test_identical_runs_are_all_same():
    a = [span("think"), span("act", type="tool_call")]
    rows = diff_runs(a, [dict(s) for s in a])
    assert [r["status"] for r in rows] == ["same", "same"]
    assert all(not r["changes"] for r in rows)


def test_changed_field_is_reported():
    a = [span("think", output="hi", tokens_out=5)]
    b = [span("think", output="bye", tokens_out=9)]
    rows = diff_runs(a, b)
    assert rows[0]["status"] == "changed"
    assert rows[0]["changes"]["output"] == ["hi", "bye"]
    assert rows[0]["changes"]["tokens_out"] == [5, 9]


def test_added_and_removed_steps():
    a = [span("think"), span("answer")]
    b = [span("think"), span("search", type="tool_call"), span("answer")]
    rows = diff_runs(a, b)
    assert [r["status"] for r in rows] == ["same", "added", "same"]
    added = rows[1]
    assert added["a"] is None and added["b"]["name"] == "search"


def test_removed_step():
    a = [span("think"), span("search", type="tool_call"), span("answer")]
    b = [span("think"), span("answer")]
    rows = diff_runs(a, b)
    assert [r["status"] for r in rows] == ["same", "removed", "same"]
    assert rows[1]["b"] is None and rows[1]["a"]["name"] == "search"


def test_alignment_keys_on_name_and_type():
    # same name, different type -> not matched
    a = [span("run", type="function")]
    b = [span("run", type="tool_call")]
    rows = diff_runs(a, b)
    assert {r["status"] for r in rows} == {"added", "removed"}


def test_empty_runs():
    assert diff_runs([], []) == []
    assert [r["status"] for r in diff_runs([span("x")], [])] == ["removed"]
    assert [r["status"] for r in diff_runs([], [span("x")])] == ["added"]


def test_cli_diff(tmp_path, capsys):
    import replai
    from replai.cli import main
    db = str(tmp_path / "d.db")
    replai.init(db=db, instrument=False)
    with replai.run("a") as ra:
        replai.record_llm_call(model="m", input="x", output="hello")
    with replai.run("b") as rb:
        replai.record_llm_call(model="m", input="x", output="world")

    main(["diff", ra.id, rb.id, "--db", db])
    out = capsys.readouterr().out
    assert "~ llm_call" in out
    assert "output: hello -> world" in out
    assert "steps 1->1" in out


def test_cli_diff_unknown_run(tmp_path):
    import pytest
    import replai
    from replai.cli import main
    db = str(tmp_path / "u.db")
    replai.init(db=db, instrument=False)
    with replai.run("a") as ra:
        replai.record_llm_call(model="m", input="x", output="y")
    with pytest.raises(SystemExit):
        main(["diff", ra.id, "nope", "--db", db])


def test_summarize():
    spans = [span("a", tokens_in=3, tokens_out=5),
             span("b", tokens_out=2, error="boom")]
    assert summarize(spans) == {"steps": 2, "tokens_in": 3, "tokens_out": 7, "errors": 1}
