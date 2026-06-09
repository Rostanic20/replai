import replai
from replai.store import Store


def test_no_redactor_stores_verbatim(tmp_path):
    db = str(tmp_path / "a.db")
    replai.init(db=db, instrument=False)
    with replai.run("t") as r:
        with replai.span("s", input={"password": "hunter2"}) as s:
            s.output = "ok"
    span = Store(db).spans(r.id)[0]
    assert span["input"] == {"password": "hunter2"}


def test_redact_keys_masks_nested_values(tmp_path):
    db = str(tmp_path / "b.db")
    replai.init(db=db, instrument=False, redact=replai.redact_keys("password", "api_key"))
    with replai.run("t") as r:
        with replai.span("s", input={"user": "bob", "password": "hunter2",
                                      "nested": {"api_key": "sk-123"}}) as s:
            s.output = {"api_key": "sk-456", "ok": True}

    span = Store(db).spans(r.id)[0]
    assert span["input"] == {"user": "bob", "password": "***", "nested": {"api_key": "***"}}
    assert span["output"] == {"api_key": "***", "ok": True}


def test_redact_keys_recurses_into_lists(tmp_path):
    db = str(tmp_path / "c.db")
    replai.init(db=db, instrument=False, redact=replai.redact_keys("token"))
    with replai.run("t") as r:
        replai.record_llm_call(model="m", input=[{"token": "abc"}, {"keep": 1}], output="x")
    span = Store(db).spans(r.id)[0]
    assert span["input"] == [{"token": "***"}, {"keep": 1}]


def test_failing_redactor_fails_closed(tmp_path):
    db = str(tmp_path / "d.db")

    def boom(_value):
        raise RuntimeError("nope")

    replai.init(db=db, instrument=False, redact=boom)
    with replai.run("t") as r:
        with replai.span("s", input={"secret": "leak"}) as s:
            s.output = "leak too"

    span = Store(db).spans(r.id)[0]
    assert span["input"] == "<redaction failed>"  # raw value never persisted
    assert span["output"] == "<redaction failed>"
