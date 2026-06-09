from types import SimpleNamespace

import pytest

import replai
from replai import instrument as I
from replai.diff import diff_runs
from replai.store import Store


def _real_message():
    from anthropic.types import Message
    return Message.model_validate({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-4-8",
        "content": [
            {"type": "text", "text": "Let me check."},
            {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "Paris"}},
        ],
        "stop_reason": "tool_use", "stop_sequence": None,
        "usage": {"input_tokens": 42, "output_tokens": 18},
    })


def _patch_messages(cls):
    I._patch(cls, "create", "anthropic.messages.create",
             I._extract_anthropic, I._AnthropicAccum, I._reconstruct_anthropic)


# --- raw capture -----------------------------------------------------------

def test_raw_captured_for_real_sdk_response(tmp_path):
    db = str(tmp_path / "a.db")
    replai.init(db=db, instrument=False)
    msg = _real_message()

    class Messages:
        def create(self, **kwargs):
            return msg

    _patch_messages(Messages)
    with replai.run("orig") as orig:
        Messages().create(model="claude-opus-4-8", messages=[{"role": "user", "content": "hi"}])

    with_raw = Store(db).spans(orig.id, with_raw=True)[0]
    assert with_raw["raw"]["content"][0]["text"] == "Let me check."
    assert "raw" not in Store(db).spans(orig.id)[0]  # excluded by default


# --- LLM replay against the real SDK ---------------------------------------

def test_llm_replay_reconstructs_real_sdk_object(tmp_path):
    from anthropic.types import Message
    db = str(tmp_path / "b.db")
    replai.init(db=db, instrument=False)
    msg = _real_message()
    calls = {"n": 0}

    class Messages:
        def create(self, **kwargs):
            calls["n"] += 1
            return msg

    _patch_messages(Messages)
    with replai.run("orig") as orig:
        Messages().create(model="claude-opus-4-8", messages=[{"role": "user", "content": "hi"}])
    assert calls["n"] == 1

    with replai.replay(orig.id) as rep:
        out = Messages().create(model="claude-opus-4-8", messages=[{"role": "user", "content": "hi"}])

    assert calls["n"] == 1  # original create body was NOT re-run
    assert isinstance(out, Message)
    assert out.content[0].text == "Let me check."
    assert out.content[1].name == "get_weather"
    assert out.usage.input_tokens == 42
    assert rep.metadata["replay_of"] == orig.id


def test_replay_then_diff_is_identical(tmp_path):
    db = str(tmp_path / "c.db")
    replai.init(db=db, instrument=False)
    msg = _real_message()

    class Messages:
        def create(self, **kwargs):
            return msg

    _patch_messages(Messages)
    with replai.run("orig") as orig:
        Messages().create(model="claude-opus-4-8", messages=[])
    with replai.replay(orig.id) as rep:
        Messages().create(model="claude-opus-4-8", messages=[])

    rows = diff_runs(Store(db).spans(orig.id), Store(db).spans(rep.id))
    assert [r["status"] for r in rows] == ["same"]


def test_replay_without_raw_raises(tmp_path):
    db = str(tmp_path / "d.db")
    replai.init(db=db, instrument=False)

    class Messages:  # SimpleNamespace has no model_dump -> raw is None (like a stream)
        def create(self, **kwargs):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="hi")],
                                   usage=SimpleNamespace(input_tokens=1, output_tokens=1))

    _patch_messages(Messages)
    with replai.run("orig") as orig:
        Messages().create(model="m", messages=[])

    with replai.replay(orig.id):
        with pytest.raises(RuntimeError, match="no raw"):
            Messages().create(model="m", messages=[])


# --- decorator replay ------------------------------------------------------

def test_tool_replay_plays_back_without_executing(tmp_path):
    db = str(tmp_path / "e.db")
    replai.init(db=db, instrument=False)
    calls = {"n": 0}

    @replai.tool
    def get_weather(city):
        calls["n"] += 1
        return {"temp": 17, "city": city}

    with replai.run("orig") as orig:
        get_weather("Paris")
    assert calls["n"] == 1

    with replai.replay(orig.id):
        out = get_weather("Paris")
    assert calls["n"] == 1  # played back, not executed
    assert out == {"temp": 17, "city": "Paris"}


def test_replay_mismatch_raises(tmp_path):
    db = str(tmp_path / "f.db")
    replai.init(db=db, instrument=False)

    @replai.tool
    def a():
        return 1

    @replai.tool
    def b():
        return 2

    with replai.run("orig") as orig:
        a()

    with replai.replay(orig.id):
        a()  # plays back
        with pytest.raises(RuntimeError, match="no recorded response"):
            b()


def test_replay_live_runs_for_real(tmp_path):
    db = str(tmp_path / "g.db")
    replai.init(db=db, instrument=False)
    calls = {"n": 0}

    @replai.tool
    def t():
        calls["n"] += 1
        return "x"

    with replai.run("orig") as orig:
        t()

    with replai.replay(orig.id, live={"tool_call"}):
        t()
    assert calls["n"] == 2  # ran live instead of playing back


def test_replay_unknown_run_raises(tmp_path):
    db = str(tmp_path / "h.db")
    replai.init(db=db, instrument=False)
    with pytest.raises(ValueError, match="No run found"):
        with replai.replay("nope"):
            pass
