import asyncio
from types import SimpleNamespace

import replai
from replai import instrument as I


# --- fake SDK shapes -------------------------------------------------------

def _anthropic_response(text="hello", tin=5, tout=7):
    return SimpleNamespace(
        content=[SimpleNamespace(text=text)],
        usage=SimpleNamespace(input_tokens=tin, output_tokens=tout),
    )


def _anthropic_stream_events(parts=("he", "llo"), tin=5, tout=7):
    yield SimpleNamespace(type="message_start",
                          message=SimpleNamespace(usage=SimpleNamespace(input_tokens=tin)))
    for p in parts:
        yield SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(text=p))
    yield SimpleNamespace(type="message_delta", usage=SimpleNamespace(output_tokens=tout))
    yield SimpleNamespace(type="message_stop")


def _openai_response(text="hi", pin=3, pout=4):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=pin, completion_tokens=pout),
    )


def _openai_stream_chunks(parts=("h", "i"), pin=3, pout=4):
    for p in parts:
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=p))],
                              usage=None)
    yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))],
                          usage=SimpleNamespace(prompt_tokens=pin, completion_tokens=pout))


def _anthropic_tool_response():
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="let me check"),
            SimpleNamespace(type="tool_use", id="t1", name="get_weather",
                            input={"city": "Paris"}),
        ],
        usage=SimpleNamespace(input_tokens=5, output_tokens=7),
    )


def _anthropic_tool_stream_events():
    yield SimpleNamespace(type="message_start",
                          message=SimpleNamespace(usage=SimpleNamespace(input_tokens=5)))
    yield SimpleNamespace(type="content_block_start", index=0,
                          content_block=SimpleNamespace(type="text"))
    yield SimpleNamespace(type="content_block_delta", index=0,
                          delta=SimpleNamespace(text="check"))
    yield SimpleNamespace(type="content_block_start", index=1,
                          content_block=SimpleNamespace(type="tool_use", id="t1", name="get_weather"))
    yield SimpleNamespace(type="content_block_delta", index=1,
                          delta=SimpleNamespace(partial_json='{"city":'))
    yield SimpleNamespace(type="content_block_delta", index=1,
                          delta=SimpleNamespace(partial_json=' "Paris"}'))
    yield SimpleNamespace(type="message_delta", usage=SimpleNamespace(output_tokens=7))


def _openai_tool_response():
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(
                id="c1", function=SimpleNamespace(name="get_weather", arguments='{"city": "Paris"}'))],
        ))],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4),
    )


def _openai_tool_stream_chunks():
    yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
        content=None,
        tool_calls=[SimpleNamespace(index=0, id="c1",
                                    function=SimpleNamespace(name="get_weather", arguments=""))]))],
        usage=None)
    yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
        content=None,
        tool_calls=[SimpleNamespace(index=0, id=None,
                                    function=SimpleNamespace(name=None, arguments='{"city":'))]))],
        usage=None)
    yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
        content=None,
        tool_calls=[SimpleNamespace(index=0, id=None,
                                    function=SimpleNamespace(name=None, arguments=' "Paris"}'))]))],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4))


class _AsyncIter:
    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        async def gen():
            for x in self._items:
                yield x
        return gen()


# --- non-streaming ---------------------------------------------------------

def test_sync_non_streaming(tmp_path):
    replai.init(db=str(tmp_path / "a.db"), instrument=False)

    class Messages:
        def create(self, **kwargs):
            return _anthropic_response()

    I._patch(Messages, "create", "anthropic.messages.create", I._extract_anthropic, I._AnthropicAccum, I._reconstruct_anthropic)

    with replai.run("t") as r:
        Messages().create(model="claude", messages=[{"role": "user", "content": "hey"}])

    span = _spans(r)[0]
    assert span["type"] == "llm_call"
    assert span["output"] == "hello"
    assert span["model"] == "claude"
    assert span["tokens_in"] == 5
    assert span["tokens_out"] == 7
    assert span["input"] == [{"role": "user", "content": "hey"}]


def test_async_non_streaming(tmp_path):
    replai.init(db=str(tmp_path / "b.db"), instrument=False)

    class AsyncCompletions:
        async def create(self, **kwargs):
            return _openai_response()

    I._patch(AsyncCompletions, "create", "openai.chat.completions.create",
             I._extract_openai, I._OpenAIAccum, I._reconstruct_openai)

    async def go():
        with replai.run("t") as r:
            await AsyncCompletions().create(model="gpt", messages=[{"role": "user", "content": "yo"}])
            return r

    r = asyncio.run(go())
    span = _spans(r)[0]
    assert span["output"] == "hi"
    assert span["model"] == "gpt"
    assert span["tokens_in"] == 3
    assert span["tokens_out"] == 4


# --- streaming -------------------------------------------------------------

def test_sync_streaming_accumulates(tmp_path):
    replai.init(db=str(tmp_path / "c.db"), instrument=False)

    class Messages:
        def create(self, **kwargs):
            return _anthropic_stream_events()

    I._patch(Messages, "create", "anthropic.messages.create", I._extract_anthropic, I._AnthropicAccum, I._reconstruct_anthropic)

    with replai.run("t") as r:
        stream = Messages().create(model="claude", stream=True, messages=[{"role": "user", "content": "hey"}])
        events = list(stream)

    assert len(events) == 5  # passthrough is intact
    span = _spans(r)[0]
    assert span["output"] == "hello"
    assert span["tokens_in"] == 5
    assert span["tokens_out"] == 7


def test_async_streaming_accumulates(tmp_path):
    replai.init(db=str(tmp_path / "d.db"), instrument=False)

    class AsyncCompletions:
        async def create(self, **kwargs):
            return _AsyncIter(_openai_stream_chunks())

    I._patch(AsyncCompletions, "create", "openai.chat.completions.create",
             I._extract_openai, I._OpenAIAccum, I._reconstruct_openai)

    async def go():
        with replai.run("t") as r:
            stream = await AsyncCompletions().create(
                model="gpt", stream=True, messages=[{"role": "user", "content": "yo"}])
            got = [c async for c in stream]
            assert len(got) == 3
            return r

    r = asyncio.run(go())
    span = _spans(r)[0]
    assert span["output"] == "hi"
    assert span["tokens_in"] == 3
    assert span["tokens_out"] == 4


def test_double_patch_is_noop(tmp_path):
    replai.init(db=str(tmp_path / "e.db"), instrument=False)

    class Messages:
        def create(self, **kwargs):
            return _anthropic_response()

    I._patch(Messages, "create", "anthropic.messages.create", I._extract_anthropic, I._AnthropicAccum, I._reconstruct_anthropic)
    first = Messages.create
    I._patch(Messages, "create", "anthropic.messages.create", I._extract_anthropic, I._AnthropicAccum, I._reconstruct_anthropic)
    assert Messages.create is first  # not re-wrapped

    with replai.run("t") as r:
        Messages().create(model="claude", messages=[])
    assert len(_spans(r)) == 1  # recorded exactly once


# --- tool calls ------------------------------------------------------------

def test_anthropic_tool_call_non_streaming(tmp_path):
    replai.init(db=str(tmp_path / "f.db"), instrument=False)

    class Messages:
        def create(self, **kwargs):
            return _anthropic_tool_response()

    I._patch(Messages, "create", "anthropic.messages.create", I._extract_anthropic, I._AnthropicAccum, I._reconstruct_anthropic)

    with replai.run("t") as r:
        Messages().create(model="claude", messages=[])

    out = _spans(r)[0]["output"]
    assert out["text"] == "let me check"
    assert out["tool_calls"] == [{"id": "t1", "name": "get_weather", "input": {"city": "Paris"}}]


def test_anthropic_tool_call_streaming(tmp_path):
    replai.init(db=str(tmp_path / "g.db"), instrument=False)

    class Messages:
        def create(self, **kwargs):
            return _anthropic_tool_stream_events()

    I._patch(Messages, "create", "anthropic.messages.create", I._extract_anthropic, I._AnthropicAccum, I._reconstruct_anthropic)

    with replai.run("t") as r:
        list(Messages().create(model="claude", stream=True, messages=[]))

    out = _spans(r)[0]["output"]
    assert out["text"] == "check"
    assert out["tool_calls"] == [{"id": "t1", "name": "get_weather", "input": {"city": "Paris"}}]


def test_openai_tool_call_non_streaming(tmp_path):
    replai.init(db=str(tmp_path / "h.db"), instrument=False)

    class Completions:
        def create(self, **kwargs):
            return _openai_tool_response()

    I._patch(Completions, "create", "openai.chat.completions.create", I._extract_openai, I._OpenAIAccum, I._reconstruct_openai)

    with replai.run("t") as r:
        Completions().create(model="gpt", messages=[])

    out = _spans(r)[0]["output"]
    assert "text" not in out  # content was None
    assert out["tool_calls"] == [{"id": "c1", "name": "get_weather", "input": {"city": "Paris"}}]


def test_openai_tool_call_streaming(tmp_path):
    replai.init(db=str(tmp_path / "i.db"), instrument=False)

    class Completions:
        def create(self, **kwargs):
            return _openai_tool_stream_chunks()

    I._patch(Completions, "create", "openai.chat.completions.create", I._extract_openai, I._OpenAIAccum, I._reconstruct_openai)

    with replai.run("t") as r:
        list(Completions().create(model="gpt", stream=True, messages=[]))

    span = _spans(r)[0]
    out = span["output"]
    assert out["tool_calls"] == [{"id": "c1", "name": "get_weather", "input": {"city": "Paris"}}]
    assert span["tokens_out"] == 4


def test_duration_reflects_call_latency(tmp_path):
    import time
    replai.init(db=str(tmp_path / "j.db"), instrument=False)

    class Messages:
        def create(self, **kwargs):
            time.sleep(0.02)
            return _anthropic_response()

    I._patch(Messages, "create", "anthropic.messages.create", I._extract_anthropic, I._AnthropicAccum, I._reconstruct_anthropic)

    with replai.run("t") as r:
        Messages().create(model="claude", messages=[])

    assert _spans(r)[0]["duration_ms"] >= 15  # real latency, not ~0


def _spans(run):
    store = replai._ctx.get_store()
    return store.spans(run.id)
