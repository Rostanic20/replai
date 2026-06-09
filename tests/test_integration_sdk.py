"""Validate capture against the real Anthropic/OpenAI SDK objects (not fakes)."""
from replai import instrument as I


def test_extract_real_anthropic_message():
    from anthropic.types import Message
    msg = Message.model_validate({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-4-8",
        "content": [
            {"type": "text", "text": "hello"},
            {"type": "tool_use", "id": "tu_1", "name": "search", "input": {"q": "x"}},
        ],
        "stop_reason": "tool_use", "stop_sequence": None,
        "usage": {"input_tokens": 11, "output_tokens": 5},
    })
    out, tin, tout = I._extract_anthropic(msg)
    assert out["text"] == "hello"
    assert out["tool_calls"] == [{"id": "tu_1", "name": "search", "input": {"q": "x"}}]
    assert (tin, tout) == (11, 5)
    # raw round-trips back into a real Message
    back = I._reconstruct_anthropic(I._raw(msg))
    assert back.content[1].name == "search"


def test_extract_real_openai_chatcompletion():
    from openai.types.chat import ChatCompletion
    cc = ChatCompletion.model_validate({
        "id": "cc_1", "object": "chat.completion", "created": 0, "model": "gpt-4o",
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "search", "arguments": "{\"q\": \"x\"}"}}]}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
    })
    out, tin, tout = I._extract_openai(cc)
    assert out["tool_calls"] == [{"id": "c1", "name": "search", "input": {"q": "x"}}]
    assert (tin, tout) == (3, 4)
    back = I._reconstruct_openai(I._raw(cc))
    assert back.choices[0].message.tool_calls[0].function.name == "search"
