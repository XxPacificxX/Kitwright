"""Backends with mocked transports: no Ollama server and no Anthropic API calls."""

import urllib.error
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from ketwright.backends import (
    FALLBACK_BETA,
    AnthropicBackend,
    BackendError,
    OllamaBackend,
    Usage,
    anthropic_cost,
    make_backend,
)

# --------------------------------------------------------------------------- Claude


def claude_response(text, stop_reason="end_turn", model="claude-opus-5-5"):
    content = [
        SimpleNamespace(type="thinking", thinking="", signature="sig-abc"),
        SimpleNamespace(type="text", text=text),
    ]
    usage = SimpleNamespace(input_tokens=1000, output_tokens=500,
                            cache_read_input_tokens=200, cache_creation_input_tokens=100)
    return SimpleNamespace(content=content, stop_reason=stop_reason, model=model, usage=usage)


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClient:
    def __init__(self, *responses):
        self.beta = SimpleNamespace(messages=FakeMessages(responses))


def api_error(cls, status):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("boom", response=httpx2.Response(status, request=request), body=None)


def test_claude_request_follows_current_api_shape():
    client = FakeClient(claude_response("hello"))
    conv = AnthropicBackend(client=client).new_conversation("SYS")
    reply = conv.send("hi")
    call = client.beta.messages.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["system"] == "SYS"
    assert call["output_config"] == {"effort": "high"}
    assert call["fallbacks"] == "default" and call["betas"] == [FALLBACK_BETA]
    assert call["cache_control"] == {"type": "ephemeral"}
    assert call["max_tokens"] == 16000
    # Opus 5.5 always uses adaptive thinking and rejects sampling params and budgets.
    for forbidden in ("thinking", "temperature", "top_p", "top_k", "budget_tokens"):
        assert forbidden not in call
    assert reply.text == "hello" and reply.stop_reason == "end_turn"


def test_claude_replays_full_assistant_content_including_thinking():
    first = claude_response("one")
    client = FakeClient(first, claude_response("two"))
    conv = AnthropicBackend(client=client).new_conversation("SYS")
    conv.send("a")
    conv.send("b")
    messages = client.beta.messages.calls[1]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[1]["content"] is first.content  # thinking block kept, unchanged


def test_claude_usage_and_cost():
    conv = AnthropicBackend(client=FakeClient(claude_response("x"))).new_conversation("S")
    usage = conv.send("q").usage
    assert (usage.input_tokens, usage.output_tokens, usage.cache_read_tokens, usage.cache_write_tokens) == (
        1000, 500, 200, 100)
    # 1000*$4 + 500*$20 + 200*$0.20 + 100*$5 per million tokens
    assert usage.cost_usd == pytest.approx(0.01454)
    assert anthropic_cost("claude-opus-5-5", 1_000_000, 0, 0, 0) == pytest.approx(4.0)


def test_claude_refusal_returns_empty_text():
    conv = AnthropicBackend(client=FakeClient(claude_response("partial", stop_reason="refusal"))).new_conversation("S")
    reply = conv.send("q")
    assert reply.stop_reason == "refusal" and reply.text == ""


def test_claude_fallbacks_can_be_turned_off():
    client = FakeClient(claude_response("x"))
    AnthropicBackend(client=client, fallbacks=False, effort="medium").new_conversation("S").send("q")
    call = client.beta.messages.calls[0]
    assert "fallbacks" not in call and "betas" not in call
    assert call["output_config"] == {"effort": "medium"}


@pytest.mark.parametrize(
    ("cls", "status", "fatal"),
    [
        (anthropic.AuthenticationError, 401, True),
        (anthropic.NotFoundError, 404, True),
        (anthropic.RateLimitError, 429, False),
        (anthropic.InternalServerError, 500, False),
    ],
)
def test_claude_errors_map_to_backend_errors(cls, status, fatal):
    conv = AnthropicBackend(client=FakeClient(api_error(cls, status))).new_conversation("S")
    with pytest.raises(BackendError) as info:
        conv.send("q")
    assert info.value.fatal is fatal


# --------------------------------------------------------------------------- Ollama


class FakePost:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, payload, timeout):
        self.calls.append((url, {**payload, "messages": list(payload["messages"])}, timeout))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def ollama_reply(text):
    return {"message": {"role": "assistant", "content": text}, "prompt_eval_count": 120, "eval_count": 30,
            "done_reason": "stop"}


def test_ollama_request_and_history():
    post = FakePost(ollama_reply("first"), ollama_reply("second"))
    conv = OllamaBackend(model="qwen2.5-coder:7b", post=post).new_conversation("SYS")
    r1 = conv.send("hello")
    conv.send("again")
    url, payload, _ = post.calls[0]
    assert url == "http://localhost:11434/api/chat"
    assert payload["model"] == "qwen2.5-coder:7b" and payload["stream"] is False
    assert payload["options"]["temperature"] == 0.0 and payload["options"]["num_ctx"] == 16384
    assert payload["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hello"}]
    roles = [m["role"] for m in post.calls[1][1]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert r1.text == "first"
    assert (r1.usage.input_tokens, r1.usage.output_tokens, r1.usage.cost_usd, r1.usage.calls) == (120, 30, 0.0, 1)


def test_ollama_unreachable_is_fatal():
    conv = OllamaBackend(post=FakePost(urllib.error.URLError("refused"))).new_conversation("S")
    with pytest.raises(BackendError) as info:
        conv.send("q")
    assert info.value.fatal and "Is it running?" in str(info.value)


def test_usage_adds_up():
    total = Usage(input_tokens=1, output_tokens=2, cost_usd=0.5, calls=1) + Usage(input_tokens=3, calls=1)
    assert (total.input_tokens, total.output_tokens, total.cost_usd, total.calls) == (4, 2, 0.5, 2)


def test_make_backend():
    assert make_backend("ollama").model == "qwen2.5-coder:7b"
    assert make_backend("ollama", "llama3.2:3b", num_ctx=8192).options["num_ctx"] == 8192
    with pytest.raises(ValueError):
        make_backend("openai")
