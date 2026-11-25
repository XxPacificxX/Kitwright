"""Chat backends: local models through Ollama, or Claude through the Anthropic API.

Both expose the same two-call interface used by the agent:

    conv = backend.new_conversation(system_prompt)
    reply = conv.send("user text")      # -> Reply(text, usage, stop_reason)
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, fields
from typing import Any, Callable, Protocol


class BackendError(RuntimeError):
    """A model call failed. `fatal` means the whole run should stop."""

    def __init__(self, message: str, fatal: bool = False):
        super().__init__(message)
        self.fatal = fatal


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0
    calls: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(**{f.name: getattr(self, f.name) + getattr(other, f.name) for f in fields(self)})

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cost_usd"] = round(d["cost_usd"], 6)
        d["seconds"] = round(d["seconds"], 2)
        return d


@dataclass
class Reply:
    text: str
    usage: Usage
    stop_reason: str | None = None


class Conversation(Protocol):
    def send(self, user_text: str) -> Reply: ...


class Backend(Protocol):
    name: str
    model: str

    def new_conversation(self, system: str) -> Conversation: ...


# --------------------------------------------------------------------------- Ollama

PostJSON = Callable[[str, dict, float], dict]


def _http_post_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class OllamaBackend:
    """A local model served by Ollama (https://ollama.com), via its /api/chat endpoint."""

    name = "ollama"

    def __init__(
        self,
        model: str = "qwen2.5-coder:7b",
        host: str = "http://localhost:11434",
        num_ctx: int = 16384,
        num_predict: int = 2048,
        temperature: float = 0.0,
        seed: int = 0,
        timeout: float = 900.0,
        post: PostJSON | None = None,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.options = {"temperature": temperature, "seed": seed, "num_ctx": num_ctx, "num_predict": num_predict}
        self.timeout = timeout
        self._post = post or _http_post_json

    def new_conversation(self, system: str) -> "OllamaConversation":
        return OllamaConversation(self, system)


class OllamaConversation:
    def __init__(self, backend: OllamaBackend, system: str):
        self.backend = backend
        self.messages: list[dict[str, str]] = [{"role": "system", "content": system}]

    def send(self, user_text: str) -> Reply:
        b = self.backend
        self.messages.append({"role": "user", "content": user_text})
        payload = {"model": b.model, "messages": self.messages, "stream": False, "options": b.options}
        start = time.perf_counter()
        try:
            data = b._post(f"{b.host}/api/chat", payload, b.timeout)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:300]
            fatal = exc.code in (400, 404)
            hint = f" Try: ollama pull {b.model}" if exc.code == 404 else ""
            raise BackendError(f"Ollama returned HTTP {exc.code}: {body}.{hint}", fatal=fatal) from exc
        except (urllib.error.URLError, ConnectionError) as exc:
            raise BackendError(f"Could not reach Ollama at {b.host} ({exc}). Is it running?", fatal=True) from exc
        except TimeoutError as exc:
            raise BackendError(f"Ollama did not answer within {b.timeout:.0f}s.") from exc
        content = data.get("message", {}).get("content", "")
        self.messages.append({"role": "assistant", "content": content})
        usage = Usage(
            input_tokens=int(data.get("prompt_eval_count") or 0),
            output_tokens=int(data.get("eval_count") or 0),
            seconds=time.perf_counter() - start,
            calls=1,
        )
        return Reply(content, usage, data.get("done_reason"))


# --------------------------------------------------------------------------- Claude

# USD per million tokens (Anthropic list prices). Cache writes use the 5-minute TTL rate.
PRICES_PER_MTOK: dict[str, dict[str, float]] = {
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_read": 0.20, "cache_write": 5.00},
}
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def anthropic_cost(model: str, input_tokens: int, output_tokens: int, cache_read: int, cache_write: int) -> float:
    price = PRICES_PER_MTOK.get(model, PRICES_PER_MTOK["claude-opus-5-5"])
    return (
        input_tokens * price["input"]
        + output_tokens * price["output"]
        + cache_read * price["cache_read"]
        + cache_write * price["cache_write"]
    ) / 1_000_000


class AnthropicBackend:
    """Claude through the Anthropic Python SDK.

    Claude Opus 5.5 always thinks (adaptive); `effort` sets how hard. Server-side
    refusal fallbacks are on by default (`fallbacks="default"`): if the model
    declines a request for policy reasons, the API retries it on a fallback model
    inside the same call.
    """

    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        effort: str = "high",
        max_tokens: int = 16000,
        fallbacks: bool = True,
        client: Any = None,
    ):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY or an `ant auth login` profile
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.fallbacks = fallbacks

    def new_conversation(self, system: str) -> "AnthropicConversation":
        return AnthropicConversation(self, system)


class AnthropicConversation:
    def __init__(self, backend: AnthropicBackend, system: str):
        self.backend = backend
        self.system = system
        self.messages: list[dict[str, Any]] = []

    def _request(self) -> dict[str, Any]:
        b = self.backend
        kwargs: dict[str, Any] = {
            "model": b.model,
            "max_tokens": b.max_tokens,
            "system": self.system,
            "messages": self.messages,
            "output_config": {"effort": b.effort},
            "cache_control": {"type": "ephemeral"},  # cache the growing conversation prefix
        }
        if b.fallbacks:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    def send(self, user_text: str) -> Reply:
        import anthropic

        b = self.backend
        self.messages.append({"role": "user", "content": user_text})
        start = time.perf_counter()
        try:
            response = b.client.beta.messages.create(**self._request())
        except anthropic.AuthenticationError as exc:
            raise BackendError("Anthropic authentication failed: set ANTHROPIC_API_KEY or run `ant auth login`.",
                               fatal=True) from exc
        except anthropic.PermissionDeniedError as exc:
            raise BackendError(f"Anthropic permission denied: {exc.message}", fatal=True) from exc
        except anthropic.NotFoundError as exc:
            raise BackendError(f"Model or endpoint not found: {exc.message}", fatal=True) from exc
        except anthropic.BadRequestError as exc:
            raise BackendError(f"Bad request: {exc.message}", fatal=True) from exc
        except anthropic.RateLimitError as exc:
            raise BackendError(f"Rate limited after SDK retries: {exc.message}") from exc
        except anthropic.APIStatusError as exc:
            raise BackendError(f"API error {exc.status_code}: {exc.message}", fatal=exc.status_code < 500) from exc
        except anthropic.APIConnectionError as exc:
            raise BackendError(f"Network error talking to the Anthropic API: {exc}") from exc

        # Keep the full content (thinking and fallback blocks included) so later
        # turns replay it unchanged.
        self.messages.append({"role": "assistant", "content": response.content})

        u = response.usage
        input_tokens = int(getattr(u, "input_tokens", 0) or 0)
        output_tokens = int(getattr(u, "output_tokens", 0) or 0)
        cache_read = int(getattr(u, "cache_read_input_tokens", 0) or 0)
        cache_write = int(getattr(u, "cache_creation_input_tokens", 0) or 0)
        usage = Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            cost_usd=anthropic_cost(getattr(response, "model", b.model) or b.model,
                                    input_tokens, output_tokens, cache_read, cache_write),
            seconds=time.perf_counter() - start,
            calls=1,
        )
        if response.stop_reason == "refusal":
            return Reply("", usage, "refusal")
        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        return Reply(text, usage, response.stop_reason)


# --------------------------------------------------------------------------- factory

def make_backend(name: str, model: str | None = None, **options: Any) -> Backend:
    if name == "ollama":
        return OllamaBackend(model=model or "qwen2.5-coder:7b",
                             **{k: v for k, v in options.items() if k in ("host", "num_ctx", "timeout")})
    if name == "anthropic":
        return AnthropicBackend(model=model or "claude-opus-5-5",
                                **{k: v for k, v in options.items() if k in ("effort",)})
    raise ValueError(f"unknown backend {name!r} (use 'ollama' or 'anthropic')")
