"""`ctx.llm` — the model adapter seam, and the neutral message vocabulary.

A *seam* in dsh's terms has three roles: a **Service Definition** declaring the interface, a
**Service Provider** implementing it, and a **Consumer** using it. This file is the definition.
`plugins/llm_openai.py` and `plugins/llm_anthropic.py` are providers. The agent loop is the
consumer.

All three are required. A "swappable" interface with one implementation is an untested guess;
you find out what actually belongs in the interface only when the second provider disagrees
with the first. That is why this project ships two adapters that speak genuinely different wire
protocols rather than one adapter and a promise.

THE NEUTRAL VOCABULARY

    Everything above the adapter speaks this, never a provider's wire types:

      Message(role="user"|"assistant"|"system"|"tool", content=str,
              tool_calls=[ToolCall], tool_call_id=str|None)
      ToolCall(id, name, arguments: dict)
      Completion(text, tool_calls, finish_reason, raw)

    Note `ToolCall.arguments` is a **dict**. OpenAI sends a JSON *string* and Anthropic sends a
    parsed object; normalizing that difference is the adapter's job, and doing it here is why
    the loop never has to care which provider is mounted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

name = "llm"
inject: list[str] = []


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict  # always parsed -- adapters own the JSON-string problem


@dataclass
class Message:
    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # set on role="tool"
    name: str | None = None


@dataclass
class Completion:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"  # "stop" | "tool_calls" | other, normalized by the adapter
    raw: Any = None  # the untouched provider payload, for debugging


class LlmAdapter:
    """What a provider plugin implements."""

    provider: str = "?"

    def generate(self, messages: list[Message], *, tools: list[dict],
                 model: str, max_tokens: int | None = None) -> Completion:
        raise NotImplementedError


class LlmService:
    def __init__(self) -> None:
        self._adapters: dict[str, LlmAdapter] = {}
        self._default: str | None = None

    def register_adapter(self, routes: list[str], adapter: LlmAdapter) -> Callable[[], None]:
        """Claim one or more provider routes. Duplicates are an error, as in dsh.

        Multi-route registration is all-or-nothing: a partial claim would leave the table in a
        state no disposer could cleanly undo.
        """
        clashes = [r for r in routes if r in self._adapters]
        if clashes:
            raise RuntimeError(f"provider route(s) already registered: {clashes}")
        for route in routes:
            self._adapters[route] = adapter
        if self._default is None:
            self._default = routes[0]

        def dispose() -> None:
            for route in routes:
                if self._adapters.get(route) is adapter:
                    del self._adapters[route]
            if self._default in routes:
                self._default = next(iter(self._adapters), None)

        return dispose

    def providers(self) -> list[str]:
        return list(self._adapters)

    def generate(self, messages: list[Message], *, tools: list[dict], model: str,
                 provider: str | None = None, max_tokens: int | None = None) -> Completion:
        route = provider or self._default
        if route is None:
            raise RuntimeError("no LLM adapter is mounted — check your profile")
        adapter = self._adapters.get(route)
        if adapter is None:
            raise RuntimeError(f"no adapter for provider {route!r} (have: {self.providers()})")
        return adapter.generate(messages, tools=tools, model=model, max_tokens=max_tokens)


def apply(ctx, config=None) -> None:
    ctx.provide("llm", LlmService())
