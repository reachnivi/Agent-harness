"""`ctx.llm` — the model adapter seam, the neutral vocabulary, and the streaming protocol.

A *seam* has three roles: a **Service Definition** declaring the interface, a **Service
Provider** implementing it, and a **Consumer** using it. This file is the definition;
`plugins/llm_openai.py` and `plugins/llm_anthropic.py` are providers; the agent loop consumes.

All three are required. A "swappable" interface with one implementation is an untested guess —
you find out what actually belongs in the interface only when the second provider disagrees
with the first. That is why this project ships two adapters over genuinely different wire
protocols rather than one adapter and a promise.

THE NEUTRAL VOCABULARY

    Everything above the adapter speaks this, never a provider's wire types:

      Message(role, content, tool_calls, tool_call_id)
      ToolCall(id, name, arguments: dict)
      Completion(text, tool_calls, finish_reason, usage, raw)

    `ToolCall.arguments` is always a **dict**. OpenAI sends a JSON string and Anthropic sends a
    parsed object; normalizing that is the adapter's job, which is why the loop never cares.

THE STREAMING PROTOCOL (`StreamChunk`)

    Ported near-verbatim from dsh, because its shape encodes hard-won constraints:

      block-start      {index, block_type}
      text-delta       {index, text}
      reasoning-delta  {index, text}
      tool-call-delta  {index, id, name?, arguments_delta}
      block-end        {index, block}
      usage            {usage}
      finish           {reason}

    Three rules an adapter must obey, each fixing a real failure:

      1. `index` ties a delta to its block, assigned in first-seen order and reused for every
         delta of that block. A response interleaves text and several tool calls; without an
         index you cannot tell whose delta you are holding.
      2. Tool arguments stay **raw JSON strings** all the way through, streamed as
         `arguments_delta` fragments. They are only parsed once complete. Parsing a fragment
         is how you get a mysterious JSONDecodeError halfway through a reply.
      3. Emit `usage` BEFORE `finish`, and nothing at all after `finish`. Consumers treat
         `finish` as terminal, so a late `usage` is silently dropped.

    The point of a raw chunk protocol is that a *UI* wants deltas and the *loop* wants a
    finished message. Both read the same stream: the UI consumes chunks directly, the loop
    folds them with `BlockAssembler`. Without the shared fold, every consumer reimplements
    delta accumulation and they disagree in different ways.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from dshpy.cancel import NEVER, CancelToken

name = "llm"
inject: list[str] = []


# --- the neutral vocabulary -----------------------------------------------------------------


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict  # always parsed -- adapters own the JSON-string problem


@dataclass
class Message:
    role: str  # user | assistant | system | tool
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # set on role="tool"
    name: str | None = None


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(self.input_tokens + other.input_tokens,
                     self.output_tokens + other.output_tokens)

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class Completion:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"  # stop | tool_calls | length | error | aborted
    usage: Usage = field(default_factory=Usage)
    raw: Any = None
    reasoning: str = ""
    error: str | None = None  # set when finish_reason is "error" or "aborted"


# --- the streaming protocol -----------------------------------------------------------------


@dataclass
class BlockStart:
    index: int
    block_type: str  # text | reasoning | tool-call
    type: str = "block-start"


@dataclass
class TextDelta:
    index: int
    text: str
    type: str = "text-delta"


@dataclass
class ReasoningDelta:
    index: int
    text: str
    type: str = "reasoning-delta"


@dataclass
class ToolCallDelta:
    index: int
    id: str
    arguments_delta: str = ""
    name: str | None = None
    type: str = "tool-call-delta"


@dataclass
class BlockEnd:
    index: int
    type: str = "block-end"


@dataclass
class UsageChunk:
    usage: Usage
    type: str = "usage"


@dataclass
class Finish:
    reason: str  # stop | tool_calls | length | error | aborted
    error: str | None = None
    type: str = "finish"


StreamChunk = (BlockStart | TextDelta | ReasoningDelta | ToolCallDelta
               | BlockEnd | UsageChunk | Finish)


class BlockAssembler:
    """Folds a `StreamChunk` stream back into a `Completion`. The one shared implementation.

    dsh makes a point of shipping exactly one of these, and the reason is worth stating: delta
    accumulation looks trivial and has several independently-wrong implementations. Fragments
    must be concatenated in arrival order, per index; a tool call's name may arrive only on its
    first delta; arguments must be parsed once at the end, never per fragment.

    Write it once, test it once, and every consumer gets the same answer.
    """

    def __init__(self) -> None:
        self._text: dict[int, list[str]] = {}
        self._reasoning: dict[int, list[str]] = {}
        self._calls: dict[int, dict] = {}
        self._order: list[int] = []
        self.usage = Usage()
        self.finish_reason = "stop"
        self.error: str | None = None
        self.finished = False

    def push(self, chunk: StreamChunk) -> None:
        if self.finished:
            # Rule 3: nothing after finish. Dropping rather than raising keeps a slightly
            # misbehaving adapter usable, but it is silent, so adapters are tested for it.
            return

        if isinstance(chunk, BlockStart):
            if chunk.index not in self._order:
                self._order.append(chunk.index)

        elif isinstance(chunk, TextDelta):
            self._text.setdefault(chunk.index, []).append(chunk.text)

        elif isinstance(chunk, ReasoningDelta):
            self._reasoning.setdefault(chunk.index, []).append(chunk.text)

        elif isinstance(chunk, ToolCallDelta):
            call = self._calls.setdefault(chunk.index, {"id": "", "name": "", "args": []})
            if chunk.id:
                call["id"] = chunk.id
            if chunk.name:  # often present only on the first delta
                call["name"] = chunk.name
            if chunk.arguments_delta:
                call["args"].append(chunk.arguments_delta)
            if chunk.index not in self._order:
                self._order.append(chunk.index)

        elif isinstance(chunk, UsageChunk):
            self.usage = chunk.usage

        elif isinstance(chunk, Finish):
            self.finish_reason = chunk.reason
            self.error = chunk.error
            self.finished = True

    def result(self) -> Completion:
        text = "".join("".join(parts) for _, parts in sorted(self._text.items()))
        reasoning = "".join("".join(p) for _, p in sorted(self._reasoning.items()))

        calls = []
        for index in self._order:
            call = self._calls.get(index)
            if call is None:
                continue
            calls.append(ToolCall(
                id=call["id"],
                name=call["name"],
                # Rule 2: parse ONCE, here, when the string is complete.
                arguments=parse_arguments("".join(call["args"])),
            ))

        reason = self.finish_reason
        if reason == "stop" and calls:
            # Some providers report "stop" while still emitting calls. The loop keys off this
            # value, so normalize it rather than making every consumer re-derive it.
            reason = "tool_calls"

        return Completion(text=text, tool_calls=calls, finish_reason=reason,
                          usage=self.usage, reasoning=reasoning, error=self.error)


def parse_arguments(raw: str | dict | None) -> dict:
    """Parse tool-call arguments defensively.

    The model writes this string token by token, so truncation and malformed JSON are ordinary
    outcomes, not exceptional ones. Raising would kill the loop; instead we return a marker the
    tool will reject, which becomes an error result the model can read and retry from.

    Some OpenAI-compatible servers (Ollama, for some models) return an already-parsed object,
    so accept that too rather than assuming the spec is followed.
    """
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"__malformed__": raw}
    return parsed if isinstance(parsed, dict) else {"__malformed__": raw}


# --- the adapter contract --------------------------------------------------------------------


@dataclass
class GenerateOptions:
    """One fully-assembled model call. Adapters receive exactly this."""

    messages: list[Message]
    tools: list[dict] = field(default_factory=list)
    model: str = ""
    max_tokens: int | None = None
    token: CancelToken = NEVER


class LlmAdapter:
    """What a provider plugin implements.

    Only `stream()` is required. `generate()` is provided here as a fold over the stream, so
    every adapter gets the non-streaming path for free and the two can never disagree — which
    they will, if you let an adapter implement both.
    """

    provider: str = "?"
    retry_policy: dict | None = None  # read by plugins/llm_retry.py; see that file

    def stream(self, options: GenerateOptions) -> Iterator[StreamChunk]:
        raise NotImplementedError

    def generate(self, options: GenerateOptions) -> Completion:
        assembler = BlockAssembler()
        for chunk in self.stream(options):
            assembler.push(chunk)
        return assembler.result()


class LlmService:
    def __init__(self, ctx) -> None:
        self._ctx = ctx
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

    def adapter(self, provider: str | None = None) -> LlmAdapter:
        route = provider or self._default
        if route is None:
            raise RuntimeError("no LLM adapter is mounted — check your profile")
        adapter = self._adapters.get(route)
        if adapter is None:
            raise RuntimeError(f"no adapter for provider {route!r} (have: {self.providers()})")
        return adapter

    def stream(self, options: GenerateOptions,
               provider: str | None = None) -> Iterator[StreamChunk]:
        """Stream a completion, through the `llm/stream` waterfall.

        The waterfall is why retry and token metering are plugins rather than special cases in
        this method. A listener receives `(options, next)` and returns an iterator: it can
        wrap the stream, count what passes through it, or replace it entirely with a retried
        attempt — all without this service knowing any of those features exist.
        """
        adapter = self.adapter(provider)

        def call(opts: GenerateOptions, adpt: LlmAdapter) -> Iterator[StreamChunk]:
            return adpt.stream(opts)  # raw: may raise, and inner listeners need to see that

        # The adapter travels as an event argument so a listener can read its retry_policy
        # without reaching back into this service.
        chain = self._ctx.waterfall("llm/stream", options, adapter, final=call)

        # Normalization is the OUTERMOST layer, wrapping the fully-assembled chain.
        #
        # This ordering is load-bearing and I got it wrong first: with normalization on the
        # inside, an adapter exception became a `finish{error}` chunk before `llm_retry` ever
        # saw it, so retry silently never fired. Inner listeners must see real exceptions to
        # classify them (a 503 is retryable, a 400 is not); only the consumer at the very top
        # wants the single normalized failure shape.
        return _normalize_failures(chain)

    def generate(self, messages: list[Message], *, tools: list[dict], model: str,
                 provider: str | None = None, max_tokens: int | None = None,
                 token: CancelToken | None = None) -> Completion:
        """Non-streaming convenience: fold the stream. The agent loop uses this."""
        options = GenerateOptions(messages=messages, tools=tools, model=model,
                                  max_tokens=max_tokens, token=token or NEVER)
        assembler = BlockAssembler()
        for chunk in self.stream(options, provider):
            assembler.push(chunk)
        return assembler.result()


def _normalize_failures(chain: Iterator[StreamChunk]) -> Iterator[StreamChunk]:
    """Turn a raising stream into a terminal `finish` chunk.

    dsh's rule: an adapter may throw (transport/protocol failure) or end the stream with
    `finish{error|aborted}` (in-band provider failure), and the *runtime* normalizes the first
    into the second before any consumer sees it. So consumers handle exactly one failure shape
    instead of two, and a half-streamed response still yields the text it produced before
    dying — which a bare exception would throw away.
    """
    from dshpy.cancel import Cancelled

    try:
        yield from chain
    except Cancelled as exc:
        yield Finish(reason="aborted", error=exc.reason)
    except Exception as exc:  # noqa: BLE001 - deliberately broad; see docstring
        yield Finish(reason="error", error=f"{type(exc).__name__}: {exc}")


def apply(ctx, config=None) -> None:
    ctx.provide("llm", LlmService(ctx))
