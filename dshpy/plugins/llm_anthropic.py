"""Anthropic Messages wire format, over raw HTTP. Streaming and non-streaming. Still no SDK.

Works against Ollama's Anthropic-compatible endpoint (v0.14+) and, with a real key, against
`https://api.anthropic.com`.

This adapter exists to prove the seam. Mount it instead of `llm_openai` and the same loop, the
same tools and the same policy plugins run over a completely different protocol, with no change
above this file. If that ever required one conditional upstream, the abstraction would be
wrong — which is why `tests/test_adapters.py` asserts it rather than trusting it.

HOW THIS PROTOCOL DIFFERS FROM OPENAI'S

    | | OpenAI / DeepSeek | Anthropic |
    |---|---|---|
    | endpoint     | `/v1/chat/completions`        | `/v1/messages` |
    | tool schema  | `{type:"function",function:…}` | `{name, description, input_schema}` |
    | system       | a `{role:"system"}` message    | a top-level `system` parameter |
    | model asks   | `message.tool_calls[]`         | a `tool_use` **content block** |
    | arguments    | a JSON **string**              | a parsed **dict** (`input`) |
    | the signal   | `finish_reason=="tool_calls"`  | `stop_reason=="tool_use"` |
    | results back | one `{role:"tool"}` PER call   | ONE user message with ALL blocks |
    | max_tokens   | optional                        | **required** |
    | auth         | `Authorization: Bearer`         | `x-api-key` + `anthropic-version` |

    The results row is the one to stare at: the same rule inverted, and both protocols degrade
    quietly rather than erroring when you get it wrong.

THE SSE DIALECT — genuinely different framing, same neutral chunks

    Where OpenAI sends anonymous frames of partial messages, Anthropic sends **named events**
    describing an explicit block lifecycle:

        event: message_start          {"message":{"usage":{"input_tokens":12}}}
        event: content_block_start    {"index":0,"content_block":{"type":"text"}}
        event: content_block_delta    {"index":0,"delta":{"type":"text_delta","text":"He"}}
        event: content_block_start    {"index":1,"content_block":{"type":"tool_use",
                                       "id":"toolu_1","name":"calculate"}}
        event: content_block_delta    {"index":1,"delta":{"type":"input_json_delta",
                                       "partial_json":"{\\"expr"}}
        event: content_block_stop     {"index":1}
        event: message_delta          {"delta":{"stop_reason":"tool_use"},
                                       "usage":{"output_tokens":9}}
        event: message_stop

    Note it already carries `index` and an explicit start/stop per block — so this dialect maps
    almost one-to-one onto the neutral protocol, while OpenAI's needs index allocation invented
    for it. That asymmetry is exactly why the neutral protocol looks the way it does: it was
    shaped by the harder of the two.

    Also note the tool name arrives on `content_block_start`, not on the deltas — and that
    `input_json_delta.partial_json` is a raw string fragment, same as OpenAI's. Both providers
    stream tool arguments as text; neither streams a structured object.
"""

from __future__ import annotations

import json
from typing import Iterator

from dshpy.services.llm import (
    BlockEnd,
    BlockStart,
    Finish,
    GenerateOptions,
    LlmAdapter,
    Message,
    StreamChunk,
    TextDelta,
    ToolCallDelta,
    Usage,
    UsageChunk,
)
from dshpy.sse import parse_sse
from dshpy.transport import post_json, post_stream

name = "llm-anthropic"
inject = ["llm"]

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 1024  # required by this API, so we must always send something

_STOP_REASONS = {
    "tool_use": "tool_calls",
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
}


class AnthropicAdapter(LlmAdapter):
    provider = "anthropic"

    def __init__(self, base_url: str, api_key: str, timeout: float = 120.0,
                 retry_policy: dict | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.retry_policy = retry_policy

    # --- outbound: neutral -> wire ---------------------------------------------------------

    @staticmethod
    def _encode_tools(tools: list[dict]) -> list[dict]:
        return [{"name": t["name"], "description": t["description"],
                 "input_schema": t["parameters"]} for t in tools]

    @staticmethod
    def _encode_messages(messages: list[Message]) -> tuple[str | None, list[dict]]:
        """Returns (system_prompt, messages).

        The system prompt is extracted because this API takes it as a top-level parameter
        rather than as a message — a structural difference, not a naming one.
        """
        system: str | None = None
        out: list[dict] = []
        pending_results: list[dict] = []

        def flush_results() -> None:
            # Rule 1, this protocol's version: ALL results in ONE user message.
            if pending_results:
                out.append({"role": "user", "content": list(pending_results)})
                pending_results.clear()

        for msg in messages:
            if msg.role == "system":
                system = msg.content
                continue

            if msg.role == "tool":
                pending_results.append({"type": "tool_result",
                                        "tool_use_id": msg.tool_call_id,
                                        "content": msg.content})
                continue

            flush_results()

            if msg.role == "assistant":
                blocks: list[dict] = []
                if msg.content:
                    blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name,
                                   "input": tc.arguments})  # a dict here, not a string
                out.append({"role": "assistant", "content": blocks or ""})
            else:
                out.append({"role": "user", "content": msg.content})

        flush_results()
        return system, out

    def _payload(self, options: GenerateOptions, stream: bool) -> dict:
        system, encoded = self._encode_messages(options.messages)
        payload: dict = {
            "model": options.model,
            "messages": encoded,
            "max_tokens": options.max_tokens or DEFAULT_MAX_TOKENS,  # required, unlike OpenAI
        }
        if system:
            payload["system"] = system
        if options.tools:
            payload["tools"] = self._encode_tools(options.tools)
        if stream:
            payload["stream"] = True
        return payload

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key, "anthropic-version": ANTHROPIC_VERSION}

    # --- streaming ---------------------------------------------------------------------------

    def stream(self, options: GenerateOptions) -> Iterator[StreamChunk]:
        byte_chunks = post_stream(
            f"{self.base_url}/v1/messages",
            self._payload(options, stream=True),
            headers=self._headers(),
            timeout=self.timeout,
            token=options.token,
        )

        call_ids: dict[int, str] = {}   # block index -> tool_use id, for later deltas
        usage = Usage()
        finish_reason = "stop"

        # This dialect never sends OpenAI's [DONE]; it ends with `message_stop`.
        for event in parse_sse(byte_chunks, done_sentinel="\x00never\x00"):
            try:
                frame = json.loads(event.data)
            except json.JSONDecodeError:
                continue

            kind = event.event or frame.get("type", "")

            if kind == "message_start":
                message = frame.get("message") or {}
                usage = Usage(
                    input_tokens=(message.get("usage") or {}).get("input_tokens", 0),
                    output_tokens=(message.get("usage") or {}).get("output_tokens", 0),
                )

            elif kind == "content_block_start":
                index = frame.get("index", 0)
                block = frame.get("content_block") or {}
                if block.get("type") == "tool_use":
                    call_ids[index] = block.get("id", "")
                    yield BlockStart(index=index, block_type="tool-call")
                    # The NAME arrives here, not on the deltas -- emit it immediately so a
                    # consumer can show "calling calculate..." before any argument exists.
                    yield ToolCallDelta(index=index, id=block.get("id", ""),
                                        name=block.get("name"), arguments_delta="")
                else:
                    yield BlockStart(index=index, block_type="text")

            elif kind == "content_block_delta":
                index = frame.get("index", 0)
                delta = frame.get("delta") or {}
                if delta.get("type") == "text_delta":
                    yield TextDelta(index=index, text=delta.get("text", ""))
                elif delta.get("type") == "input_json_delta":
                    yield ToolCallDelta(index=index, id=call_ids.get(index, ""),
                                        arguments_delta=delta.get("partial_json", ""))

            elif kind == "content_block_stop":
                yield BlockEnd(index=frame.get("index", 0))

            elif kind == "message_delta":
                stop = (frame.get("delta") or {}).get("stop_reason")
                if stop:
                    finish_reason = _STOP_REASONS.get(stop, "stop")
                out_tokens = (frame.get("usage") or {}).get("output_tokens")
                if out_tokens is not None:
                    usage = Usage(usage.input_tokens, out_tokens)

            elif kind == "message_stop":
                break

        # Rule 3: usage BEFORE finish, nothing after finish.
        yield UsageChunk(usage=usage)
        yield Finish(reason=finish_reason)

    # --- non-streaming -----------------------------------------------------------------------

    def generate_once(self, options: GenerateOptions):
        return post_json(
            f"{self.base_url}/v1/messages",
            self._payload(options, stream=False),
            headers=self._headers(),
            timeout=self.timeout,
        )


def apply(ctx, config=None) -> None:
    config = config or {}
    adapter = AnthropicAdapter(
        base_url=config.get("base_url", "http://localhost:11434"),
        api_key=config.get("api_key", "ollama"),
        timeout=config.get("timeout", 120.0),
        retry_policy=config.get("retry_policy"),
    )
    ctx.effect(ctx.llm.register_adapter(config.get("routes", ["anthropic"]), adapter))
