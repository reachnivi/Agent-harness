"""OpenAI / DeepSeek wire format, over raw HTTP. Streaming and non-streaming.

Works against DeepSeek (`https://api.deepseek.com/v1`) and any OpenAI-compatible server,
including Ollama at `http://localhost:11434/v1`.

THE WIRE FORMAT, AND WHERE IT BITES

    Request:   {"model", "messages", "tools":[{"type":"function","function":{...}}], "stream"}
    Response:  choices[0].message.tool_calls = [{"id","type","function":{"name","arguments"}}]
               choices[0].finish_reason == "tool_calls"
    Results:   ONE {"role":"tool","tool_call_id","content"} message PER CALL.

    Three things the neutral vocabulary exists to hide from everything above:

    1. `function.arguments` is a **JSON string**, not an object — generated token by token, so
       it can be truncated or malformed. The Anthropic format hands you a parsed dict and never
       exposes you to this.
    2. Tool results are **one message each**. The Anthropic format wants the exact opposite:
       one user message carrying every result as a block. Getting this backwards doesn't always
       error — it can just quietly degrade parallel tool calling.
    3. `content` is `null` when there are tool calls, so a naive read gives `None`.

THE SSE DIALECT

    Anonymous frames — no `event:` line, just `data:` — terminated by a literal `data: [DONE]`.
    Each frame holds `choices[0].delta`, a *partial* message:

        {"choices":[{"delta":{"content":"He"}}]}
        {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",
                              "function":{"name":"calculate","arguments":"{\\"exp"}}]}}]}
        {"choices":[{"delta":{"tool_calls":[{"index":0,
                              "function":{"arguments":"ression\\":\\"2+2\\"}"}}]}}]}

    Note `tool_calls[].index`: that is the provider telling you which call a fragment belongs
    to, and it is why the neutral protocol carries an index too. `id` and `name` usually arrive
    only on the first fragment of a call; `arguments` arrives in pieces that must be
    concatenated in order and parsed only once complete.
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

name = "llm-openai"
inject = ["llm"]

_FINISH_REASONS = {"stop": "stop", "tool_calls": "tool_calls", "length": "length"}


class OpenAIAdapter(LlmAdapter):
    provider = "openai"

    def __init__(self, base_url: str, api_key: str, timeout: float = 120.0,
                 retry_policy: dict | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.retry_policy = retry_policy

    # --- outbound: neutral -> wire ---------------------------------------------------------

    @staticmethod
    def _encode_tools(tools: list[dict]) -> list[dict]:
        return [
            {"type": "function",
             "function": {"name": t["name"], "description": t["description"],
                          "parameters": t["parameters"]}}
            for t in tools
        ]

    @staticmethod
    def _encode_messages(messages: list[Message]) -> list[dict]:
        out: list[dict] = []
        for msg in messages:
            if msg.role == "tool":
                # Rule 1, this protocol's version: one message per result.
                out.append({"role": "tool", "tool_call_id": msg.tool_call_id,
                            "content": msg.content})
            elif msg.role == "assistant" and msg.tool_calls:
                out.append({
                    "role": "assistant",
                    "content": msg.content or None,  # null, not "", when there are calls
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.name,
                                      # back to a STRING on the way out
                                      "arguments": json.dumps(tc.arguments)}}
                        for tc in msg.tool_calls
                    ],
                })
            else:
                out.append({"role": msg.role, "content": msg.content})
        return out

    def _payload(self, options: GenerateOptions, stream: bool) -> dict:
        payload: dict = {
            "model": options.model,
            "messages": self._encode_messages(options.messages),
        }
        if options.tools:
            payload["tools"] = self._encode_tools(options.tools)
        if options.max_tokens is not None:
            payload["max_tokens"] = options.max_tokens  # optional here; required by Anthropic
        if stream:
            payload["stream"] = True
            # Without this, most OpenAI-compatible servers send no usage at all when streaming.
            payload["stream_options"] = {"include_usage": True}
        return payload

    # --- streaming ---------------------------------------------------------------------------

    def stream(self, options: GenerateOptions) -> Iterator[StreamChunk]:
        byte_chunks = post_stream(
            f"{self.base_url}/chat/completions",
            self._payload(options, stream=True),
            api_key=self.api_key,
            timeout=self.timeout,
            token=options.token,
        )

        text_index: int | None = None
        open_calls: dict[int, bool] = {}   # provider index -> has block-start been emitted
        next_index = 0                      # OUR index, allocated in first-seen order
        call_index: dict[int, int] = {}     # provider tool_call index -> our block index
        usage = Usage()
        finish_reason = "stop"

        for event in parse_sse(byte_chunks):
            try:
                frame = json.loads(event.data)
            except json.JSONDecodeError:
                continue  # a keep-alive or a malformed frame: skip rather than abort the turn

            # A usage-only frame has an empty choices list, hence the `or [{}]`.
            for choice in frame.get("choices") or []:
                delta = choice.get("delta") or {}

                if delta.get("content"):
                    if text_index is None:
                        text_index = next_index
                        next_index += 1
                        yield BlockStart(index=text_index, block_type="text")
                    yield TextDelta(index=text_index, text=delta["content"])

                for raw_call in delta.get("tool_calls") or []:
                    provider_index = raw_call.get("index", 0)
                    if provider_index not in call_index:
                        call_index[provider_index] = next_index
                        next_index += 1
                    index = call_index[provider_index]

                    fn = raw_call.get("function") or {}
                    if not open_calls.get(provider_index):
                        open_calls[provider_index] = True
                        yield BlockStart(index=index, block_type="tool-call")

                    yield ToolCallDelta(
                        index=index,
                        id=raw_call.get("id") or "",
                        name=fn.get("name"),
                        # Rule 2: a raw fragment. Never parsed here.
                        arguments_delta=fn.get("arguments") or "",
                    )

                if choice.get("finish_reason"):
                    finish_reason = _FINISH_REASONS.get(choice["finish_reason"], "stop")

            if frame.get("usage"):
                usage = Usage(
                    input_tokens=frame["usage"].get("prompt_tokens", 0),
                    output_tokens=frame["usage"].get("completion_tokens", 0),
                )

        for index in sorted({*([text_index] if text_index is not None else []),
                             *call_index.values()}):
            yield BlockEnd(index=index)

        # Rule 3: usage BEFORE finish, nothing after finish.
        yield UsageChunk(usage=usage)
        yield Finish(reason=finish_reason)

    # --- non-streaming -----------------------------------------------------------------------
    #
    # Kept rather than inherited from LlmAdapter.generate() because some deployments disable
    # streaming, and because the swap test is clearer when both paths exist and agree.

    def generate_once(self, options: GenerateOptions):
        return post_json(
            f"{self.base_url}/chat/completions",
            self._payload(options, stream=False),
            api_key=self.api_key,
            timeout=self.timeout,
        )


def apply(ctx, config=None) -> None:
    config = config or {}
    adapter = OpenAIAdapter(
        base_url=config.get("base_url", "http://localhost:11434/v1"),
        api_key=config.get("api_key", "ollama"),
        timeout=config.get("timeout", 120.0),
        retry_policy=config.get("retry_policy"),
    )
    ctx.effect(ctx.llm.register_adapter(config.get("routes", ["openai"]), adapter))
