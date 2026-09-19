"""Anthropic Messages wire format, over raw HTTP. Still no SDK.

Works against Ollama's Anthropic-compatible endpoint (`http://localhost:11434`, v0.14+) and,
with a real key, against `https://api.anthropic.com`.

This adapter exists to prove the seam. Mount it instead of `llm_openai` and the same loop, the
same tools and the same policy plugins run over a completely different protocol, with no
change above this file. If that required one conditional anywhere upstream, the abstraction
would be wrong — so the swap test in tests/test_adapters.py asserts it.

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

    The results row is the one to stare at: it is the same rule inverted, and both protocols
    degrade rather than error when you get it wrong.

    Auth differs too — `x-api-key` plus `anthropic-version`, not `Authorization: Bearer`.
"""

from __future__ import annotations

from dshpy.services.llm import Completion, LlmAdapter, Message, ToolCall
from dshpy.transport import post_json

name = "llm-anthropic"
inject = ["llm"]

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 1024  # required by this API, so we must always send something


class AnthropicAdapter(LlmAdapter):
    provider = "anthropic"

    def __init__(self, base_url: str, api_key: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    # --- outbound: neutral -> wire ---------------------------------------------------------

    @staticmethod
    def _encode_tools(tools: list[dict]) -> list[dict]:
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
            for t in tools
        ]

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
                pending_results.append({
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": msg.content,
                })
                continue

            flush_results()

            if msg.role == "assistant":
                blocks: list[dict] = []
                if msg.content:
                    blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,  # a dict here, not a string
                    })
                out.append({"role": "assistant", "content": blocks or ""})
            else:
                out.append({"role": "user", "content": msg.content})

        flush_results()
        return system, out

    # --- inbound: wire -> neutral ----------------------------------------------------------

    @staticmethod
    def _decode(payload: dict) -> Completion:
        text_parts: list[str] = []
        calls: list[ToolCall] = []

        for block in payload.get("content") or []:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                calls.append(ToolCall(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    arguments=block.get("input") or {},  # already parsed
                ))

        stop_reason = payload.get("stop_reason") or "end_turn"
        return Completion(
            text="".join(text_parts),
            tool_calls=calls,
            # Normalize to the neutral vocabulary so the loop sees one spelling.
            finish_reason="tool_calls" if stop_reason == "tool_use" else "stop",
            raw=payload,
        )

    # --- the call --------------------------------------------------------------------------

    def generate(self, messages: list[Message], *, tools: list[dict],
                 model: str, max_tokens: int | None = None) -> Completion:
        system, encoded = self._encode_messages(messages)
        payload: dict = {
            "model": model,
            "messages": encoded,
            "max_tokens": max_tokens or DEFAULT_MAX_TOKENS,  # required, unlike OpenAI's
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = self._encode_tools(tools)

        return self._decode(post_json(
            f"{self.base_url}/v1/messages",
            payload,
            headers={"x-api-key": self.api_key, "anthropic-version": ANTHROPIC_VERSION},
            timeout=self.timeout,
        ))


def apply(ctx, config=None) -> None:
    config = config or {}
    adapter = AnthropicAdapter(
        base_url=config.get("base_url", "http://localhost:11434"),
        api_key=config.get("api_key", "ollama"),
        timeout=config.get("timeout", 120.0),
    )
    ctx.effect(ctx.llm.register_adapter(config.get("routes", ["anthropic"]), adapter))
