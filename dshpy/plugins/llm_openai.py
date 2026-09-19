"""OpenAI / DeepSeek wire format, over raw HTTP.

Works against DeepSeek's API (`https://api.deepseek.com/v1`) and any OpenAI-compatible server,
including Ollama at `http://localhost:11434/v1`.

THE WIRE FORMAT, AND WHERE IT BITES

    Request:   {"model", "messages", "tools": [{"type":"function","function":{...}}]}
    Response:  choices[0].message.tool_calls = [
                   {"id", "type":"function", "function":{"name","arguments"}}
               ]
               choices[0].finish_reason == "tool_calls"
    Results:   ONE {"role":"tool","tool_call_id","content"} message PER CALL.

    Three things about this protocol that the neutral vocabulary in services/llm.py exists to
    hide from everything above:

    1. `function.arguments` is a **JSON string**, not an object. The model generates it token by
       token, so it can be truncated or malformed — a normal occurrence, not an exception. The
       Anthropic format hands you a parsed dict and never exposes you to this.

    2. Tool results are **one message each**, in order. The Anthropic format wants the exact
       opposite: one user message carrying every result as a block. Getting this backwards
       doesn't always error — it can just quietly degrade parallel tool calling.

    3. `content` is `null` when there are tool calls, so a naive `message["content"]` gives you
       `None` where you expected a string.

    All three are handled here and nowhere else. That containment is the point of the seam.
"""

from __future__ import annotations

import json

from dshpy.services.llm import Completion, LlmAdapter, Message, ToolCall
from dshpy.transport import post_json

name = "llm-openai"
inject = ["llm"]


class OpenAIAdapter(LlmAdapter):
    provider = "openai"

    def __init__(self, base_url: str, api_key: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    # --- outbound: neutral -> wire ---------------------------------------------------------

    @staticmethod
    def _encode_tools(tools: list[dict]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in tools
        ]

    @staticmethod
    def _encode_messages(messages: list[Message]) -> list[dict]:
        out: list[dict] = []
        for msg in messages:
            if msg.role == "tool":
                # Rule 1, this protocol's version: one message per result.
                out.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                })
            elif msg.role == "assistant" and msg.tool_calls:
                out.append({
                    "role": "assistant",
                    "content": msg.content or None,  # null, not "", when there are calls
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                # Back to a STRING on the way out, since that is what we parsed.
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                })
            else:
                out.append({"role": msg.role, "content": msg.content})
        return out

    # --- inbound: wire -> neutral ----------------------------------------------------------

    @staticmethod
    def _decode(payload: dict) -> Completion:
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}

        calls = []
        for raw in message.get("tool_calls") or []:
            fn = raw.get("function") or {}
            calls.append(ToolCall(
                id=raw.get("id", ""),
                name=fn.get("name", ""),
                arguments=_parse_arguments(fn.get("arguments")),
            ))

        return Completion(
            text=message.get("content") or "",  # `or ""` handles the null-content case
            tool_calls=calls,
            finish_reason=choice.get("finish_reason") or "stop",
            raw=payload,
        )

    # --- the call --------------------------------------------------------------------------

    def generate(self, messages: list[Message], *, tools: list[dict],
                 model: str, max_tokens: int | None = None) -> Completion:
        payload: dict = {
            "model": model,
            "messages": self._encode_messages(messages),
        }
        if tools:
            payload["tools"] = self._encode_tools(tools)
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens  # optional here; required in the Anthropic API

        return self._decode(post_json(
            f"{self.base_url}/chat/completions",
            payload,
            api_key=self.api_key,
            timeout=self.timeout,
        ))


def _parse_arguments(raw: str | dict | None) -> dict:
    """Turn `function.arguments` into a dict, defensively.

    The model writes this string token by token, so truncation and malformed JSON are ordinary
    outcomes. Raising here would kill the loop; instead we return a marker the tool will reject,
    which becomes an error result the model can read and retry from. Same reasoning as rule 3.

    Some OpenAI-compatible servers (Ollama among them, for some models) helpfully hand back an
    already-parsed object, so accept that too rather than assuming the spec is followed.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"__malformed__": raw}
    return parsed if isinstance(parsed, dict) else {"__malformed__": raw}


def apply(ctx, config=None) -> None:
    config = config or {}
    adapter = OpenAIAdapter(
        base_url=config.get("base_url", "http://localhost:11434/v1"),
        api_key=config.get("api_key", "ollama"),
        timeout=config.get("timeout", 120.0),
    )
    ctx.effect(ctx.llm.register_adapter(config.get("routes", ["openai"]), adapter))
