"""Adapter tests: the two SSE dialects, and the swap test the architecture is staked on.

Both adapters are driven from canned SSE byte streams — no key, no model, no network. What
these verify is the *translation*: that each adapter speaks its provider's wire format
correctly, and that everything above the adapter cannot tell which one is mounted.
"""

from __future__ import annotations

import json

import pytest

from dshpy.core.context import Runtime
from dshpy.plugins import llm_anthropic, llm_openai, tool_core
from dshpy.services import agent_loop, llm as llm_service, sessions, tools as tools_service
from dshpy.services.llm import (
    BlockAssembler,
    GenerateOptions,
    Message,
    ToolCall,
    ToolCallDelta,
    parse_arguments,
)


class FakeStream:
    """Scripted stand-in for transport.post_stream. Records the payloads it was sent."""

    def __init__(self, *bodies: bytes, chunk_size: int | None = None):
        self.bodies = list(bodies)
        self.sent: list[dict] = []
        self.chunk_size = chunk_size

    def __call__(self, url, payload, **kwargs):
        self.sent.append({"url": url, "payload": payload, **kwargs})
        if not self.bodies:
            raise AssertionError(f"unscripted request #{len(self.sent)} to {url}")
        body = self.bodies.pop(0)
        if self.chunk_size:
            return iter([body[i:i + self.chunk_size]
                         for i in range(0, len(body), self.chunk_size)])
        return iter([body])


# --- canned wire bodies ---------------------------------------------------------------------


def openai_sse(*frames: dict, done: bool = True) -> bytes:
    out = b"".join(f"data: {json.dumps(f)}\n\n".encode() for f in frames)
    return out + (b"data: [DONE]\n\n" if done else b"")


def openai_text(text="391.", usage=(10, 4)) -> bytes:
    return openai_sse(
        {"choices": [{"delta": {"content": text}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]}},
    )


def openai_tool(call_id="call_1", name="calculate", args='{"expression": "17 * 23"}') -> bytes:
    """Arguments split across two frames, as a real server sends them."""
    half = len(args) // 2
    return openai_sse(
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": call_id, "function": {"name": name, "arguments": args[:half]}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": args[half:]}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    )


def anthropic_sse(*events: tuple[str, dict]) -> bytes:
    return b"".join(f"event: {n}\ndata: {json.dumps(d)}\n\n".encode() for n, d in events)


def anthropic_text(text="391.", usage=(10, 4)) -> bytes:
    return anthropic_sse(
        ("message_start", {"message": {"usage": {"input_tokens": usage[0]}}}),
        ("content_block_start", {"index": 0, "content_block": {"type": "text"}}),
        ("content_block_delta", {"index": 0,
                                 "delta": {"type": "text_delta", "text": text}}),
        ("content_block_stop", {"index": 0}),
        ("message_delta", {"delta": {"stop_reason": "end_turn"},
                           "usage": {"output_tokens": usage[1]}}),
        ("message_stop", {}),
    )


def anthropic_tool(call_id="call_1", name="calculate", args='{"expression": "17 * 23"}') -> bytes:
    half = len(args) // 2
    return anthropic_sse(
        ("message_start", {"message": {"usage": {"input_tokens": 10}}}),
        ("content_block_start", {"index": 0, "content_block": {
            "type": "tool_use", "id": call_id, "name": name}}),
        ("content_block_delta", {"index": 0, "delta": {
            "type": "input_json_delta", "partial_json": args[:half]}}),
        ("content_block_delta", {"index": 0, "delta": {
            "type": "input_json_delta", "partial_json": args[half:]}}),
        ("content_block_stop", {"index": 0}),
        ("message_delta", {"delta": {"stop_reason": "tool_use"},
                           "usage": {"output_tokens": 7}}),
        ("message_stop", {}),
    )


def opts(messages=None, tools=None, **kw):
    return GenerateOptions(messages=messages or [Message("user", "hi")],
                           tools=tools or [], model="m", **kw)


# --- OpenAI / DeepSeek dialect ----------------------------------------------------------------


def test_openai_streams_text_and_reports_usage(monkeypatch):
    monkeypatch.setattr(llm_openai, "post_stream", FakeStream(openai_text("hello")))
    adapter = llm_openai.OpenAIAdapter("http://x/v1", "k")

    completion = adapter.generate(opts())
    assert completion.text == "hello"
    assert completion.finish_reason == "stop"
    assert (completion.usage.input_tokens, completion.usage.output_tokens) == (10, 4)


def test_openai_concatenates_argument_fragments_then_parses_once(monkeypatch):
    """Rule 2: arguments stay raw strings until complete. Parsing a fragment is the bug."""
    monkeypatch.setattr(llm_openai, "post_stream", FakeStream(openai_tool()))
    adapter = llm_openai.OpenAIAdapter("http://x/v1", "k")

    completion = adapter.generate(opts())
    assert completion.finish_reason == "tool_calls"
    assert completion.tool_calls[0].arguments == {"expression": "17 * 23"}
    assert completion.tool_calls[0].name == "calculate"


def test_openai_allocates_one_block_index_per_tool_call(monkeypatch):
    """Two interleaved calls must not have their fragments merged."""
    body = openai_sse(
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "a", "function": {"name": "t", "arguments": '{"x":'}},
            {"index": 1, "id": "b", "function": {"name": "t", "arguments": '{"y":'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 1, "function": {"arguments": "2}"}},
            {"index": 0, "function": {"arguments": "1}"}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    )
    monkeypatch.setattr(llm_openai, "post_stream", FakeStream(body))

    completion = llm_openai.OpenAIAdapter("http://x/v1", "k").generate(opts())
    by_id = {c.id: c.arguments for c in completion.tool_calls}
    assert by_id == {"a": {"x": 1}, "b": {"y": 2}}


def test_openai_survives_a_stream_split_at_every_byte(monkeypatch):
    monkeypatch.setattr(llm_openai, "post_stream", FakeStream(openai_tool(), chunk_size=1))
    completion = llm_openai.OpenAIAdapter("http://x/v1", "k").generate(opts())
    assert completion.tool_calls[0].arguments == {"expression": "17 * 23"}


def test_openai_requests_usage_when_streaming(monkeypatch):
    """Most OpenAI-compatible servers send no usage at all without this opt-in."""
    fake = FakeStream(openai_text())
    monkeypatch.setattr(llm_openai, "post_stream", fake)
    llm_openai.OpenAIAdapter("http://x/v1", "k").generate(opts())
    assert fake.sent[0]["payload"]["stream_options"] == {"include_usage": True}
    assert fake.sent[0]["url"] == "http://x/v1/chat/completions"


def test_openai_encodes_tools_in_the_function_envelope(monkeypatch):
    fake = FakeStream(openai_text())
    monkeypatch.setattr(llm_openai, "post_stream", fake)
    llm_openai.OpenAIAdapter("http://x/v1", "k").generate(opts(tools=[
        {"name": "calculate", "description": "d", "parameters": {"type": "object"}}]))

    tool = fake.sent[0]["payload"]["tools"][0]
    assert tool["type"] == "function" and tool["function"]["name"] == "calculate"


def test_openai_sends_one_tool_message_per_result(monkeypatch):
    """RULE 1, this protocol's version."""
    fake = FakeStream(openai_text())
    monkeypatch.setattr(llm_openai, "post_stream", fake)
    llm_openai.OpenAIAdapter("http://x/v1", "k").generate(opts(messages=[
        Message("user", "go"),
        Message("assistant", "", tool_calls=[ToolCall("a", "t", {}), ToolCall("b", "t", {})]),
        Message("tool", "result-a", tool_call_id="a"),
        Message("tool", "result-b", tool_call_id="b"),
    ]))

    tool_msgs = [m for m in fake.sent[0]["payload"]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["a", "b"]


def test_openai_reserializes_arguments_to_a_string_on_the_way_out(monkeypatch):
    fake = FakeStream(openai_text())
    monkeypatch.setattr(llm_openai, "post_stream", fake)
    llm_openai.OpenAIAdapter("http://x/v1", "k").generate(opts(messages=[
        Message("assistant", "", tool_calls=[ToolCall("a", "t", {"x": 1})])]))

    raw = fake.sent[0]["payload"]["messages"][0]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(raw, str) and json.loads(raw) == {"x": 1}


# --- Anthropic dialect --------------------------------------------------------------------


def test_anthropic_streams_text_and_reports_usage(monkeypatch):
    monkeypatch.setattr(llm_anthropic, "post_stream", FakeStream(anthropic_text("hello")))
    completion = llm_anthropic.AnthropicAdapter("http://x", "k").generate(opts())
    assert completion.text == "hello"
    assert (completion.usage.input_tokens, completion.usage.output_tokens) == (10, 4)


def test_anthropic_takes_the_tool_name_from_content_block_start(monkeypatch):
    """This dialect sends the name once, on block start -- never on the deltas."""
    monkeypatch.setattr(llm_anthropic, "post_stream", FakeStream(anthropic_tool()))
    completion = llm_anthropic.AnthropicAdapter("http://x", "k").generate(opts())
    assert completion.tool_calls[0].name == "calculate"
    assert completion.tool_calls[0].arguments == {"expression": "17 * 23"}
    assert completion.finish_reason == "tool_calls"


def test_anthropic_survives_a_stream_split_at_every_byte(monkeypatch):
    monkeypatch.setattr(llm_anthropic, "post_stream", FakeStream(anthropic_tool(), chunk_size=1))
    completion = llm_anthropic.AnthropicAdapter("http://x", "k").generate(opts())
    assert completion.tool_calls[0].arguments == {"expression": "17 * 23"}


def test_anthropic_hoists_system_to_a_top_level_parameter(monkeypatch):
    fake = FakeStream(anthropic_text())
    monkeypatch.setattr(llm_anthropic, "post_stream", fake)
    llm_anthropic.AnthropicAdapter("http://x", "k").generate(opts(
        messages=[Message("system", "be brief"), Message("user", "hi")]))

    payload = fake.sent[0]["payload"]
    assert payload["system"] == "be brief"
    assert [m["role"] for m in payload["messages"]] == ["user"]


def test_anthropic_always_sends_max_tokens(monkeypatch):
    """Required by this API, optional in OpenAI's -- omitting it is a 400."""
    fake = FakeStream(anthropic_text())
    monkeypatch.setattr(llm_anthropic, "post_stream", fake)
    llm_anthropic.AnthropicAdapter("http://x", "k").generate(opts(max_tokens=None))
    assert fake.sent[0]["payload"]["max_tokens"] > 0


def test_anthropic_uses_x_api_key_not_bearer(monkeypatch):
    fake = FakeStream(anthropic_text())
    monkeypatch.setattr(llm_anthropic, "post_stream", fake)
    llm_anthropic.AnthropicAdapter("http://x", "secret").generate(opts())
    assert fake.sent[0]["headers"]["x-api-key"] == "secret"
    assert "anthropic-version" in fake.sent[0]["headers"]


def test_anthropic_packs_all_results_into_one_user_message(monkeypatch):
    """RULE 1 INVERTED -- the single most important difference between the two protocols."""
    fake = FakeStream(anthropic_text())
    monkeypatch.setattr(llm_anthropic, "post_stream", fake)
    llm_anthropic.AnthropicAdapter("http://x", "k").generate(opts(messages=[
        Message("user", "go"),
        Message("assistant", "", tool_calls=[ToolCall("a", "t", {}), ToolCall("b", "t", {})]),
        Message("tool", "result-a", tool_call_id="a"),
        Message("tool", "result-b", tool_call_id="b"),
    ]))

    sent = fake.sent[0]["payload"]["messages"]
    result_msgs = [m for m in sent if m["role"] == "user" and isinstance(m["content"], list)]
    assert len(result_msgs) == 1, "Anthropic format needs ONE user message holding all results"
    assert [b["tool_use_id"] for b in result_msgs[0]["content"]] == ["a", "b"]


# --- the assembler ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ['{"expression": "17 *', "not json at all", '"a string"'])
def test_malformed_arguments_become_a_marker_not_an_exception(bad):
    """The model writes this token by token; truncation is normal, not exceptional."""
    assert parse_arguments(bad) == {"__malformed__": bad}


def test_an_already_parsed_object_is_accepted():
    """Ollama returns one for some models, despite the spec saying string."""
    assert parse_arguments({"expression": "1+1"}) == {"expression": "1+1"}


def test_assembler_ignores_chunks_after_finish():
    """Rule 3: finish is terminal. A late chunk must not corrupt the result."""
    from dshpy.services.llm import Finish, TextDelta

    assembler = BlockAssembler()
    assembler.push(TextDelta(index=0, text="good"))
    assembler.push(Finish(reason="stop"))
    assembler.push(TextDelta(index=0, text="LATE"))
    assert assembler.result().text == "good"


def test_assembler_normalizes_stop_with_tool_calls_to_tool_calls():
    """Some providers report 'stop' while still emitting calls; the loop keys off this."""
    from dshpy.services.llm import Finish

    assembler = BlockAssembler()
    assembler.push(ToolCallDelta(index=0, id="a", name="t", arguments_delta="{}"))
    assembler.push(Finish(reason="stop"))
    assert assembler.result().finish_reason == "tool_calls"


# --- THE SWAP TEST ---------------------------------------------------------------------------


def build(adapter_plugin, adapter_config, bodies, monkeypatch, extra=()):
    monkeypatch.setattr(adapter_plugin, "post_stream", FakeStream(*bodies))
    rt = Runtime()
    rt.mount(llm_service)
    rt.mount(tools_service)
    rt.mount(sessions)
    rt.mount(tool_core)
    rt.mount(adapter_plugin, adapter_config)
    for plugin_mod, cfg in extra:
        rt.mount(plugin_mod, cfg)
    rt.mount(agent_loop, {"model": "m", "provider": adapter_config["routes"][0]})
    return rt


def test_the_same_loop_and_tools_work_over_both_protocols(monkeypatch):
    """If this passes, the seam is real. If it needed a conditional above the adapter, it isn't.

    Same prompt, same tool registry, same loop. Two completely different wire formats and two
    completely different SSE framings underneath. Observable behavior must be identical.
    """
    executed = []

    def record(rt):
        rt.events.on("tools/result", lambda e, r: executed.append((e.name, r.content)),
                     owner="test")

    rt_openai = build(llm_openai, {"routes": ["openai"]},
                      [openai_tool(), openai_text("17 * 23 = 391.")], monkeypatch)
    record(rt_openai)
    text_openai = rt_openai.services["agent_loop"].run("what is 17 * 23?")
    openai_calls = list(executed)

    executed.clear()

    rt_anthropic = build(llm_anthropic, {"routes": ["anthropic"]},
                         [anthropic_tool(), anthropic_text("17 * 23 = 391.")], monkeypatch)
    record(rt_anthropic)
    text_anthropic = rt_anthropic.services["agent_loop"].run("what is 17 * 23?")

    assert openai_calls == [("calculate", "391")]
    assert executed == [("calculate", "391")]
    assert text_openai == text_anthropic == "17 * 23 = 391."


def test_a_policy_plugin_works_identically_under_both_adapters(monkeypatch):
    """Policy is written once and is protocol-agnostic -- the practical payoff of the seam."""
    from dshpy.services.tools import Deny

    denier = type("m", (), {
        "name": "denier", "inject": ["tools"],
        "apply": staticmethod(lambda ctx, config=None: ctx.on(
            "tools/pre-execute", lambda exec_, next: Deny("nope"))),
    })

    for plugin_mod, cfg, bodies in [
        (llm_openai, {"routes": ["openai"]}, [openai_tool(), openai_text("blocked")]),
        (llm_anthropic, {"routes": ["anthropic"]}, [anthropic_tool(), anthropic_text("blocked")]),
    ]:
        rt = build(plugin_mod, cfg, bodies, monkeypatch)
        rt.mount(denier)
        seen = []
        rt.events.on("tools/result", lambda e, r: seen.append(r.content), owner="test")
        rt.services["agent_loop"].run("go")
        assert seen == ["Error: nope"], f"policy differed under {plugin_mod.name}"


def test_swapping_the_adapter_is_an_unmount_and_a_mount(monkeypatch):
    monkeypatch.setattr(llm_openai, "post_stream", FakeStream())
    monkeypatch.setattr(llm_anthropic, "post_stream", FakeStream())

    rt = Runtime()
    rt.mount(llm_service)
    entry = rt.mount(llm_openai, {"routes": ["openai"]})
    assert rt.services["llm"].providers() == ["openai"]

    rt.unmount(entry)
    assert rt.services["llm"].providers() == []

    rt.mount(llm_anthropic, {"routes": ["anthropic"]})
    assert rt.services["llm"].providers() == ["anthropic"]


def test_two_adapters_can_be_mounted_at_once(monkeypatch):
    monkeypatch.setattr(llm_openai, "post_stream", FakeStream())
    monkeypatch.setattr(llm_anthropic, "post_stream", FakeStream())
    rt = Runtime()
    rt.mount(llm_service)
    rt.mount(llm_openai, {"routes": ["openai"]})
    rt.mount(llm_anthropic, {"routes": ["anthropic"]})
    assert sorted(rt.services["llm"].providers()) == ["anthropic", "openai"]


def test_duplicate_provider_routes_are_rejected(monkeypatch):
    monkeypatch.setattr(llm_openai, "post_stream", FakeStream())
    rt = Runtime()
    rt.mount(llm_service)
    rt.mount(llm_openai, {"routes": ["openai"]})
    with pytest.raises(RuntimeError, match="already registered"):
        rt.mount(llm_openai, {"routes": ["openai"]})
