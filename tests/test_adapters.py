"""Adapter tests, including the swap test that the whole architecture is staked on.

Both adapters are driven against a fake HTTP transport, so these run with no key, no model and
no network. What they verify is the *translation*: that each adapter speaks its provider's wire
format correctly, and that everything above the adapter cannot tell which one is mounted.
"""

from __future__ import annotations

import json

import pytest

from dshpy.core.context import Runtime
from dshpy.plugins import llm_anthropic, llm_openai, tool_core
from dshpy.services import agent_loop, llm as llm_service, sessions, tools as tools_service
from dshpy.services.llm import Message, ToolCall


class FakeHttp:
    """Scripted stand-in for transport.post_json. Records what was sent."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.sent = []

    def __call__(self, url, payload, **kwargs):
        self.sent.append({"url": url, "payload": payload, **kwargs})
        if not self.responses:
            raise AssertionError(f"unscripted request #{len(self.sent)} to {url}")
        return self.responses.pop(0)


# --- wire payloads both providers would really send ----------------------------------------


def openai_tool_turn(call_id="call_1", name="calculate", args='{"expression": "17 * 23"}'):
    return {"choices": [{
        "finish_reason": "tool_calls",
        "message": {"role": "assistant", "content": None,
                    "tool_calls": [{"id": call_id, "type": "function",
                                    "function": {"name": name, "arguments": args}}]},
    }]}


def openai_text_turn(text="391."):
    return {"choices": [{"finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}]}


def anthropic_tool_turn(call_id="call_1", name="calculate", args=None):
    return {"stop_reason": "tool_use",
            "content": [{"type": "tool_use", "id": call_id, "name": name,
                         "input": args or {"expression": "17 * 23"}}]}


def anthropic_text_turn(text="391."):
    return {"stop_reason": "end_turn", "content": [{"type": "text", "text": text}]}


# --- OpenAI/DeepSeek format ----------------------------------------------------------------


def test_openai_encodes_tools_in_the_function_envelope(monkeypatch):
    http = FakeHttp([openai_text_turn()])
    monkeypatch.setattr(llm_openai, "post_json", http)
    adapter = llm_openai.OpenAIAdapter("http://x/v1", "k")

    adapter.generate([Message("user", "hi")], model="m", tools=[
        {"name": "calculate", "description": "d",
         "parameters": {"type": "object", "properties": {}, "required": []}}])

    tool = http.sent[0]["payload"]["tools"][0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "calculate"
    assert http.sent[0]["url"] == "http://x/v1/chat/completions"


def test_openai_parses_the_json_string_arguments(monkeypatch):
    monkeypatch.setattr(llm_openai, "post_json", FakeHttp([openai_tool_turn()]))
    adapter = llm_openai.OpenAIAdapter("http://x/v1", "k")

    completion = adapter.generate([Message("user", "hi")], model="m", tools=[])
    assert completion.tool_calls[0].arguments == {"expression": "17 * 23"}, \
        "arguments must be normalized from a JSON string to a dict"


@pytest.mark.parametrize("bad", ['{"expression": "17 *', "not json at all", '"a string"'])
def test_openai_survives_malformed_arguments(monkeypatch, bad):
    """The model writes this string token by token; truncation is normal, not exceptional."""
    monkeypatch.setattr(llm_openai, "post_json", FakeHttp([openai_tool_turn(args=bad)]))
    adapter = llm_openai.OpenAIAdapter("http://x/v1", "k")

    completion = adapter.generate([Message("user", "hi")], model="m", tools=[])
    assert completion.tool_calls[0].arguments == {"__malformed__": bad}


def test_openai_accepts_an_already_parsed_object(monkeypatch):
    """Some OpenAI-compatible servers, Ollama included, return an object instead of a string."""
    monkeypatch.setattr(llm_openai, "post_json",
                        FakeHttp([openai_tool_turn(args={"expression": "1+1"})]))
    adapter = llm_openai.OpenAIAdapter("http://x/v1", "k")
    completion = adapter.generate([Message("user", "hi")], model="m", tools=[])
    assert completion.tool_calls[0].arguments == {"expression": "1+1"}


def test_openai_null_content_becomes_empty_string(monkeypatch):
    monkeypatch.setattr(llm_openai, "post_json", FakeHttp([openai_tool_turn()]))
    adapter = llm_openai.OpenAIAdapter("http://x/v1", "k")
    assert adapter.generate([Message("user", "hi")], model="m", tools=[]).text == ""


def test_openai_sends_one_tool_message_per_result(monkeypatch):
    """RULE 1, this protocol's version."""
    http = FakeHttp([openai_text_turn()])
    monkeypatch.setattr(llm_openai, "post_json", http)
    adapter = llm_openai.OpenAIAdapter("http://x/v1", "k")

    adapter.generate([
        Message("user", "go"),
        Message("assistant", "", tool_calls=[ToolCall("a", "t", {}), ToolCall("b", "t", {})]),
        Message("tool", "result-a", tool_call_id="a"),
        Message("tool", "result-b", tool_call_id="b"),
    ], model="m", tools=[])

    sent = http.sent[0]["payload"]["messages"]
    tool_msgs = [m for m in sent if m["role"] == "tool"]
    assert len(tool_msgs) == 2, "OpenAI format needs one message per result"
    assert [m["tool_call_id"] for m in tool_msgs] == ["a", "b"]


def test_openai_reserializes_arguments_to_a_string_on_the_way_out(monkeypatch):
    http = FakeHttp([openai_text_turn()])
    monkeypatch.setattr(llm_openai, "post_json", http)
    adapter = llm_openai.OpenAIAdapter("http://x/v1", "k")

    adapter.generate([Message("assistant", "", tool_calls=[ToolCall("a", "t", {"x": 1})])],
                     model="m", tools=[])

    raw = http.sent[0]["payload"]["messages"][0]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(raw, str) and json.loads(raw) == {"x": 1}


# --- Anthropic format ----------------------------------------------------------------------


def test_anthropic_encodes_tools_with_input_schema(monkeypatch):
    http = FakeHttp([anthropic_text_turn()])
    monkeypatch.setattr(llm_anthropic, "post_json", http)
    adapter = llm_anthropic.AnthropicAdapter("http://x", "k")

    adapter.generate([Message("user", "hi")], model="m", tools=[
        {"name": "calculate", "description": "d", "parameters": {"type": "object"}}])

    tool = http.sent[0]["payload"]["tools"][0]
    assert set(tool) == {"name", "description", "input_schema"}
    assert http.sent[0]["url"] == "http://x/v1/messages"


def test_anthropic_hoists_system_to_a_top_level_parameter(monkeypatch):
    http = FakeHttp([anthropic_text_turn()])
    monkeypatch.setattr(llm_anthropic, "post_json", http)
    adapter = llm_anthropic.AnthropicAdapter("http://x", "k")

    adapter.generate([Message("system", "be brief"), Message("user", "hi")],
                     model="m", tools=[])

    payload = http.sent[0]["payload"]
    assert payload["system"] == "be brief"
    assert [m["role"] for m in payload["messages"]] == ["user"]


def test_anthropic_always_sends_max_tokens(monkeypatch):
    """Required by this API, optional in OpenAI's — omitting it is a 400."""
    http = FakeHttp([anthropic_text_turn()])
    monkeypatch.setattr(llm_anthropic, "post_json", http)
    adapter = llm_anthropic.AnthropicAdapter("http://x", "k")
    adapter.generate([Message("user", "hi")], model="m", tools=[], max_tokens=None)
    assert http.sent[0]["payload"]["max_tokens"] > 0


def test_anthropic_uses_x_api_key_not_bearer(monkeypatch):
    http = FakeHttp([anthropic_text_turn()])
    monkeypatch.setattr(llm_anthropic, "post_json", http)
    llm_anthropic.AnthropicAdapter("http://x", "secret").generate(
        [Message("user", "hi")], model="m", tools=[])
    headers = http.sent[0]["headers"]
    assert headers["x-api-key"] == "secret"
    assert "anthropic-version" in headers


def test_anthropic_packs_all_results_into_one_user_message(monkeypatch):
    """RULE 1 INVERTED — the single most important difference between the two protocols."""
    http = FakeHttp([anthropic_text_turn()])
    monkeypatch.setattr(llm_anthropic, "post_json", http)
    adapter = llm_anthropic.AnthropicAdapter("http://x", "k")

    adapter.generate([
        Message("user", "go"),
        Message("assistant", "", tool_calls=[ToolCall("a", "t", {}), ToolCall("b", "t", {})]),
        Message("tool", "result-a", tool_call_id="a"),
        Message("tool", "result-b", tool_call_id="b"),
    ], model="m", tools=[])

    sent = http.sent[0]["payload"]["messages"]
    result_msgs = [m for m in sent if m["role"] == "user" and isinstance(m["content"], list)]
    assert len(result_msgs) == 1, "Anthropic format needs ONE user message holding all results"
    blocks = result_msgs[0]["content"]
    assert [b["tool_use_id"] for b in blocks] == ["a", "b"]


def test_anthropic_decodes_tool_use_blocks_with_dict_input(monkeypatch):
    monkeypatch.setattr(llm_anthropic, "post_json", FakeHttp([anthropic_tool_turn()]))
    adapter = llm_anthropic.AnthropicAdapter("http://x", "k")

    completion = adapter.generate([Message("user", "hi")], model="m", tools=[])
    assert completion.tool_calls[0].arguments == {"expression": "17 * 23"}
    assert completion.finish_reason == "tool_calls", "stop_reason must normalize to the neutral name"


# --- THE SWAP TEST -------------------------------------------------------------------------


def build(adapter_plugin, adapter_config, responses, monkeypatch):
    """Boot a complete harness with one adapter mounted, against a scripted transport."""
    monkeypatch.setattr(adapter_plugin, "post_json", FakeHttp(responses))
    rt = Runtime()
    rt.mount(llm_service)
    rt.mount(tools_service)
    rt.mount(sessions)
    rt.mount(tool_core)
    rt.mount(adapter_plugin, adapter_config)
    rt.mount(agent_loop, {"model": "m", "provider": adapter_config["routes"][0]})
    return rt


def test_the_same_loop_and_tools_work_over_both_protocols(monkeypatch):
    """If this passes, the seam is real. If it needed a conditional above the adapter, it isn't.

    Same prompt, same tool registry, same loop, same policy surface. Two completely different
    wire formats underneath. The observable behavior must be identical.
    """
    executed = []

    def record(rt):
        rt.events.on("tools/result", lambda e, r: executed.append((e.name, r.content)),
                     owner="test")

    rt_openai = build(llm_openai, {"routes": ["openai"]},
                      [openai_tool_turn(), openai_text_turn("17 * 23 = 391.")], monkeypatch)
    record(rt_openai)
    text_openai = rt_openai.services["agent_loop"].run("what is 17 * 23?")
    openai_calls = list(executed)

    executed.clear()

    rt_anthropic = build(llm_anthropic, {"routes": ["anthropic"]},
                         [anthropic_tool_turn(), anthropic_text_turn("17 * 23 = 391.")],
                         monkeypatch)
    record(rt_anthropic)
    text_anthropic = rt_anthropic.services["agent_loop"].run("what is 17 * 23?")

    assert openai_calls == [("calculate", "391")]
    assert executed == [("calculate", "391")]
    assert text_openai == text_anthropic == "17 * 23 = 391."


def test_a_policy_plugin_works_identically_under_both_adapters(monkeypatch):
    """Policy is written once and is protocol-agnostic — the practical payoff of the seam."""
    from dshpy.services.tools import Deny

    def deny_all(ctx, config=None):
        ctx.on("tools/pre-execute", lambda exec_, next: Deny("nope"))

    denier = type("m", (), {"name": "denier", "inject": ["tools"], "apply": staticmethod(deny_all)})

    for plugin_mod, cfg, responses in [
        (llm_openai, {"routes": ["openai"]}, [openai_tool_turn(), openai_text_turn("blocked")]),
        (llm_anthropic, {"routes": ["anthropic"]},
         [anthropic_tool_turn(), anthropic_text_turn("blocked")]),
    ]:
        rt = build(plugin_mod, cfg, responses, monkeypatch)
        rt.mount(denier)
        seen = []
        rt.events.on("tools/result", lambda e, r: seen.append(r.content), owner="test")
        rt.services["agent_loop"].run("go")
        assert seen == ["Error: nope"], f"policy behaved differently under {plugin_mod.name}"


def test_swapping_the_adapter_is_an_unmount_and_a_mount(monkeypatch):
    """No restart, no edit above the seam — the claim 'replaceable from configuration'."""
    monkeypatch.setattr(llm_openai, "post_json", FakeHttp([]))
    monkeypatch.setattr(llm_anthropic, "post_json", FakeHttp([]))

    rt = Runtime()
    rt.mount(llm_service)
    entry = rt.mount(llm_openai, {"routes": ["openai"]})
    assert rt.services["llm"].providers() == ["openai"]

    rt.unmount(entry)
    assert rt.services["llm"].providers() == []

    rt.mount(llm_anthropic, {"routes": ["anthropic"]})
    assert rt.services["llm"].providers() == ["anthropic"]


def test_two_adapters_can_be_mounted_at_once(monkeypatch):
    monkeypatch.setattr(llm_openai, "post_json", FakeHttp([]))
    monkeypatch.setattr(llm_anthropic, "post_json", FakeHttp([]))
    rt = Runtime()
    rt.mount(llm_service)
    rt.mount(llm_openai, {"routes": ["openai"]})
    rt.mount(llm_anthropic, {"routes": ["anthropic"]})
    assert sorted(rt.services["llm"].providers()) == ["anthropic", "openai"]


def test_duplicate_provider_routes_are_rejected(monkeypatch):
    monkeypatch.setattr(llm_openai, "post_json", FakeHttp([]))
    rt = Runtime()
    rt.mount(llm_service)
    rt.mount(llm_openai, {"routes": ["openai"]})
    with pytest.raises(RuntimeError, match="already registered"):
        rt.mount(llm_openai, {"routes": ["openai"]})
