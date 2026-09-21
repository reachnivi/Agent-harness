"""Tests for scoped registration and subagents.

The scope tests come first because they are the kernel change everything else rests on. Each
asserts a visibility property that, if it quietly stopped holding, would leak a parent's tools
into a child with nothing failing.
"""

from __future__ import annotations

import json

import pytest

from dshpy.core.context import Runtime
from dshpy.core.scope import ScopedRegistry
from dshpy.plugins import subagent_in_process, tool_core, tool_subagent
from dshpy.services import (
    agent_loop as agent_loop_service,
    llm as llm_service,
    sessions as sessions_service,
    subagents as subagents_service,
    tools as tools_service,
)
from dshpy.services.llm import Finish, LlmAdapter, TextDelta, ToolCallDelta
from dshpy.services.subagents import (
    Capabilities,
    SubagentError,
    SubagentProvider,
    SubagentRequest,
    SubagentResult,
    SubagentsService,
    UnsupportedCapability,
)
from dshpy.services.tools import define_tool


# --- the scope primitive ---------------------------------------------------------------------


def test_a_global_entry_is_visible_everywhere():
    reg: ScopedRegistry[str] = ScopedRegistry()
    reg.add("global")
    assert reg.visible() == ["global"]
    assert reg.visible(object()) == ["global"]


def test_a_scoped_entry_is_visible_only_to_its_scope():
    reg: ScopedRegistry[str] = ScopedRegistry()
    key, other = object(), object()
    reg.add("mine", scope=key)

    assert reg.visible(key) == ["mine"]
    assert reg.visible(other) == []
    assert reg.visible() == [], "a scoped entry leaked into the global view"


def test_scope_matching_is_by_identity_not_equality():
    """Two agents that compare equal must still not see each other's registrations."""
    class AlwaysEqual:
        def __eq__(self, other): return True
        def __hash__(self): return 1

    reg: ScopedRegistry[str] = ScopedRegistry()
    a, b = AlwaysEqual(), AlwaysEqual()
    reg.add("a-only", scope=a)
    assert reg.visible(a) == ["a-only"]
    assert reg.visible(b) == [], "equality matching leaked one scope into another"


def test_disposing_a_scoped_entry_removes_it():
    reg: ScopedRegistry[str] = ScopedRegistry()
    key = object()
    dispose = reg.add("temp", scope=key)
    dispose()
    assert reg.visible(key) == []


def test_drop_scope_removes_everything_bound_to_it():
    reg: ScopedRegistry[str] = ScopedRegistry()
    key = object()
    reg.add("global")
    reg.add("a", scope=key)
    reg.add("b", scope=key)
    assert reg.drop_scope(key) == 2
    assert reg.visible(key) == ["global"]


# --- scoping in the tool registry --------------------------------------------------------------


@pytest.fixture
def rt():
    runtime = Runtime()
    runtime.mount(tools_service)
    runtime.mount(tool_core)
    return runtime


def test_global_tools_are_visible_inside_a_scope(rt):
    """Scoping must be ADDITIVE -- a subagent must not lose the tools everyone already had."""
    child = object()
    assert set(rt.services["tools"].names(child)) >= {"get_time", "calculate"}


def test_a_scoped_tool_is_invisible_to_the_parent(rt):
    child = object()
    rt.services["tools"].register(define_tool(
        name="child_only", description="d", parameters={},
        execute=lambda a, e: "x"), scope=child)

    assert "child_only" in rt.services["tools"].names(child)
    assert "child_only" not in rt.services["tools"].names()


def test_the_same_tool_name_can_exist_in_two_scopes(rt):
    """Impossible with a flat registry, and the reason scoping beats unmount-and-remount."""
    a, b = object(), object()
    rt.services["tools"].register(define_tool(
        name="search", description="a", parameters={}, execute=lambda ar, e: "A"), scope=a)
    rt.services["tools"].register(define_tool(
        name="search", description="b", parameters={}, execute=lambda ar, e: "B"), scope=b)

    assert rt.services["tools"].execute("c", "search", {}, scope=a).content == "A"
    assert rt.services["tools"].execute("c", "search", {}, scope=b).content == "B"


def test_calling_a_tool_outside_your_scope_reports_it_as_unknown(rt):
    """Not 'forbidden': the caller was never shown it, so 'unknown' is true from its position."""
    child = object()
    rt.services["tools"].register(define_tool(
        name="secret", description="d", parameters={},
        execute=lambda a, e: "x"), scope=child)

    result = rt.services["tools"].execute("c", "secret", {})
    assert result.is_error and "unknown tool" in result.content
    assert "secret" not in result.content.split("Available:")[-1]


def test_ctx_fork_carries_the_scope_key_for_explicit_use():
    """fork() carries the key; it does NOT silently rescope every registration.

    Inferring the scope from the calling context was the first design and cannot work: one
    ToolsService serves every context, so it holds the ctx it was *created* with and would have
    scoped everything to that one regardless of the caller. An explicit argument beats magic
    that silently reads the wrong value.
    """
    runtime = Runtime()
    runtime.mount(tools_service)
    captured = {}
    runtime.mount(type("m", (), {
        "name": "p", "inject": ["tools"],
        "apply": staticmethod(lambda ctx, config=None: captured.update(ctx=ctx))}))

    child = object()
    child_ctx = captured["ctx"].fork(child)
    assert child_ctx.scope_key is child

    child_ctx.tools.register(
        define_tool(name="t", description="d", parameters={}, execute=lambda a, e: "x"),
        scope=child_ctx.scope_key,
    )
    assert "t" in runtime.services["tools"].names(child)
    assert "t" not in runtime.services["tools"].names()


def test_a_scoped_tool_shadows_a_global_one_of_the_same_name():
    """So a child can REPLACE a tool, not only add or remove one."""
    runtime = Runtime()
    runtime.mount(tools_service)
    tools = runtime.services["tools"]
    child = object()

    tools.register(define_tool(name="search", description="global", parameters={},
                               execute=lambda a, e: "GLOBAL"))
    tools.register(define_tool(name="search", description="child", parameters={},
                               execute=lambda a, e: "CHILD"), scope=child)

    assert tools.execute("c", "search", {}).content == "GLOBAL"
    assert tools.execute("c", "search", {}, scope=child).content == "CHILD"
    assert tools.names(child).count("search") == 1, "the child sees it twice"


def test_restrict_subtracts_from_an_additive_view():
    """Scope visibility is additive, so removing a tool needs its own mechanism."""
    runtime = Runtime()
    runtime.mount(tools_service)
    runtime.mount(tool_core)
    tools = runtime.services["tools"]
    child = object()

    assert set(tools.names(child)) >= {"get_time", "calculate"}
    restore = tools.restrict(child, ["get_time"])
    assert tools.names(child) == ["get_time"]
    assert set(tools.names()) >= {"get_time", "calculate"}, "the parent was restricted too"

    restore()
    assert set(tools.names(child)) >= {"get_time", "calculate"}


# --- the subagent registry ---------------------------------------------------------------------


class FakeProvider(SubagentProvider):
    def __init__(self, name="fake", caps=None, output="done"):
        self.name = name
        self.capabilities = caps or Capabilities(depth_limit=True, tool_filter=True)
        self.output = output
        self.requests = []

    def start(self, ctx, request):
        self.requests.append(request)
        return SubagentResult(label=request.label, output=self.output, steps=1)


@pytest.fixture
def sub_rt():
    runtime = Runtime()
    runtime.mount(tools_service)
    runtime.mount(subagents_service)
    return runtime


def test_several_providers_coexist_and_are_chosen_by_name(sub_rt):
    """Shaped after the LLM adapter registry, not the single-provider fs seam."""
    svc = sub_rt.services["subagents"]
    svc.register_provider(FakeProvider("in-process", output="A"))
    svc.register_provider(FakeProvider("remote", output="B"))

    assert sorted(svc.providers()) == ["in-process", "remote"]
    assert svc.start(SubagentRequest(task="t"), provider="remote").output == "B"


def test_an_unsupported_capability_is_refused_loudly(sub_rt):
    """Silently ignoring a SAFETY option is the worst kind of silent degradation."""
    svc = sub_rt.services["subagents"]
    svc.register_provider(FakeProvider(caps=Capabilities(depth_limit=True, tool_filter=False)))

    with pytest.raises(UnsupportedCapability, match="cannot restrict"):
        svc.start(SubagentRequest(task="t", allowed_tools=["read_file"]))


def test_a_provider_without_depth_support_is_refused(sub_rt):
    svc = sub_rt.services["subagents"]
    svc.register_provider(FakeProvider(caps=Capabilities(depth_limit=False)))
    with pytest.raises(UnsupportedCapability, match="depth limit"):
        svc.start(SubagentRequest(task="t", max_depth=3))


def test_the_depth_limit_stops_runaway_delegation(sub_rt):
    svc = sub_rt.services["subagents"]
    svc.register_provider(FakeProvider())
    with pytest.raises(SubagentError, match="depth limit reached"):
        svc.start(SubagentRequest(task="t", depth=3, max_depth=3))


def test_a_delegation_cycle_is_caught_even_within_the_depth_limit(sub_rt):
    """Three agents in a ring never exceed depth 1 each while looping forever."""
    svc = sub_rt.services["subagents"]
    svc.register_provider(FakeProvider())
    with pytest.raises(SubagentError, match="cycle"):
        svc.start(SubagentRequest(task="t", label="search",
                                  parents=("root", "search", "verify"), depth=1, max_depth=5))


def test_no_provider_mounted_is_a_clear_error(sub_rt):
    with pytest.raises(SubagentError, match="no subagent provider"):
        sub_rt.services["subagents"].start(SubagentRequest(task="t"))


# --- the in-process provider, end to end ---------------------------------------------------------


class ScriptedAdapter(LlmAdapter):
    """Replies differently to the parent and the child, keyed on the system prompt."""

    provider = "fake"

    def __init__(self, parent_script, child_reply="I looked and found three matches."):
        self.parent_script = list(parent_script)
        self.child_reply = child_reply
        self.child_tool_schemas = None

    def stream(self, options):
        is_child = any("sub-agent" in (m.content or "") for m in options.messages
                       if m.role == "system")
        if is_child:
            self.child_tool_schemas = [t["name"] for t in options.tools]
            yield TextDelta(index=0, text=self.child_reply)
            yield Finish(reason="stop")
            return

        chunks = self.parent_script.pop(0)
        yield from chunks


def delegating_turn(task="find the date parsing", tools=None):
    args = {"task": task, "label": "search"}
    if tools is not None:
        args["tools"] = tools
    return [
        ToolCallDelta(index=0, id="c1", name="task", arguments_delta=json.dumps(args)),
        Finish(reason="tool_calls"),
    ]


def final_turn(text="The parsing lives in dates.py."):
    return [TextDelta(index=0, text=text), Finish(reason="stop")]


def build(parent_script, **subagent_config):
    rt = Runtime()
    rt.mount(llm_service)
    rt.mount(tools_service)
    rt.mount(sessions_service)
    rt.mount(subagents_service)
    rt.mount(tool_core)
    adapter = ScriptedAdapter(parent_script)
    rt.services["llm"].register_adapter(["fake"], adapter)
    rt.mount(agent_loop_service, {"model": "m", "provider": "fake", "system": "parent"})
    rt.mount(subagent_in_process, {"model": "m", "provider": "fake"})
    rt.mount(tool_subagent, subagent_config)
    return rt, adapter


def test_a_parent_delegates_and_gets_a_summary_back():
    rt, _ = build([delegating_turn(), final_turn()])
    answer = rt.services["agent_loop"].run("where is date parsing?")

    assert answer == "The parsing lives in dates.py."
    results = [e for e in rt.services["sessions"].events() if e.kind == "tool/result"]
    assert "three matches" in results[0].data["content"]


def test_the_childs_tool_calls_do_not_land_in_the_parents_history():
    """The entire point of delegating: hand over a task, get back a paragraph."""
    rt, _ = build([delegating_turn(), final_turn()])
    rt.services["agent_loop"].run("go")

    parent_messages = rt.services["sessions"].derive_messages()
    rendered = " ".join(m.content for m in parent_messages)
    assert "I looked and found" in rendered          # the summary IS there
    assert rendered.count("I looked and found") == 1  # ...once, not per child step


def test_the_child_sees_only_the_tools_it_was_allowed():
    rt, adapter = build([delegating_turn(tools=["get_time"]), final_turn()])
    rt.services["agent_loop"].run("go")

    assert adapter.child_tool_schemas == ["get_time"], (
        f"the child saw {adapter.child_tool_schemas}, not the restricted set"
    )


def test_the_parents_tools_are_unaffected_by_the_childs_restriction():
    rt, _ = build([delegating_turn(tools=["get_time"]), final_turn()])
    rt.services["agent_loop"].run("go")
    assert set(rt.services["tools"].names()) >= {"get_time", "calculate", "task"}


def test_the_childs_scope_is_cleaned_up_afterwards():
    rt, _ = build([delegating_turn(), final_turn()])
    before = len(rt.services["tools"]._registry)
    rt.services["agent_loop"].run("go")
    assert len(rt.services["tools"]._registry) == before, "the child's scope leaked"


def test_a_refused_delegation_comes_back_as_a_readable_result_not_a_crash():
    """The model must be able to read the refusal and do the work itself instead."""
    rt, _ = build([delegating_turn(), final_turn()], max_depth=0)
    rt.services["agent_loop"].run("go")

    results = [e for e in rt.services["sessions"].events() if e.kind == "tool/result"]
    assert "Delegation refused" in results[0].data["content"]


def test_a_child_failure_does_not_kill_the_parent_turn():
    rt, adapter = build([delegating_turn(), final_turn()])

    def boom(options):
        raise RuntimeError("child exploded")
        yield  # pragma: no cover

    original = adapter.stream

    def route(options):
        if any("sub-agent" in (m.content or "") for m in options.messages if m.role == "system"):
            return boom(options)
        return original(options)

    adapter.stream = route
    answer = rt.services["agent_loop"].run("go")
    assert answer == "The parsing lives in dates.py.", "the parent turn died with the child"
