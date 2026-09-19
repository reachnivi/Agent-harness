"""Tests for the tool execution pipeline — the pattern's showcase.

Each test is one extension point doing its job *without the loop or the tool knowing it exists*.
That is the property being verified: not "the code runs", but "the code runs and the thing it
wrapped was never modified to accommodate it".
"""

from __future__ import annotations

import types

import pytest

from dshpy.core.context import Runtime
from dshpy.plugins import tool_core
from dshpy.services import tools as tools_service
from dshpy.services.tools import Deny, ToolResult, define_tool


def plugin(name, inject=(), apply=None):
    mod = types.SimpleNamespace()
    mod.name = name
    mod.inject = list(inject)
    mod.apply = apply or (lambda ctx, config=None: None)
    return mod


@pytest.fixture
def rt():
    runtime = Runtime()
    runtime.mount(tools_service)
    runtime.mount(tool_core)
    return runtime


def run(rt, tool="calculate", args=None, call_id="c1"):
    return rt.services["tools"].execute(call_id, tool, args or {"expression": "6 * 7"})


# --- the baseline -------------------------------------------------------------------------


def test_a_tool_registered_by_a_plugin_is_callable(rt):
    assert run(rt).content == "42"


def test_tool_schemas_are_provider_neutral(rt):
    schema = next(s for s in rt.services["tools"].schemas() if s["name"] == "calculate")
    # Neither OpenAI's {type:"function", function:{...}} nor Anthropic's input_schema.
    assert set(schema) == {"name", "description", "parameters"}
    assert schema["parameters"]["required"] == ["expression"]


def test_unmounting_the_tool_plugin_unregisters_its_tools():
    runtime = Runtime()
    runtime.mount(tools_service)
    entry = runtime.mount(tool_core)
    assert "calculate" in runtime.services["tools"].names()

    runtime.unmount(entry)
    assert runtime.services["tools"].names() == []


def test_an_unknown_tool_is_an_error_result_not_an_exception(rt):
    result = run(rt, tool="nope", args={})
    assert result.is_error and "unknown tool" in result.content


def test_a_raising_tool_becomes_an_error_result(rt):
    result = run(rt, args={"expression": "1/0"})
    assert result.is_error and "Error:" in result.content


# --- stage 1: tools/pre-execute (reorderable policy) ---------------------------------------


def test_pre_execute_deny_skips_the_tool_body(rt):
    ran = []
    rt.mount(plugin("blocker", inject=["tools"], apply=lambda ctx, cfg=None: ctx.on(
        "tools/pre-execute", lambda exec_, next: Deny("blocked by policy"))))
    rt.mount(plugin("spy", inject=["tools"], apply=lambda ctx, cfg=None: ctx.on(
        "tools/execute", lambda exec_, next: ran.append(1) or next())))

    result = run(rt)
    assert result.is_error
    assert "blocked by policy" in result.content
    assert ran == [], "the tool body ran despite a denial"


def test_pre_execute_policy_is_reorderable_an_outer_listener_can_allow(rt):
    """Policy composes: a later-registered listener runs first and may decline to delegate."""
    rt.mount(plugin("strict", inject=["tools"], apply=lambda ctx, cfg=None: ctx.on(
        "tools/pre-execute", lambda exec_, next: Deny("strict says no"))))
    # Registered with prepend -> runs before "strict" and short-circuits it.
    rt.mount(plugin("override", inject=["tools"], apply=lambda ctx, cfg=None: ctx.on(
        "tools/pre-execute", lambda exec_, next: None, prepend=True)))

    assert run(rt).content == "42"


def test_a_gate_sees_the_call_it_is_gating(rt):
    seen = {}
    rt.mount(plugin("inspector", inject=["tools"], apply=lambda ctx, cfg=None: ctx.on(
        "tools/pre-execute",
        lambda exec_, next: seen.update(name=exec_.name, args=exec_.arguments) or next())))
    run(rt)
    assert seen["name"] == "calculate"
    assert seen["args"] == {"expression": "6 * 7"}


# --- stage 2: guards (monotonic) -----------------------------------------------------------


def test_a_guard_denies_and_cannot_be_overridden_by_later_policy(rt):
    """THE distinction that keeps permission systems from growing holes."""
    rt.mount(plugin("invariant", inject=["tools"], apply=lambda ctx, cfg=None: ctx.effect(
        ctx.tools.guard(lambda exec_: Deny("invariant violated")))))
    # Mounted AFTER the guard and prepended -- the strongest an attacker-ordered policy can be.
    rt.mount(plugin("permissive", inject=["tools"], apply=lambda ctx, cfg=None: ctx.on(
        "tools/pre-execute", lambda exec_, next: None, prepend=True)))

    result = run(rt)
    assert result.is_error
    assert "invariant violated" in result.content
    assert result.meta["denied_by"] == "guard"


def test_a_guard_that_abstains_lets_the_call_through(rt):
    rt.mount(plugin("abstainer", inject=["tools"], apply=lambda ctx, cfg=None: ctx.effect(
        ctx.tools.guard(lambda exec_: None))))
    assert run(rt).content == "42"


# --- stage 3: tools/execute (around-dispatch) ----------------------------------------------


def test_execute_wraps_the_body_on_both_sides(rt):
    order = []

    def wrap(ctx, cfg=None):
        def around(exec_, next):
            order.append("before")
            result = next()
            order.append("after")
            return result
        ctx.on("tools/execute", around)

    rt.mount(plugin("timer", inject=["tools"], apply=wrap))
    assert run(rt).content == "42"
    assert order == ["before", "after"], "around-dispatch did not wrap the body"


def test_execute_wrapper_can_replace_the_result_entirely(rt):
    rt.mount(plugin("cache", inject=["tools"], apply=lambda ctx, cfg=None: ctx.on(
        "tools/execute",
        lambda exec_, next: ToolResult(exec_.call_id, exec_.name, "cached!"))))
    assert run(rt).content == "cached!"


def test_a_wrapper_returning_a_non_result_is_normalized(rt):
    """A sloppy plugin must not corrupt the message history the loop builds downstream."""
    rt.mount(plugin("sloppy", inject=["tools"], apply=lambda ctx, cfg=None: ctx.on(
        "tools/execute", lambda exec_, next: "bare string")))
    result = run(rt)
    assert isinstance(result, ToolResult) and result.content == "bare string"


# --- stage 4: tools/post-execute (transform) -----------------------------------------------


def test_post_execute_can_rewrite_the_result(rt):
    rt.mount(plugin("redact", inject=["tools"], apply=lambda ctx, cfg=None: ctx.on(
        "tools/post-execute",
        lambda exec_, result, next: ToolResult(result.call_id, result.name, "[redacted]"))))
    assert run(rt).content == "[redacted]"


def test_post_execute_sees_the_real_result_before_rewriting_it(rt):
    seen = {}

    def observe(ctx, cfg=None):
        def after(exec_, result, next):
            seen["content"] = result.content
            return next()
        ctx.on("tools/post-execute", after)

    rt.mount(plugin("observer", inject=["tools"], apply=observe))
    run(rt)
    assert seen["content"] == "42"


# --- stage 5: tools/result (observe only) --------------------------------------------------


def test_result_observers_run_but_cannot_change_the_outcome(rt):
    seen = []

    def watch(ctx, cfg=None):
        def observer(exec_, result):
            seen.append(result.content)
            result.content = "mutated"  # allowed on the object, but nothing re-reads it
        ctx.on("tools/result", observer)

    rt.mount(plugin("telemetry", inject=["tools"], apply=watch))
    result = run(rt)
    assert seen == ["42"]
    # The caller already holds the object; the contract is that observation is the LAST stage,
    # so no further pipeline stage consults it. Documented and asserted rather than enforced.
    assert result.content == "mutated"


def test_a_raising_observer_does_not_break_the_call(rt):
    rt.mount(plugin("bad", inject=["tools"], apply=lambda ctx, cfg=None: ctx.on(
        "tools/result", lambda exec_, result: (_ for _ in ()).throw(RuntimeError("boom")))))
    assert run(rt).content == "42"


def test_observers_see_denied_calls_too(rt):
    seen = []
    rt.mount(plugin("deny", inject=["tools"], apply=lambda ctx, cfg=None: ctx.on(
        "tools/pre-execute", lambda exec_, next: Deny("no"))))
    rt.mount(plugin("watch", inject=["tools"], apply=lambda ctx, cfg=None: ctx.on(
        "tools/result", lambda exec_, result: seen.append(result.is_error))))
    run(rt)
    assert seen == [True], "a denial must still be observable for audit"


# --- ordering ------------------------------------------------------------------------------


def test_the_full_pipeline_runs_in_the_documented_order(rt):
    order = []

    def everything(ctx, cfg=None):
        ctx.on("tools/pre-execute", lambda e, next: order.append("pre") or next())
        ctx.effect(ctx.tools.guard(lambda e: order.append("guard") or None))
        ctx.on("tools/execute", lambda e, next: order.append("execute") or next())
        ctx.on("tools/post-execute", lambda e, r, next: order.append("post") or next())
        ctx.on("tools/result", lambda e, r: order.append("result"))

    rt.mount(plugin("all", inject=["tools"], apply=everything))
    run(rt)
    assert order == ["pre", "guard", "execute", "post", "result"]
