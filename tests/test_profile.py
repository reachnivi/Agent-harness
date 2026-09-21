"""Tests for profile composition and the policy plugins.

The headline here is `test_profile_row_order_does_not_matter`. The default profile's docstring
claims you can shuffle its rows freely because `inject` decides activation order. That is
exactly the kind of claim that quietly stops being true, so it gets a test.
"""

from __future__ import annotations

import random

import pytest

from dshpy.core.context import Runtime
from dshpy.plugins import llm_openai, permission, telemetry, timeout, tool_core
from dshpy.profiles import default as default_profile
from dshpy.services import agent_loop, llm, sessions, tools
from dshpy.services.tools import Deny, ToolResult


def base_rows():
    return [
        {"plugin": llm}, {"plugin": tools}, {"plugin": sessions},
        {"plugin": llm_openai, "config": {"routes": ["openai"]}},
        {"plugin": tool_core},
        {"plugin": agent_loop, "config": {"model": "m", "provider": "openai"}},
    ]


def test_the_default_profile_boots_completely():
    rt = Runtime()
    rt.mount_all(default_profile.rows())
    for key in ("llm", "tools", "sessions", "agent_loop"):
        assert key in rt.services, f"{key} never activated:\n{rt.dump()}"


def test_profile_row_order_does_not_matter():
    """A profile is a set, not a sequence — the practical payoff of inject."""
    for seed in range(12):
        rows = base_rows()
        random.Random(seed).shuffle(rows)
        rt = Runtime()
        rt.mount_all(rows)
        assert set(rt.services) == {"llm", "tools", "sessions", "agent_loop"}, (
            f"shuffle seed {seed} failed to boot:\n{rt.dump()}"
        )


def test_a_disabled_row_is_not_mounted():
    rows = base_rows() + [{"plugin": tool_core, "disabled": True}]
    rt = Runtime()
    rt.mount_all(rows)
    assert rt.services["tools"].names() == ["get_time", "calculate"]  # mounted exactly once


def test_dropping_the_adapter_row_leaves_the_loop_waiting_not_crashed():
    """A missing capability must be a legible 'never activated', not an AttributeError."""
    rows = [r for r in base_rows() if r["plugin"] is not llm_openai]
    rt = Runtime()
    rt.mount_all(rows)
    assert "agent_loop" in rt.services  # it injects llm the SERVICE, which is present
    assert rt.services["llm"].providers() == []
    with pytest.raises(RuntimeError, match="no LLM adapter is mounted"):
        rt.services["llm"].generate([], tools=[], model="m")


def test_switching_provider_changes_only_the_adapter_row():
    default_profile.PROVIDER = "openai"
    openai_names = {getattr(r["plugin"], "name", "?") for r in default_profile.rows()}
    default_profile.PROVIDER = "anthropic"
    anthropic_names = {getattr(r["plugin"], "name", "?") for r in default_profile.rows()}
    default_profile.PROVIDER = "openai"  # restore

    assert openai_names ^ anthropic_names == {"llm-openai", "llm-anthropic"}, (
        "swapping the provider changed something other than the adapter row"
    )


# --- the permission plugin ------------------------------------------------------------------


def harness(permission_config):
    rt = Runtime()
    rt.mount_all(base_rows())
    rt.mount(permission, permission_config)
    return rt


def run(rt, tool="calculate", args=None):
    return rt.services["tools"].execute("c1", tool, args or {"expression": "6 * 7"})


def test_allow_policy_lets_a_tool_run():
    assert run(harness({"policy": {"calculate": "allow"}})).content == "42"


def test_deny_policy_blocks_and_explains_itself_to_the_model():
    result = run(harness({"policy": {"calculate": "deny"}}))
    assert result.is_error and "denied by policy" in result.content


def test_declining_an_ask_is_a_result_not_an_exception():
    """The model reads 'user declined' and can pick another approach. Rule 3, structurally."""
    rt = harness({"policy": {"calculate": "ask"}, "ask": lambda exec_: False})
    result = run(rt)
    assert result.is_error
    assert "declined" in result.content


def test_accepting_an_ask_runs_the_tool():
    rt = harness({"policy": {"calculate": "ask"}, "ask": lambda exec_: True})
    assert run(rt).content == "42"


def test_the_ask_callback_sees_what_it_is_approving():
    seen = {}
    rt = harness({"policy": {"calculate": "ask"},
                  "ask": lambda exec_: seen.update(name=exec_.name, args=exec_.arguments) or True})
    run(rt)
    assert seen == {"name": "calculate", "args": {"expression": "6 * 7"}}


def test_a_known_tool_keeps_its_verdict_under_a_deny_default():
    # get_time is in DEFAULT_POLICY as allow, so it runs even when default=deny.
    result = run(harness({"default": "deny"}), tool="get_time", args={})
    assert not result.is_error


def test_an_unlisted_tool_falls_back_to_the_default_verdict():
    from dshpy.services.tools import define_tool

    rt = harness({"default": "deny"})
    rt.services["tools"].register(define_tool(
        name="unlisted", description="d", parameters={},
        execute=lambda args, exec: "ran",
    ))
    result = rt.services["tools"].execute("c1", "unlisted", {})
    assert result.is_error and "denied by policy" in result.content


# --- the path-confinement guard --------------------------------------------------------------


@pytest.fixture
def path_tool_rt(tmp_path):
    from dshpy.services.tools import define_tool

    rt = Runtime()
    rt.mount_all(base_rows())
    rt.mount(permission, {"default": "allow", "root": str(tmp_path)})
    rt.services["tools"].register(define_tool(
        name="read_file", description="read",
        parameters={"path": {"type": "string", "required": True}},
        execute=lambda args, exec: f"read {args['path']}",
    ))
    return rt, tmp_path


def test_a_path_inside_the_root_is_allowed(path_tool_rt):
    rt, tmp_path = path_tool_rt
    target = tmp_path / "notes.txt"
    result = rt.services["tools"].execute("c1", "read_file", {"path": str(target)})
    assert not result.is_error


def test_a_path_escaping_the_root_is_denied_by_the_guard(path_tool_rt):
    """Schema validation is not path validation: "string" happily accepts ../../.ssh/..."""
    rt, _ = path_tool_rt
    result = rt.services["tools"].execute("c1", "read_file", {"path": "../../etc/passwd"})
    assert result.is_error
    assert "outside the project root" in result.content
    assert result.meta["denied_by"] == "guard"


def test_the_guard_survives_an_allow_everything_policy(path_tool_rt):
    """The invariant must not be switchable off by configuration. That's why it's a guard."""
    rt, _ = path_tool_rt
    rt.mount(permission, {"default": "allow", "policy": {"read_file": "allow"},
                          "root": "/"})  # a second, maximally permissive permission plugin
    result = rt.services["tools"].execute("c1", "read_file", {"path": "../../etc/passwd"})
    assert result.is_error, "a permissive second policy overrode a monotonic guard"


# --- timeout and telemetry --------------------------------------------------------------------


def test_timeout_interrupts_a_tool_that_polls_its_token():
    """Cancellation is COOPERATIVE: the tool must look at the token for this to work."""
    import time
    from dshpy.services.tools import define_tool

    def polls(args, exec):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            exec.token.check()   # <- the whole contract, in one line
            time.sleep(0.005)
        return "finished"

    rt = Runtime()
    rt.mount_all(base_rows())
    rt.mount(timeout, {"seconds": 0.05})
    rt.services["tools"].register(define_tool(
        name="slow", description="slow", parameters={}, execute=polls))

    started = time.monotonic()
    result = rt.services["tools"].execute("c1", "slow", {})
    elapsed = time.monotonic() - started

    assert result.is_error
    assert result.meta["timed_out"] is True
    assert elapsed < 1.0, "the call should have returned at the deadline, not run to completion"


def test_a_tool_that_ignores_its_token_is_NOT_interrupted():
    """The honest limitation, asserted rather than hidden.

    Python cannot safely interrupt an arbitrary thread, so a tool body that never polls runs
    to completion no matter what the deadline says. Phase 5's daemon-thread version *appeared*
    to handle this by returning early while the work carried on invisibly -- which is worse,
    because the caller believes the work stopped. Failing visibly beats succeeding falsely.
    """
    import time
    from dshpy.services.tools import define_tool

    rt = Runtime()
    rt.mount_all(base_rows())
    rt.mount(timeout, {"seconds": 0.01})
    rt.services["tools"].register(define_tool(
        name="stubborn", description="never polls", parameters={},
        execute=lambda args, exec: (time.sleep(0.1), "done anyway")[1]))

    result = rt.services["tools"].execute("c1", "stubborn", {})
    assert result.content == "done anyway"
    assert not result.is_error


def test_timeout_does_not_interfere_with_a_fast_tool():
    rt = Runtime()
    rt.mount_all(base_rows())
    rt.mount(timeout, {"seconds": 5.0})
    assert run(rt).content == "42"


def test_telemetry_counts_without_changing_anything(capsys):
    rt = Runtime()
    rt.mount_all(base_rows())
    rt.mount(telemetry, {"verbose": True})
    assert run(rt).content == "42"
    assert "calculate" in capsys.readouterr().out
