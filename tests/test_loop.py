"""Tests for stage 3's agent loop — one test per rule, plus the calculator's safety.

Run: uv run pytest -v

These need no API key, no model and no network, and finish in milliseconds. Run them after
every refactor from stage 4 onward: they are how you find out that a "harmless" cleanup quietly
broke rule 1.
"""

from __future__ import annotations

import pytest

from stages.s03_loop import calculate, dispatch, run_agent
from tests.fake_client import FakeClient, says, wants

TOOLS: list[dict] = [{"name": "get_time", "description": "", "input_schema": {}}]


def _run(script, dispatch_fn=dispatch, **kwargs):
    client = FakeClient(script)
    history = run_agent(
        client,
        "go",
        tools=TOOLS,
        dispatch_fn=dispatch_fn,
        verbose=False,
        **kwargs,
    )
    return client, history


def _results_in(message: dict) -> list[dict]:
    return [b for b in message["content"] if b["type"] == "tool_result"]


# --- the happy path ------------------------------------------------------------------------


def test_tool_is_executed_and_its_result_is_sent_back():
    client, history = _run(
        [wants(("call_1", "calculate", {"expression": "17 * 23"})), says("It's 391.")]
    )

    assert len(client.calls) == 2, "expected one call to ask, one to deliver the result"

    # history: user, assistant(tool_use), user(tool_result), assistant(text)
    results = _results_in(history[2])
    assert len(results) == 1
    assert results[0]["content"] == "391"
    assert results[0]["tool_use_id"] == "call_1"


def test_loop_stops_when_model_stops_asking_for_tools():
    client, history = _run([says("No tools needed.")])
    assert len(client.calls) == 1
    assert len(history) == 2


# --- rule 1: all results for one turn go in ONE user message -------------------------------


def test_parallel_tool_results_land_in_a_single_user_message():
    client, history = _run(
        [
            wants(
                ("call_1", "calculate", {"expression": "17 * 23"}),
                ("call_2", "calculate", {"expression": "2 + 2"}),
            ),
            says("391 and 4."),
        ]
    )

    user_turns_after_first = [m for m in history[1:] if m["role"] == "user"]
    assert len(user_turns_after_first) == 1, (
        "rule 1 violated: results were split across multiple user messages"
    )
    assert len(_results_in(user_turns_after_first[0])) == 2


# --- rule 2: every tool_use gets a result --------------------------------------------------


def test_every_tool_use_block_gets_exactly_one_result():
    _, history = _run(
        [
            wants(
                ("call_1", "calculate", {"expression": "1 + 1"}),
                ("call_2", "get_time", {}),
                ("call_3", "calculate", {"expression": "3 * 3"}),
            ),
            says("done"),
        ]
    )

    requested = [b.id for b in history[1]["content"] if b.type == "tool_use"]
    answered = [r["tool_use_id"] for r in _results_in(history[2])]
    assert sorted(answered) == sorted(requested)


# --- rule 3: failures are results, not exceptions ------------------------------------------


def test_tool_error_becomes_an_is_error_result_and_the_loop_continues():
    client, history = _run(
        [wants(("call_1", "calculate", {"expression": "1/0"})), says("That's undefined.")]
    )

    result = _results_in(history[2])[0]
    assert result["is_error"] is True
    assert "Error:" in result["content"]
    assert len(client.calls) == 2, "the loop must keep going so the model can recover"


def test_unknown_tool_is_reported_to_the_model_rather_than_crashing():
    _, history = _run([wants(("call_1", "nonexistent", {})), says("Sorry.")])

    result = _results_in(history[2])[0]
    assert result["is_error"] is True
    assert "unknown tool" in result["content"]


def test_a_failing_tool_does_not_prevent_its_siblings_from_running():
    _, history = _run(
        [
            wants(
                ("call_1", "calculate", {"expression": "1/0"}),
                ("call_2", "calculate", {"expression": "6 * 7"}),
            ),
            says("done"),
        ]
    )

    by_id = {r["tool_use_id"]: r for r in _results_in(history[2])}
    assert by_id["call_1"].get("is_error") is True
    assert by_id["call_2"]["content"] == "42"
    assert "is_error" not in by_id["call_2"]


# --- rule 4: the loop is bounded -----------------------------------------------------------


def test_max_steps_caps_a_model_that_never_stops():
    never_stops = [wants((f"call_{i}", "get_time", {})) for i in range(50)]
    client, _ = _run(never_stops, max_steps=3)
    assert len(client.calls) == 3


# --- the stage 1 gotcha --------------------------------------------------------------------


def test_assistant_turns_store_blocks_not_flattened_text():
    """Flattening to a string drops tool_use blocks and corrupts the history."""
    _, history = _run([wants(("call_1", "get_time", {})), says("done")])

    assistant_turn = history[1]
    assert not isinstance(assistant_turn["content"], str)
    assert any(b.type == "tool_use" for b in assistant_turn["content"])


# --- the calculator is an allowlist, not eval() --------------------------------------------


@pytest.mark.parametrize(
    "expression,expected",
    [("17 * 23", "391"), ("2 + 3 * 4", "14"), ("(2 + 3) ** 3", "125"), ("-7 + 2", "-5")],
)
def test_calculate_does_arithmetic(expression, expected):
    assert calculate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo pwned')",  # the reason this isn't eval()
        "open('/etc/passwd').read()",
        "[].__class__",
        "print(1)",
        "x + 1",
        "'a' * 3",
    ],
)
def test_calculate_rejects_anything_that_is_not_arithmetic(expression):
    with pytest.raises(ValueError):
        calculate(expression)


def test_calculate_rejects_a_denial_of_service_exponent():
    with pytest.raises(ValueError, match="too large"):
        calculate("2 ** 100000000")
