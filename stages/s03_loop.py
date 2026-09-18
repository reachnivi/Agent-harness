"""Stage 3 — The agentic loop.

CONCEPT:

    Wrap stage 2 in a loop and you have an agent. That is genuinely it.

    Everything after this stage — registries, streaming, permissions, subagents, context
    compaction — is ergonomics, safety, or performance. The engine is the ~15 lines in
    `run_agent()` below, and if you can rewrite those from memory you understand agents.

THE FOUR RULES (you will otherwise rediscover these painfully):

    1. ALL tool_result blocks for one assistant turn go in ONE user message.
       Splitting them across several messages quietly teaches the model to stop making
       parallel calls, and you'll never get an error telling you that's what happened.

    2. EVERY tool_use gets a result. Drop one and the API rejects the next request.

    3. Tool failures are RESULTS (is_error=True), not exceptions.
       The model reads the error text and adapts — retries with different arguments, or tells
       the user what went wrong. Raising instead kills that recovery loop dead.

    4. BOUND the loop. An unbounded agent loop is an unbounded bill (or, locally, an
       unbounded afternoon).

RUN
    uv run stages/s03_loop.py
    uv run stages/s03_loop.py "what is 17 * 23, and what time is it?"

    Watch for ONE assistant turn containing TWO tool_use blocks. That's parallel tool calling,
    and rule 1 is what keeps it working.

TRY BREAKING IT
    1. Break rule 1: append each tool_result as its own message. It still works! For a while.
       Then the model stops batching calls and everything gets slower. This is the nastiest
       bug class in agent work — no error, just quiet degradation.
    2. Break rule 3: delete the try/except so tool errors raise. Ask it to calculate "1/0".
       Compare crashing against what happens when the error goes back as a result.
    3. Break rule 4: set MAX_STEPS=1000 and give it something open-ended. Watch it churn.
    4. Add a third tool of your own. Notice you have to touch TOOLS *and* dispatch() — two
       places, which is exactly the duplication stage 4 removes.

NOTE
    `run_agent()` takes `client` and `dispatch` as arguments rather than reaching for globals.
    That isn't ceremony — it's what lets tests/test_loop.py drive this exact function with a
    scripted fake client, no model involved. Loop bugs are brutal to find when every run is
    non-deterministic; those tests make them deterministic and instant.
"""

from __future__ import annotations

import ast
import operator
import sys
from datetime import datetime
from typing import Any, Callable

from harness.client import MAX_TOKENS, MODEL, PreflightError, build_client, preflight

MAX_STEPS = 8  # rule 4

# ---------------------------------------------------------------------------
# Tools, as the model sees them.
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "get_time",
        "description": (
            "Get the current date and time in ISO 8601 format. "
            "Use this whenever the user asks about the current time, date, or day of the week."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "calculate",
        "description": (
            "Evaluate an arithmetic expression and return the result. "
            "Supports + - * / // % ** and parentheses. Use this instead of doing mental math."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "An arithmetic expression, e.g. '17 * 23' or '(2+3)**4'.",
                }
            },
            "required": ["expression"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tools, as your code sees them.
# ---------------------------------------------------------------------------
def get_time() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


_OPS: dict[type, Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

MAX_EXPONENT = 1000  # 2 ** 10**9 is not a calculation, it's a denial of service


def calculate(expression: str) -> str:
    """Evaluate arithmetic safely.

    NOT eval(). The expression is model-generated text, which makes it untrusted input by
    definition — and eval() on untrusted input is arbitrary code execution. The model doesn't
    have to be malicious for this to matter; it only has to be wrong, or to be repeating
    something a user pasted at it.

    So: parse to an AST, then walk it allowing *only* numeric literals and arithmetic operators.
    Anything else — names, calls, attributes, subscripts — is rejected. This is an allowlist,
    which is the only kind of input validation that actually works.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"not a valid expression: {expression!r}") from exc
    return str(_eval_node(tree.body))


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"only numbers allowed, got {node.value!r}")
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError(f"exponent {right} too large (max {MAX_EXPONENT})")
        return _OPS[type(node.op)](left, right)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))

    raise ValueError(f"unsupported expression element: {type(node).__name__}")


def dispatch(name: str, tool_input: dict) -> str:
    """Map a tool name from the model onto a real function.

    This raises on an unknown tool -- and that is fine, because rule 3 means the loop catches it
    and hands the message back to the model as an is_error result. The model occasionally
    hallucinates a tool that doesn't exist; being told so is far more useful than a crash.
    Note how rule 3 lets every layer below the loop just raise normally.
    """
    if name == "get_time":
        return get_time()
    if name == "calculate":
        return calculate(tool_input["expression"])
    raise ValueError(f"unknown tool {name!r}. Available: get_time, calculate")


# ---------------------------------------------------------------------------
# THE LOOP. This is the whole lesson.
# ---------------------------------------------------------------------------
def run_agent(
    client: Any,
    user_input: str,
    *,
    tools: list[dict],
    dispatch_fn: Callable[[str, dict], str],
    model: str = MODEL,
    max_steps: int = MAX_STEPS,
    verbose: bool = True,
) -> list[dict]:
    """Run until the model stops asking for tools. Returns the full message history."""
    messages: list[dict] = [{"role": "user", "content": user_input}]

    for step in range(max_steps):  # rule 4: bounded
        resp = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            tools=tools,
            messages=messages,
        )

        # Always record the assistant turn, tool_use blocks and all (the stage 1 gotcha).
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            if verbose:
                print(f"\nbot> {_text_of(resp.content)}")
            return messages

        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        if verbose:
            narration = _text_of(resp.content).strip()
            if narration:
                print(f"bot> {narration}")
            print(f"[step {step + 1}: {len(tool_uses)} tool call(s)]")

        # Rule 2: every tool_use gets a result. One iteration, one result, no early exits.
        results = []
        for block in tool_uses:
            try:
                output = dispatch_fn(block.name, block.input)
                if verbose:
                    print(f"  {block.name}({block.input}) -> {output}")
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": output}
                )
            except Exception as exc:  # rule 3: failures are results, not exceptions
                if verbose:
                    print(f"  {block.name}({block.input}) -> ERROR: {exc}")
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Error: {exc}",
                        "is_error": True,
                    }
                )

        # Rule 1: ONE user message carrying ALL the results.
        messages.append({"role": "user", "content": results})

    if verbose:
        print(f"\n[hit max_steps={max_steps} — giving up]")
    return messages


def _text_of(content: list) -> str:
    return "".join(block.text for block in content if block.type == "text")


def main(argv: list[str]) -> int:
    try:
        preflight()
    except PreflightError as exc:
        print(f"\nPreflight failed:\n{exc}\n", file=sys.stderr)
        return 1

    question = argv[1] if len(argv) > 1 else "What is 17 * 23, and what time is it?"
    print(f"you> {question}\n")

    run_agent(
        build_client(),
        question,
        tools=TOOLS,
        dispatch_fn=dispatch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
