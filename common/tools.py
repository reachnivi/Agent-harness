"""Tool implementations, independent of provider *and* of architecture.

Moved here from `stages/s03_loop.py` so that both harnesses can use them:

  - the hand-written Anthropic loop in `stages/s03_loop.py`
  - the plugin harness in `dshpy/plugins/tool_core.py`

That both can share these unchanged is itself part of the lesson. A tool is a function.
Everything else — the wire format that carries the call, the registry that finds it, the policy
that gates it — is the harness's problem, not the tool's. If a tool ever needs to know which
provider it is running under, a seam has been drawn in the wrong place.
"""

from __future__ import annotations

import ast
import operator
from datetime import datetime
from typing import Any, Callable


def get_time() -> str:
    """Current local time, ISO 8601."""
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
