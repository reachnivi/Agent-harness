"""The built-in tools, contributed as a plugin.

Contrast with stages/s03_loop.py, where adding a tool meant editing TOOLS *and* dispatch() --
two places, in the loop's own file. Here it is one registration in a plugin the loop never
mentions, and unmounting this plugin removes the tools cleanly.

The implementations come from common/tools.py unchanged. A tool is a function; the registry,
the wire format and the policy around it are all somebody else's problem.
"""

from __future__ import annotations

from common.tools import calculate, get_time
from dshpy.services.tools import define_tool

name = "tool-core"
inject = ["tools"]


def apply(ctx, config=None) -> None:
    ctx.effect(ctx.tools.register(define_tool(
        name="get_time",
        description=(
            "Get the current date and time in ISO 8601 format. "
            "Use this whenever the user asks about the current time, date, or day of the week."
        ),
        parameters={},
        execute=lambda args, exec: get_time(),
    )))

    ctx.effect(ctx.tools.register(define_tool(
        name="calculate",
        description=(
            "Evaluate an arithmetic expression and return the result. "
            "Supports + - * / // % ** and parentheses. Use this instead of doing mental math."
        ),
        parameters={
            "expression": {
                "type": "string",
                "required": True,
                "description": "An arithmetic expression, e.g. '17 * 23' or '(2+3)**4'.",
            }
        },
        execute=lambda args, exec: calculate(args["expression"]),
    )))
