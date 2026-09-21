"""A deadline around every tool call, using real cancellation.

WHAT CHANGED FROM PHASE 5

    The first version used a daemon thread: the *call* returned on time, but the work kept
    running. I flagged it as an honest simplification; this is the fix.

    Now the plugin sets a deadline on the execution's `CancelToken`. Long-running tool bodies --
    the HTTP read loop, a subprocess poll, a directory walk -- check the token and stop. The
    work actually stops, instead of continuing with nobody listening.

    The trade is explicit and worth stating: cancellation is COOPERATIVE. A tool that never
    polls its token cannot be interrupted, and no amount of harness design fixes that, because
    Python has no safe way to interrupt an arbitrary thread. What the harness can do is make
    the token available everywhere and make polling the obvious thing to write -- which is why
    `exec.token` is handed to every tool body.

WHY `tools/execute` AND NOT INSIDE THE TOOLS

    Around-dispatch concerns wrap the body without the body participating. Every tool gets a
    deadline, including ones written later by someone who never heard of this plugin.
"""

from __future__ import annotations

from dshpy.cancel import Cancelled, CancelToken
from dshpy.services.tools import ToolResult

name = "timeout"
inject = ["tools"]


def apply(ctx, config=None) -> None:
    seconds = (config or {}).get("seconds", 30.0)

    def around(exec_, next):
        # Give this execution a token with a deadline, linked to any outer token so an outer
        # cancellation still wins. The tool body reads it as `exec.token`.
        outer = exec_.token
        deadline = outer.child(timeout=seconds) if outer is not None else CancelToken(
            timeout=seconds
        )
        exec_.token = deadline
        try:
            result = next()
            # The pipeline's own handler usually catches Cancelled first and has already
            # turned it into an error result — so the job here is to ANNOTATE rather than to
            # catch. Two layers both converting the same exception would be redundant, and the
            # one that ran first would silently win.
            if (deadline.is_set() and isinstance(result, ToolResult) and result.is_error):
                result.meta["timed_out"] = True
                result.meta["limit_seconds"] = seconds
            return result
        except Cancelled as exc:
            # Reached only if something above the tool body raised — e.g. a wrapper that
            # polls the token itself.
            return ToolResult(exec_.call_id, exec_.name,
                              f"Error: {exc.reason} (limit {seconds}s)", is_error=True,
                              meta={"timed_out": True, "limit_seconds": seconds})
        finally:
            exec_.token = outer

    ctx.on("tools/execute", around)
