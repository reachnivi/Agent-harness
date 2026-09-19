"""A deadline around every tool call, as a plugin.

Why this belongs on `tools/execute` and not in the tools: around-dispatch concerns wrap the
body without the body participating. Every tool gets a deadline, including ones written later
by someone who never heard of this plugin.

The thread here is a deliberate simplification: a real implementation would pass a cancellation
signal into the tool (dsh's `exec.signal`) so the work actually stops. A daemon thread lets the
*call* return on time but leaves the work running, which is honest for a learning harness and
would not be acceptable in production. Noted rather than hidden.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

from dshpy.services.tools import ToolResult

name = "timeout"
inject = ["tools"]


def apply(ctx, config=None) -> None:
    seconds = (config or {}).get("seconds", 30.0)
    pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool")
    ctx.effect(lambda: pool.shutdown(wait=False))

    def around(exec_, next):
        try:
            return pool.submit(next).result(timeout=seconds)
        except FutureTimeout:
            return ToolResult(exec_.call_id, exec_.name,
                              f"Error: tool timed out after {seconds}s", is_error=True,
                              meta={"timed_out": True})

    ctx.on("tools/execute", around)
