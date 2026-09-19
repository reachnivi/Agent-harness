"""Observation, as a plugin. The smallest possible demonstration of `emit`.

It listens and never participates: `tools/result` is a notification, so nothing here can change
an outcome. That is the right mode for logging, metrics and UI -- and the EventBus deliberately
swallows exceptions from emit listeners so a broken observer cannot break the thing it watches.
"""

from __future__ import annotations

import time

name = "telemetry"
inject: list[str] = []


def apply(ctx, config=None) -> None:
    verbose = (config or {}).get("verbose", True)
    counts = {"calls": 0, "errors": 0}
    ctx.effect(lambda: verbose and print(
        f"[telemetry] {counts['calls']} tool call(s), {counts['errors']} error(s)"))

    started: dict[str, float] = {}

    def on_call(exec_, next):
        started[exec_.call_id] = time.monotonic()
        return next()

    def on_result(exec_, result):
        counts["calls"] += 1
        counts["errors"] += int(result.is_error)
        if verbose:
            elapsed = time.monotonic() - started.pop(exec_.call_id, time.monotonic())
            mark = "!" if result.is_error else " "
            preview = result.content if len(result.content) <= 60 else result.content[:57] + "..."
            print(f"  {mark} {exec_.name}({exec_.arguments}) -> {preview}  [{elapsed*1000:.0f}ms]")

    ctx.on("tools/pre-execute", on_call, prepend=True)
    ctx.on("tools/result", on_result)
