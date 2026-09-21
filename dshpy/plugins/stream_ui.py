"""Render the model's output as it arrives, as a plugin.

The smallest possible consumer of the raw chunk protocol, and the reason that protocol exists.
The agent loop folds the same stream into a finished message because it cannot continue without
one; this plugin wants the opposite -- every fragment, immediately, and nothing assembled.

Both read the SAME stream. That is what `agent/stream` buys: without it, a UI either waits for
the turn to finish (no streaming) or reimplements the loop (two loops to keep in sync).

Note it prints tool calls by NAME as soon as the name is known, before any argument has
finished streaming. The Anthropic dialect gives the name on block-start, so "calling
calculate..." can appear while its arguments are still arriving -- a small thing that makes an
agent feel responsive instead of stuck.

It also filters on scope. A subagent shares this event bus, so without that check the UI
renders the child's reply AND the parent's, and the user sees two answers to one question with
no way to tell which is authoritative. Sharing a bus across agents makes scope part of the
event's contract, not an optional extra.
"""

from __future__ import annotations

import sys

from dshpy.services.llm import BlockStart, Finish, TextDelta, ToolCallDelta

name = "stream-ui"
inject: list[str] = []


def apply(ctx, config=None) -> None:
    out = (config or {}).get("stream", sys.stdout)
    state = {"open": False, "named": set()}

    def on_chunk(chunk, scope=None) -> None:
        # Only the root agent renders. A subagent's stream belongs to its parent's tool
        # result, not to the user's transcript -- showing it produces two answers to one
        # question, and the user cannot tell which is the real one.
        if scope is not None:
            return
        if isinstance(chunk, TextDelta):
            if not state["open"]:
                out.write("\nbot> ")
                state["open"] = True
            out.write(chunk.text)
            out.flush()

        elif isinstance(chunk, ToolCallDelta) and chunk.name and chunk.id not in state["named"]:
            state["named"].add(chunk.id)
            out.write(f"\n  · calling {chunk.name}…\n")
            out.flush()

        elif isinstance(chunk, Finish):
            if state["open"]:
                out.write("\n")
                out.flush()
            state["open"] = False

    ctx.on("agent/stream", on_chunk)
