"""Count tokens, as a plugin on the `llm/stream` waterfall.

Possible only because phase 6 added `usage` chunks. Two jobs:

  1. tell you what a turn cost
  2. give phase 8's compaction a number to threshold on

WHY A WATERFALL LISTENER AND NOT A CALL IN THE LOOP

    Counting is an around-dispatch concern: it needs to see every model call, including ones
    the loop does not make -- a compaction summary, a session-title generation, a subagent's
    turn. A counter in the agent loop would miss all three and silently under-report.

    Note it wraps the stream lazily rather than draining it. Draining would break streaming for
    every consumer downstream, which is the obvious way to write this and completely wrong.
"""

from __future__ import annotations

from typing import Iterator

from dshpy.services.llm import GenerateOptions, LlmAdapter, StreamChunk, Usage, UsageChunk

name = "token-meter"
inject = ["llm"]


class TokenMeter:
    def __init__(self) -> None:
        self.total = Usage()
        self.calls = 0
        self.last = Usage()

    def record(self, usage: Usage) -> None:
        self.total = self.total + usage
        self.last = usage
        self.calls += 1

    def reset(self) -> None:
        self.total = Usage()
        self.calls = 0


def apply(ctx, config=None) -> None:
    meter = TokenMeter()
    ctx.provide("tokens", meter)

    def counting(options: GenerateOptions, adapter: LlmAdapter, next) -> Iterator[StreamChunk]:
        def wrapped() -> Iterator[StreamChunk]:
            for chunk in next():
                if isinstance(chunk, UsageChunk):
                    meter.record(chunk.usage)
                    ctx.emit("llm/usage", chunk.usage, meter.total)
                yield chunk  # pass it on -- never swallow, never buffer
        return wrapped()

    ctx.on("llm/stream", counting)
