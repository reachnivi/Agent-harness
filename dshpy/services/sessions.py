"""`ctx.sessions` — the append-only event log, and the projection the model actually sees.

dsh's rule: **model-visible means logged.** Anything that reaches a model request must be
reconstructable from the log. The log, not a mutable `messages` list, is the source of truth;
the model's view is *derived* from it by a projection.

WHY THAT INVERSION IS WORTH THE EXTRA STEP

    In stages/s03_loop.py the `messages` list is both the durable record and the thing sent to
    the model. That is simple, and it is also why nothing in that design can support forking a
    conversation, replaying it, auditing what the model actually saw, or dropping old tool
    results from the prompt while keeping them on disk.

    Separating "what happened" (append-only events) from "what the model sees"
    (`derive_messages()`) gets all four, because they become different projections of one log
    rather than different copies of one list. Phase 8 cashes that in: persistence, resume,
    fork, and compaction are each a few dozen lines, and none of them required touching the
    agent loop.

SURFACE OPS — how compaction works without destroying anything

    An event may carry `surface_op = {"op": "replace", "start_seq": N, "end_seq": M}`.
    `derive_messages()` honors it by skipping the events in that range, so the summary event
    stands in for them.

    The events themselves are never deleted. That is the whole trick: the model's context gets
    shorter while the record stays complete, so a compacted session can still be forked,
    audited, or replayed at full fidelity. dsh does exactly this — the summary rides on an
    ordinary `user/message` and the compaction bookkeeping is log-only.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from dshpy.services.llm import Message, ToolCall

name = "sessions"
inject: list[str] = []


@dataclass
class SessionEvent:
    kind: str
    seq: int = 0
    data: dict = field(default_factory=dict)
    surface_op: dict | None = None

    def to_json(self) -> dict:
        return {"kind": self.kind, "seq": self.seq, "data": self.data,
                "surface_op": self.surface_op}

    @classmethod
    def from_json(cls, raw: dict) -> "SessionEvent":
        return cls(kind=raw["kind"], seq=raw.get("seq", 0), data=raw.get("data", {}),
                   surface_op=raw.get("surface_op"))


class SessionsService:
    def __init__(self, ctx, session_id: str | None = None) -> None:
        self._ctx = ctx
        self._events: list[SessionEvent] = []
        self._next_seq = 0
        self.session_id = session_id or uuid.uuid4().hex[:12]

    # --- writing -----------------------------------------------------------------------------

    def append(self, kind: str, *, surface_op: dict | None = None, **data: Any) -> SessionEvent:
        event = SessionEvent(kind=kind, seq=self._next_seq, data=data, surface_op=surface_op)
        self._next_seq += 1
        self._events.append(event)
        self._ctx.emit("session/event", event, self.session_id)
        return event

    def restore(self, events: list[SessionEvent], *, session_id: str | None = None) -> None:
        """Replace the log wholesale — used by resume and fork.

        Deliberately NOT an `append` loop: replaying would re-emit `session/event` for every
        historical event, and a persistence listener would then write the whole history back
        out again. Restoring is not the same as happening.
        """
        self._events = list(events)
        self._next_seq = (max((e.seq for e in self._events), default=-1)) + 1
        if session_id:
            self.session_id = session_id

    # --- reading -----------------------------------------------------------------------------

    def events(self) -> list[SessionEvent]:
        return list(self._events)

    def turn_boundaries(self) -> list[int]:
        """Sequence numbers of every `turn/end`. These are the safe places to fork."""
        return [e.seq for e in self._events if e.kind == "turn/end"]

    def events_upto(self, seq: int) -> list[SessionEvent]:
        return [e for e in self._events if e.seq <= seq]

    def derive_messages(self) -> list[Message]:
        """Project model history from the log, honoring surface ops.

        Everything the model will ever see passes through here, which is what makes the
        "model-visible means logged" invariant checkable rather than aspirational.
        """
        # Collect the ranges shadowed by a replace op, so we can skip them below.
        shadowed: set[int] = set()
        for event in self._events:
            op = event.surface_op
            if op and op.get("op") == "replace":
                shadowed.update(range(op["start_seq"], op["end_seq"] + 1))

        messages: list[Message] = []
        for event in self._events:
            # An event may shadow a range that includes itself; its own surface_op wins.
            if event.seq in shadowed and event.surface_op is None:
                continue

            if event.kind == "system/message":
                messages.append(Message(role="system", content=event.data["content"]))
            elif event.kind == "user/message":
                messages.append(Message(role="user", content=event.data["content"]))
            elif event.kind == "assistant/message":
                messages.append(Message(
                    role="assistant",
                    content=event.data.get("content", ""),
                    tool_calls=[ToolCall(**tc) for tc in event.data.get("tool_calls", [])],
                ))
            elif event.kind == "tool/result":
                messages.append(Message(
                    role="tool",
                    content=event.data["content"],
                    tool_call_id=event.data["call_id"],
                    name=event.data.get("name"),
                ))
        return messages

    def estimate_tokens(self) -> int:
        """A crude proxy, and honestly labelled as one.

        Neither Ollama nor this harness has a tokenizer, and `count_tokens` is not part of
        Ollama's compatibility layer. ~4 characters per token is close enough to decide *when*
        to compact; it would not be close enough to bill anyone. `ctx.tokens` holds the real
        numbers the provider reported, when there are any.
        """
        return sum(len(m.content) + sum(len(str(tc.arguments)) for tc in m.tool_calls)
                   for m in self.derive_messages()) // 4


def apply(ctx, config=None) -> None:
    config = config or {}
    ctx.provide("sessions", SessionsService(ctx, session_id=config.get("session_id")))
