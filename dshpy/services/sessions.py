"""`ctx.sessions` — the append-only event log.

dsh's rule: **model-visible means logged.** Anything that reaches a model request must be
reconstructable from the log. The log, not a mutable `messages` list, is the source of truth;
the model's view is *derived* from it by a projection.

WHY THAT INVERSION IS WORTH THE EXTRA STEP

    In stages/s03_loop.py the `messages` list is both the durable record and the thing sent to
    the model. That is simple and it is also why nothing in that design can support forking a
    conversation, replaying it, auditing what the model actually saw, or dropping old tool
    results from the prompt while keeping them on disk.

    Separating "what happened" (append-only events) from "what the model sees"
    (`derive_messages()`) gets all of those for free, because they become different projections
    of the same log rather than different copies of the same list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dshpy.services.llm import Message, ToolCall

name = "sessions"
inject: list[str] = []


@dataclass
class SessionEvent:
    kind: str  # user/message, assistant/message, tool/call, tool/result, turn/start, turn/end
    data: dict = field(default_factory=dict)


class SessionsService:
    def __init__(self, ctx) -> None:
        self._ctx = ctx
        self._events: list[SessionEvent] = []

    def append(self, kind: str, **data: Any) -> SessionEvent:
        event = SessionEvent(kind=kind, data=data)
        self._events.append(event)
        self._ctx.emit("session/event", event)
        return event

    def events(self) -> list[SessionEvent]:
        return list(self._events)

    def derive_messages(self) -> list[Message]:
        """Project model history from the log.

        The adapters translate this neutral list into their own wire format. Everything the
        model will see passes through here, which is what makes the invariant checkable.
        """
        messages: list[Message] = []
        for event in self._events:
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


def apply(ctx, config=None) -> None:
    ctx.provide("sessions", SessionsService(ctx))
