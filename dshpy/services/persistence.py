"""`ctx.persistence` — storing and reloading a session log.

Another three-role seam: this contract, `plugins/persistence_jsonl.py` as the provider, and
resume/fork as the consumers. A different backend (sqlite, S3, an HTTP service) is a sibling
plugin and changes nothing above it.

WHAT MAKES THIS CHEAP

    The log is append-only and the model's view is derived. So "save" is "append a line", and
    "resume" is "read the lines back". There is no state to serialize, no ordering to
    reconstruct, no partially-applied mutation to repair. Every hard problem that persistence
    usually has was avoided three phases earlier by choosing an append-only log.

    That is the real argument for the design, and it only becomes visible now.
"""

from __future__ import annotations

from typing import Callable

from dshpy.services.sessions import SessionEvent

name = "persistence"
inject: list[str] = []


class PersistenceProvider:
    def append(self, session_id: str, event: SessionEvent) -> None:
        raise NotImplementedError

    def load(self, session_id: str) -> list[SessionEvent]:
        raise NotImplementedError

    def write_all(self, session_id: str, events: list[SessionEvent]) -> None:
        """Replace a session's whole log. Used by fork, which creates a new one."""
        raise NotImplementedError

    def list_sessions(self) -> list[str]:
        raise NotImplementedError


class PersistenceService:
    def __init__(self, ctx) -> None:
        self._ctx = ctx
        self._provider: PersistenceProvider | None = None

    def register_provider(self, provider: PersistenceProvider) -> Callable[[], None]:
        if self._provider is not None:
            raise RuntimeError("a persistence provider is already registered")
        self._provider = provider

        def dispose() -> None:
            if self._provider is provider:
                self._provider = None

        return dispose

    @property
    def provider(self) -> PersistenceProvider:
        if self._provider is None:
            raise RuntimeError("no persistence provider is mounted")
        return self._provider

    def load(self, session_id: str) -> list[SessionEvent]:
        return self.provider.load(session_id)

    def list_sessions(self) -> list[str]:
        return self.provider.list_sessions()

    # --- the two consumers ---------------------------------------------------------------

    def resume(self, session_id: str) -> int:
        """Load a stored session into `ctx.sessions`. Returns the event count."""
        events = self.provider.load(session_id)
        self._ctx.sessions.restore(events, session_id=session_id)
        self._ctx.emit("session/resumed", session_id, len(events))
        return len(events)

    def fork(self, session_id: str, at_seq: int | None = None,
             new_id: str | None = None) -> str:
        """Copy a session up to a turn boundary into a new one, and return its id.

        WHY ONLY AT A TURN BOUNDARY

            Forking mid-turn produces a log whose last assistant message asked for tools that
            never got results. That history is not merely untidy -- both wire protocols reject
            it, because a tool_use with no matching tool_result is malformed. So the default
            cut point is the last `turn/end`, and an explicit `at_seq` is validated against the
            recorded boundaries rather than trusted.
        """
        import uuid

        events = self.provider.load(session_id)
        boundaries = [e.seq for e in events if e.kind == "turn/end"]

        if at_seq is None:
            if not boundaries:
                raise ValueError(f"session {session_id} has no completed turn to fork at")
            at_seq = boundaries[-1]
        elif at_seq not in boundaries:
            raise ValueError(
                f"seq {at_seq} is not a turn boundary in {session_id}; "
                f"completed turns end at {boundaries}"
            )

        child_id = new_id or uuid.uuid4().hex[:12]
        kept = [e for e in events if e.seq <= at_seq]
        self.provider.write_all(child_id, kept)
        self._ctx.emit("session/forked", session_id, child_id, at_seq)
        return child_id


def apply(ctx, config=None) -> None:
    ctx.provide("persistence", PersistenceService(ctx))
