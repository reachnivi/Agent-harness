"""`ctx.compaction` — replacing a span of history with a summary.

dsh treats compaction as **one optional capability, not part of the agent-loop spine**, and
this port keeps that: nothing in `agent_loop.py` mentions compaction. It is a seam
(`services/compaction.py`), a backend (`plugins/compaction_basic.py`), and a consumer (either
a human command or the auto-trigger in the backend).

THE THREE LOG-ONLY EVENTS, AND WHY THE LOCK RELEASES LAST

    compaction/start    acquires a lock, recorded IN the log
    compaction/summary  the summary, the shadowed range, the model call that produced it
    compaction/end      releases the lock

    All three are log-only: they never reach the model. The summary that *does* reach the model
    rides on an ordinary `user/message` carrying
    `surface_op={"op":"replace","start_seq":…,"end_seq":…}`, which is the single surface
    mutation the whole operation performs.

    The lock brackets the *entire* operation — start, then summarize, then the summary record,
    then the replacement message, and only then end. Releasing last is deliberate, and dsh
    spells out why: a crash mid-operation then leaves a `compaction/start` with no matching
    `compaction/end`, which is detectable. Release first and a crash leaves an `end` that
    falsely claims the compaction finished, and you cannot tell a completed compaction from an
    abandoned one by reading the log.

WHY NOTHING IS DELETED

    Compaction changes the *projection*, not the record. `derive_messages()` skips the shadowed
    range; the events stay on disk. So a compacted session is still forkable at any earlier
    turn, still auditable, and still replayable at full fidelity.

    The alternative — trimming the list in place — is simpler and throws away the ability to
    answer "what did the model actually see three turns ago", which is the question you most
    want answered when an agent does something inexplicable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

name = "compaction"
inject: list[str] = []


@dataclass
class CompactionResult:
    summary: str
    start_seq: int
    end_seq: int
    shadowed_events: int
    tokens_before: int
    tokens_after: int


class CompactionBackend:
    """What a compaction strategy implements: turn a span of history into a summary string."""

    def summarize(self, ctx, events: list, *, reason: str) -> str:
        raise NotImplementedError


class CompactionService:
    def __init__(self, ctx) -> None:
        self._ctx = ctx
        self._backend: CompactionBackend | None = None

    def register_backend(self, backend: CompactionBackend) -> Callable[[], None]:
        if self._backend is not None:
            raise RuntimeError("a compaction backend is already registered")
        self._backend = backend

        def dispose() -> None:
            if self._backend is backend:
                self._backend = None

        return dispose

    # --- the lock ----------------------------------------------------------------------------

    def is_locked(self) -> bool:
        """True if a `compaction/start` has no matching `compaction/end`.

        Also the orphaned-lock detector: a lock still held with no operation in flight means a
        previous attempt died partway through.
        """
        depth = 0
        for event in self._ctx.sessions.events():
            if event.kind == "compaction/start":
                depth += 1
            elif event.kind == "compaction/end":
                depth -= 1
        return depth > 0

    def clear_orphaned_lock(self) -> bool:
        """Close a lock left behind by a crashed attempt. Returns True if one was found."""
        if not self.is_locked():
            return False
        self._ctx.sessions.append("compaction/end", error="orphaned lock cleared")
        return True

    # --- the operation -------------------------------------------------------------------

    def compact(self, *, keep_last_turns: int = 1,
                reason: str = "manual") -> CompactionResult | None:
        """Summarize everything before the last `keep_last_turns` turns.

        Returns None when there is nothing worth compacting, which is a normal outcome rather
        than an error — an agent that auto-compacts will call this speculatively.
        """
        if self._backend is None:
            raise RuntimeError("no compaction backend is mounted")
        if self.is_locked():
            raise RuntimeError(
                "compaction is already in progress (or a previous attempt left an orphaned "
                "lock — call clear_orphaned_lock())"
            )

        sessions = self._ctx.sessions
        boundaries = sessions.turn_boundaries()
        if len(boundaries) <= keep_last_turns:
            return None  # not enough completed turns to be worth summarizing

        cut = boundaries[-(keep_last_turns + 1)]
        events = sessions.events()

        # Never shadow the system prompt: it is instruction, not history, and summarizing it
        # away silently changes the agent's behavior for the rest of the session.
        shadowable = [e for e in events
                      if e.seq <= cut and e.kind != "system/message"
                      and not e.kind.startswith("compaction/")]
        if not shadowable:
            return None

        start_seq = min(e.seq for e in shadowable)
        end_seq = max(e.seq for e in shadowable)
        tokens_before = sessions.estimate_tokens()

        sessions.append("compaction/start", reason=reason)
        try:
            summary = self._backend.summarize(self._ctx, shadowable, reason=reason)

            sessions.append("compaction/summary", summary=summary,
                            start_seq=start_seq, end_seq=end_seq,
                            shadowed=len(shadowable))

            # The ONE surface mutation: a user message standing in for the shadowed range.
            sessions.append(
                "user/message",
                content=f"[Earlier conversation, summarized]\n{summary}",
                surface_op={"op": "replace", "start_seq": start_seq, "end_seq": end_seq},
            )
        except Exception as exc:
            sessions.append("compaction/end", error=str(exc))
            raise
        else:
            sessions.append("compaction/end")  # released LAST, on purpose

        result = CompactionResult(
            summary=summary, start_seq=start_seq, end_seq=end_seq,
            shadowed_events=len(shadowable), tokens_before=tokens_before,
            tokens_after=sessions.estimate_tokens(),
        )
        self._ctx.emit("compaction/done", result)
        return result


def apply(ctx, config=None) -> None:
    ctx.provide("compaction", CompactionService(ctx))
