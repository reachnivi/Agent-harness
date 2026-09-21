"""Cancellation, as a token you pass rather than a thread you abandon.

WHY THIS FILE EXISTS

    Phase 5's `plugins/timeout.py` used a daemon thread: the *call* returned on time, but the
    work kept running. I flagged it as an honest simplification. This is the fix.

    The difference matters more than it sounds. Abandoned work still holds the file handle,
    still finishes the subprocess, still writes the file you cancelled — it just does so with
    nobody listening. Under an agent loop that retries, you get the operation twice.

THE CONTRACT

    Cancellation is COOPERATIVE. A token cannot stop code that never looks at it; Python has no
    safe way to interrupt an arbitrary thread. So the deal is:

      - whoever starts long work accepts a token
      - long work polls `token.check()` at points where stopping is safe
      - `check()` raises `Cancelled`, which the tools pipeline turns into an error result

    Every loop that can run long — the HTTP read loop, a subprocess poll, a directory walk —
    must poll. A token threaded into a function that never checks it is a lie, and the only
    defence is remembering to poll.

    `deadline()` is the common case: a token that cancels itself after N seconds, with no
    thread involved at all. It just compares clocks when asked.
"""

from __future__ import annotations

import threading
import time


class Cancelled(Exception):
    """Raised by `CancelToken.check()` once the token is set."""

    def __init__(self, reason: str = "cancelled") -> None:
        super().__init__(reason)
        self.reason = reason


class CancelToken:
    """A one-way flag, optionally with a deadline. Thread-safe, cheap to poll."""

    __slots__ = ("_event", "_reason", "_expires_at")

    def __init__(self, *, timeout: float | None = None, reason: str = "cancelled") -> None:
        self._event = threading.Event()
        self._reason = reason
        self._expires_at = (time.monotonic() + timeout) if timeout is not None else None

    def cancel(self, reason: str | None = None) -> None:
        if reason:
            self._reason = reason
        self._event.set()

    def is_set(self) -> bool:
        """True once cancelled — including by a deadline that has now passed.

        The deadline is evaluated lazily, on read. No timer thread, nothing to clean up, and a
        token nobody polls costs nothing.
        """
        if self._event.is_set():
            return True
        if self._expires_at is not None and time.monotonic() >= self._expires_at:
            self._reason = "deadline exceeded"
            self._event.set()
            return True
        return False

    def check(self) -> None:
        """Raise `Cancelled` if cancelled. Call this from inside any loop that can run long."""
        if self.is_set():
            raise Cancelled(self._reason)

    @property
    def reason(self) -> str:
        return self._reason

    def remaining(self) -> float | None:
        """Seconds left, for handing a deadline to something that takes one (e.g. socket reads)."""
        if self._expires_at is None:
            return None
        return max(0.0, self._expires_at - time.monotonic())

    def child(self, *, timeout: float | None = None) -> "CancelToken":
        """A token cancelled when this one is, optionally with a tighter deadline of its own.

        Used when a subagent or a nested call needs its own budget without escaping the
        parent's. Linked by *polling* rather than by callback: the child asks the parent on
        every `is_set()`, so there is no registration to leak and nothing to unsubscribe.
        Cancelling the child does not affect the parent.
        """
        parent = self

        class _Linked(CancelToken):
            __slots__ = ()

            def is_set(self) -> bool:
                if parent.is_set():
                    self.cancel(parent.reason)
                    return True
                return CancelToken.is_set(self)

        return _Linked(timeout=timeout)


NEVER = CancelToken()
"""A token that is never set. Use as a default so callers can always pass one."""
