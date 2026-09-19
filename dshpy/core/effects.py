"""Reversible effects.

The fifth Cordis idea: *registrations are reversible effects*. Every registration — a tool, a
listener, a model adapter, a service — hands back a disposer, and a plugin collects its own
disposers into a scope. Unmounting the plugin runs them in reverse order.

WHY THIS IS NOT BOOKKEEPING PEDANTRY

    Without it, "everything is a plugin" is a slogan. A plugin system where unloading leaves
    listeners attached and tools registered is a plugin system you can only load *once*, which
    means no hot reload, no per-session capability sets, and no honest test isolation (every
    test leaks into the next).

    Reverse order matters for the same reason it does in a destructor: later registrations may
    depend on earlier ones, so they must come down first.
"""

from __future__ import annotations

from typing import Callable

Disposer = Callable[[], None]


class EffectScope:
    """A bag of disposers belonging to one plugin instance."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._disposers: list[Disposer] = []
        self.disposed = False

    def add(self, dispose: Disposer) -> Disposer:
        """Record a disposer and hand it straight back, so callers can also dispose early."""
        self._disposers.append(dispose)
        return dispose

    def dispose(self) -> None:
        """Unwind every registration, newest first. Idempotent."""
        if self.disposed:
            return
        self.disposed = True
        while self._disposers:
            dispose = self._disposers.pop()
            try:
                dispose()
            except Exception as exc:  # noqa: BLE001 - one bad disposer must not strand the rest
                print(f"[effects] disposer in {self.name!r} raised: {exc}")

    def __len__(self) -> int:
        return len(self._disposers)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = "disposed" if self.disposed else f"{len(self._disposers)} effects"
        return f"<EffectScope {self.name} ({state})>"
