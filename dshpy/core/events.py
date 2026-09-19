"""Event dispatch modes.

Ported from Cordis, the plugin framework under DeepSeek Harness. Their primer lists five
dispatch modes; we implement the three that carry the architecture:

    | Mode        | Order                        | Returns a value? |
    |-------------|------------------------------|------------------|
    | emit        | registration order           | no               |
    | waterfall   | registration order, nested   | yes              |
    | bail        | until one listener bails     | yes              |

(`parallel` and `serial` are the async variants; adding them is mechanical and they would teach
nothing new here.)

WHY WATERFALL IS THE INTERESTING ONE

    `waterfall` is around-middleware, not a notification. A listener receives the event's
    arguments plus a `next` callable:

        def listener(exec, next):
            if not allowed(exec):
                return Deny("policy")       # short-circuit: downstream never runs
            result = next()                 # delegate to the rest of the chain
            return annotate(result)         # ...and you may still transform the result

    Calling `next()` delegates. Returning without calling it short-circuits. Because the call to
    `next()` is *inside* the listener, a listener can also run code after the inner chain
    finishes — which is how a timeout, a retry, or a metrics wrapper is written without the
    thing being wrapped knowing about it.

    That single property is what lets DeepSeek Harness put permission policy, sandboxing,
    timeouts and result rewriting into plugins while the agent loop stays ignorant of all four.
"""

from __future__ import annotations

from typing import Any, Callable


class Listener:
    """One registered listener, with the plugin that owns it (for teardown and debugging)."""

    __slots__ = ("event", "fn", "owner", "prepend")

    def __init__(self, event: str, fn: Callable, owner: str, prepend: bool = False) -> None:
        self.event = event
        self.fn = fn
        self.owner = owner
        self.prepend = prepend

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Listener {self.event} from {self.owner}>"


class EventBus:
    """Holds listeners by event name and dispatches them in one of three modes."""

    def __init__(self) -> None:
        self._listeners: dict[str, list[Listener]] = {}

    # --- registration ---------------------------------------------------------------------

    def on(self, event: str, fn: Callable, *, owner: str = "?", prepend: bool = False) -> Callable:
        """Register a listener. Returns a disposer — see core/effects.py for why that matters."""
        listener = Listener(event, fn, owner, prepend)
        bucket = self._listeners.setdefault(event, [])
        if prepend:
            bucket.insert(0, listener)
        else:
            bucket.append(listener)

        def dispose() -> None:
            if listener in bucket:
                bucket.remove(listener)

        return dispose

    def listeners(self, event: str) -> list[Listener]:
        return list(self._listeners.get(event, ()))

    # --- dispatch -------------------------------------------------------------------------

    def emit(self, event: str, *args: Any) -> None:
        """Fire-and-forget. Listeners observe; nothing they return is used.

        A raising observer must not break the thing it is observing, so exceptions are
        swallowed and attributed. Use `emit` only where that is the right trade —
        telemetry, logging, UI updates.
        """
        for listener in self.listeners(event):
            try:
                listener.fn(*args)
            except Exception as exc:  # noqa: BLE001 - observers must not break producers
                print(f"[events] listener {listener.owner!r} on {event!r} raised: {exc}")

    def bail(self, event: str, *args: Any) -> Any:
        """Run listeners in order until one returns a non-None value; that value wins.

        The "first opinion wins" mode. Used for single-answer lookups where several plugins
        could supply an answer and you want the first that can.
        """
        for listener in self.listeners(event):
            result = listener.fn(*args)
            if result is not None:
                return result
        return None

    def waterfall(self, event: str, *args: Any, final: Callable[..., Any] | None = None) -> Any:
        """Around-middleware. Each listener gets `(*args, next)`.

        `final` is the innermost behavior — the thing being wrapped. If every listener delegates
        via `next()`, `final(*args)` runs and its value propagates back out through each
        listener, which may transform it. If a listener returns without calling `next()`,
        neither `final` nor any later listener runs.
        """
        chain = self.listeners(event)

        def step(index: int) -> Any:
            if index >= len(chain):
                return final(*args) if final is not None else None
            listener = chain[index]
            return listener.fn(*args, lambda: step(index + 1))

        return step(0)
