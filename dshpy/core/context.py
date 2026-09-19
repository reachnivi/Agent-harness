"""The Context: a repository of services, and the handle every plugin is given.

This is the second and third Cordis ideas, which are the ones that actually buy you anything:

    2. A context is a repository of services. A service claims a stable key — `ctx.tools`,
       `ctx.llm`, `ctx.sessions` — and other plugins find it BY KEY, never by importing the
       implementation.

    3. Declare service dependency via `inject`. A plugin naming required services waits until
       they exist, and deactivates if one disappears. Load order is expressed through service
       requirements rather than manual boot sequencing.

WHY LOOKUP-BY-KEY IS THE WHOLE TRICK

    `from dshpy.plugins.llm_openai import OpenAIAdapter` welds your loop to one provider. The
    import is a hard edge: to swap the provider you edit the consumer.

    `ctx.llm` does not. Whoever claimed the key is what you get, and who claimed it is decided
    by configuration at boot. The loop can be written, tested, and read without any provider
    existing at all — which is exactly why the two adapters in this project can be swapped by
    changing one row in a profile.

    The cost is real and worth naming: you lose static "go to definition" on `ctx.llm`. That is
    the price of a seam, and it is why `inject` exists — it is the declaration that recovers,
    at boot, the guarantee the import used to give you at import time.

WHY DEFERRED ACTIVATION BEATS ORDERING YOUR BOOT BY HAND

    The naive version is a boot script: start sessions, then tools, then llm, then the loop.
    That works until a plugin is optional, or two are mutually dependent, or a user reorders
    their config. Then the ordering is wrong in a way that shows up as an AttributeError deep
    inside someone else's plugin.

    `inject` inverts it. A plugin states what it needs; the loader works out when that is true.
    Mount order stops being load order, so a profile is a *set* of plugins rather than a
    sequence, and a missing service is a clear "never activated" rather than a crash.
"""

from __future__ import annotations

from typing import Any, Callable

from dshpy.core.effects import EffectScope
from dshpy.core.events import EventBus


class ServiceMissing(AttributeError):
    """Asked for a service key that nothing has provided."""


class Context:
    """What a plugin's `apply(ctx)` receives.

    A Context is a *view* onto shared state (services, the event bus) bound to one plugin's
    effect scope. Every registration made through it is recorded against that plugin, so
    unmounting the plugin unwinds exactly its own work and nothing else.
    """

    def __init__(self, runtime: "Runtime", scope: EffectScope) -> None:
        # Leading underscores keep the namespace clear for service keys: `ctx.tools`, `ctx.llm`.
        self._runtime = runtime
        self._scope = scope

    # --- services -------------------------------------------------------------------------

    def __getattr__(self, key: str) -> Any:
        """`ctx.tools` resolves to whatever plugin claimed the key 'tools'."""
        if key.startswith("_"):
            raise AttributeError(key)
        try:
            return self._runtime.services[key]
        except KeyError:
            raise ServiceMissing(
                f"no service named {key!r} (available: {sorted(self._runtime.services)}). "
                f"Did the providing plugin fail to activate, or is {key!r} missing from this "
                f"plugin's `inject` list?"
            ) from None

    def provide(self, key: str, value: Any) -> Callable[[], None]:
        """Claim a service key. Reversible: unmounting the provider frees the key.

        Freeing the key is what makes deactivation-on-service-loss possible, which is the half
        of idea 3 that systems usually skip.
        """
        if key in self._runtime.services:
            raise RuntimeError(
                f"service {key!r} already provided by {self._runtime.providers.get(key)!r}"
            )
        self._runtime.services[key] = value
        self._runtime.providers[key] = self._scope.name

        def dispose() -> None:
            if self._runtime.services.get(key) is value:
                del self._runtime.services[key]
                self._runtime.providers.pop(key, None)
                self._runtime._on_service_lost(key)

        self._scope.add(dispose)
        self._runtime._on_service_gained(key)
        return dispose

    def has(self, key: str) -> bool:
        return key in self._runtime.services

    # --- events ---------------------------------------------------------------------------

    def on(self, event: str, fn: Callable, *, prepend: bool = False) -> Callable[[], None]:
        """Listen for an event. Reversible, like every other registration."""
        dispose = self._runtime.events.on(event, fn, owner=self._scope.name, prepend=prepend)
        return self._scope.add(dispose)

    def emit(self, event: str, *args: Any) -> None:
        self._runtime.events.emit(event, *args)

    def bail(self, event: str, *args: Any) -> Any:
        return self._runtime.events.bail(event, *args)

    def waterfall(self, event: str, *args: Any, final: Callable | None = None) -> Any:
        return self._runtime.events.waterfall(event, *args, final=final)

    # --- effects & composition ------------------------------------------------------------

    def effect(self, dispose: Callable[[], None]) -> Callable[[], None]:
        """Record an arbitrary teardown against this plugin."""
        return self._scope.add(dispose)

    def plugin(self, module: Any, config: dict | None = None) -> None:
        """Mount another plugin. A plugin may compose others; that is how bundles work."""
        self._runtime.mount(module, config)

    @property
    def name(self) -> str:
        return self._scope.name


class Runtime:
    """Owns the service table, the event bus, and every mounted plugin's lifecycle.

    This is the "orchestration shell" from the dsh architecture doc: lifecycle management,
    plugin resolution, context propagation. Note what is *not* here — no tools, no model, no
    loop. There is no privileged core to patch; capability arrives entirely by mounting.
    """

    def __init__(self) -> None:
        self.services: dict[str, Any] = {}
        self.providers: dict[str, str] = {}
        self.events = EventBus()
        self._mounted: list[_Mounted] = []
        self._settling = False

    # --- mounting -------------------------------------------------------------------------

    def mount(self, module: Any, config: dict | None = None) -> "_Mounted":
        """Register a plugin. It activates now if its `inject` is satisfied, else later."""
        name = getattr(module, "name", getattr(module, "__name__", repr(module)))
        inject = list(getattr(module, "inject", ()))
        apply_fn = getattr(module, "apply", None)
        if apply_fn is None:
            raise TypeError(f"plugin {name!r} has no apply(ctx, config) function")

        entry = _Mounted(name=name, module=module, config=config or {},
                         inject=inject, apply_fn=apply_fn)
        self._mounted.append(entry)
        self._settle()
        return entry

    def mount_all(self, rows: list[dict]) -> None:
        """Mount a profile: a list of {'plugin': module, 'config': {...}} rows.

        Order here is *mount* order, not load order. The loader sorts activation out from
        `inject`, which is the point — a profile is a set, not a sequence.
        """
        for row in rows:
            if row.get("disabled"):
                continue
            self.mount(row["plugin"], row.get("config"))

    # --- activation -----------------------------------------------------------------------

    def _satisfied(self, entry: "_Mounted") -> bool:
        return all(key in self.services for key in entry.inject)

    def _settle(self) -> None:
        """Activate everything whose dependencies are now met; repeat until stable.

        The loop is required, not defensive: activating one plugin can provide a service that
        satisfies another, which can provide a service that satisfies a third. Iterating to a
        fixed point is what removes hand-written boot ordering.
        """
        if self._settling:
            return  # re-entrant call from inside an apply(); the outer loop will catch it
        self._settling = True
        try:
            changed = True
            while changed:
                changed = False
                for entry in list(self._mounted):
                    if entry.active or not self._satisfied(entry):
                        continue
                    self._activate(entry)
                    changed = True
        finally:
            self._settling = False

    def _activate(self, entry: "_Mounted") -> None:
        entry.scope = EffectScope(entry.name)
        ctx = Context(self, entry.scope)
        entry.active = True
        try:
            entry.apply_fn(ctx, entry.config)
        except Exception:
            entry.active = False
            entry.scope.dispose()
            entry.scope = None
            raise

    def _deactivate(self, entry: "_Mounted") -> None:
        if not entry.active:
            return
        entry.active = False
        if entry.scope is not None:
            entry.scope.dispose()
            entry.scope = None

    def _on_service_gained(self, key: str) -> None:
        self._settle()

    def _on_service_lost(self, key: str) -> None:
        """Deactivate anything that injected the service that just vanished.

        This is the half of idea 3 that is easy to skip and expensive to skip: a plugin holding
        a reference to a service that no longer exists is a plugin that will fail later, at a
        confusing moment, instead of now.
        """
        for entry in list(self._mounted):
            if entry.active and key in entry.inject:
                self._deactivate(entry)

    def unmount(self, entry: "_Mounted") -> None:
        self._deactivate(entry)
        if entry in self._mounted:
            self._mounted.remove(entry)

    def dispose(self) -> None:
        for entry in reversed(list(self._mounted)):
            self._deactivate(entry)
        self._mounted.clear()

    # --- introspection --------------------------------------------------------------------

    def dump(self) -> str:
        """What `dsh --dump-config` does: show the mounted tree and why each row is where."""
        lines = ["mounted plugins:"]
        for entry in self._mounted:
            if entry.active:
                state = "active"
            else:
                missing = [k for k in entry.inject if k not in self.services]
                state = f"WAITING on {missing}" if missing else "inactive"
            inject = f" inject={entry.inject}" if entry.inject else ""
            lines.append(f"  [{state:>16}] {entry.name}{inject}")
        lines.append("services:")
        for key in sorted(self.services):
            lines.append(f"  ctx.{key:<12} <- {self.providers.get(key, '?')}")
        return "\n".join(lines)


class _Mounted:
    """One plugin's mount record."""

    __slots__ = ("name", "module", "config", "inject", "apply_fn", "active", "scope")

    def __init__(self, name: str, module: Any, config: dict,
                 inject: list[str], apply_fn: Callable) -> None:
        self.name = name
        self.module = module
        self.config = config
        self.inject = inject
        self.apply_fn = apply_fn
        self.active = False
        self.scope: EffectScope | None = None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Mounted {self.name} {'active' if self.active else 'inactive'}>"
