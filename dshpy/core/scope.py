"""Scoped registration — the kernel upgrade `docs/PLUGINS.md` listed as deliberately omitted.

THE PROBLEM IT SOLVES

    Until now every registration is global: a tool registered by any plugin is visible to every
    agent, because there was only ever one agent. The moment a subagent exists that stops being
    true. A research subagent should not get `write_file`; a parent should not see the child's
    private scratchpad tool. Without scoping, "give this agent a restricted tool set" means
    unmounting plugins globally and remounting them afterwards — which is racy, and impossible
    if the two agents overlap in time.

HOW dsh DOES IT, AND WHAT THAT BUYS

    A `ScopeKey` is **an opaque object identity**. dsh uses the live Agent object as its own
    key and states that the primitive never inspects it. Copying that exactly matters:

      - identity comparison means no naming scheme, no registry of scope ids, no collisions
      - "never inspects it" means the scope primitive has no opinion about agents at all, so
        the same machinery scopes anything else later (a session, a request, a tab)
      - a scope dies when its key is dropped, so lifetime is ownership rather than bookkeeping

    The visibility rule is one line and worth stating plainly:

        a registration with scope S is visible to S, and to nothing else;
        a registration with no scope is visible to everyone.

    Global registrations stay visible inside a scope because that is what makes the feature
    additive — mounting a subagent must not hide the tools everyone already had.
"""

from __future__ import annotations

from typing import Any, Callable, Generic, Iterator, TypeVar

T = TypeVar("T")

ScopeKey = Any  # an opaque object identity; never inspected


class ScopedRegistry(Generic[T]):
    """A registry whose entries may be global or bound to one opaque scope key.

    Entries are held in insertion order, because for tools and listeners alike the order of
    registration is part of the observable behavior.
    """

    def __init__(self) -> None:
        self._entries: list[tuple[ScopeKey | None, T]] = []

    def add(self, item: T, *, scope: ScopeKey | None = None) -> Callable[[], None]:
        entry = (scope, item)
        self._entries.append(entry)

        def dispose() -> None:
            try:
                self._entries.remove(entry)
            except ValueError:
                pass  # already disposed; disposers must be idempotent

        return dispose

    def visible(self, scope: ScopeKey | None = None) -> list[T]:
        """Everything global, plus everything bound to exactly this scope.

        Identity comparison (`is`), not equality: two different agents that happen to compare
        equal must still not see each other's registrations.
        """
        return [item for entry_scope, item in self._entries
                if entry_scope is None or entry_scope is scope]

    def in_scope(self, scope: ScopeKey | None = None) -> list[T]:
        """Only entries bound to EXACTLY this scope — global entries are not included.

        Distinct from `visible()` on purpose. `visible()` answers "what can this caller use",
        which is additive; `in_scope()` answers "what did this scope itself register", which is
        what a duplicate check needs. Conflating them makes it impossible to shadow a global
        registration with a scoped one.
        """
        return [item for entry_scope, item in self._entries if entry_scope is scope]

    def drop_scope(self, scope: ScopeKey) -> int:
        """Remove every entry bound to a scope. Returns how many went."""
        before = len(self._entries)
        self._entries = [e for e in self._entries if e[0] is not scope]
        return before - len(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[T]:
        return iter(item for _, item in self._entries)
