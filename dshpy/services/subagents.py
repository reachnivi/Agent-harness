"""`ctx.subagents` — delegating work to a child agent.

WHY THIS IS A NAMED REGISTRY AND NOT A SINGLE-PROVIDER SEAM

    `ctx.fs` and `ctx.shell` allow one provider each: "read this file" has no selector, so a
    second backend would be ambiguous rather than useful.

    Subagents are the opposite, and dsh says so explicitly: **multiple providers coexist,
    registered by name** — an in-process child, a forked session, another product entirely.
    The request names which one it wants, exactly like `ctx.llm`'s provider routes. So this
    registry is shaped after the LLM adapter registry, not after the filesystem.

    That is a design decision you can get wrong in either direction, and the deciding question
    is simply: does the caller need to choose? If yes, a named registry; if no, one provider.

CAPABILITIES ARE CHECKED BEFORE THE RUN, AND FAIL LOUD

    dsh: a provider advertises start-time features on a static descriptor, and "a request that
    needs one the provider lacks is rejected loud, never accepted-then-ignored".

    The failure this prevents is specific and nasty. If a caller asks for a depth limit and the
    provider silently ignores it, nothing appears wrong — until a recursive delegation runs
    until the budget is gone. Silent degradation of a *safety* option is the worst kind,
    because the thing you asked for is exactly the thing that would have told you.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

name = "subagents"
inject: list[str] = []


class SubagentError(RuntimeError):
    pass


class UnsupportedCapability(SubagentError):
    """The chosen provider cannot honor something the request asked for."""


@dataclass(frozen=True)
class Capabilities:
    """What a provider can do, checked BEFORE a run exists."""

    depth_limit: bool = False      # honors max_depth
    tool_filter: bool = False      # honors allowed_tools
    cancellation: bool = False     # honors a cancel token


@dataclass
class SubagentRequest:
    task: str
    allowed_tools: list[str] | None = None
    max_depth: int = 3
    depth: int = 0
    max_steps: int = 8
    label: str = ""
    parents: tuple[str, ...] = ()   # ancestry, for the cycle guard


@dataclass
class SubagentResult:
    label: str
    output: str
    steps: int = 0
    is_error: bool = False
    events: list = field(default_factory=list)


class SubagentProvider:
    name: str = "?"
    capabilities: Capabilities = Capabilities()

    def start(self, ctx, request: SubagentRequest) -> SubagentResult:
        raise NotImplementedError


class SubagentsService:
    def __init__(self, ctx) -> None:
        self._ctx = ctx
        self._providers: dict[str, SubagentProvider] = {}
        self._default: str | None = None

    def register_provider(self, provider: SubagentProvider) -> Callable[[], None]:
        if provider.name in self._providers:
            raise RuntimeError(f"subagent provider {provider.name!r} is already registered")
        self._providers[provider.name] = provider
        if self._default is None:
            self._default = provider.name

        def dispose() -> None:
            if self._providers.get(provider.name) is provider:
                del self._providers[provider.name]
            if self._default == provider.name:
                self._default = next(iter(self._providers), None)

        return dispose

    def providers(self) -> list[str]:
        return list(self._providers)

    def start(self, request: SubagentRequest,
              provider: str | None = None) -> SubagentResult:
        route = provider or self._default
        if route is None:
            raise SubagentError("no subagent provider is mounted")
        impl = self._providers.get(route)
        if impl is None:
            raise SubagentError(f"no subagent provider {route!r} (have: {self.providers()})")

        # --- the guards, all before any work starts -----------------------------------------

        caps = impl.capabilities
        if request.allowed_tools is not None and not caps.tool_filter:
            raise UnsupportedCapability(
                f"provider {route!r} cannot restrict a child's tools, and the request asked "
                f"for it. Refusing rather than running the child unrestricted."
            )
        if request.max_depth and not caps.depth_limit:
            raise UnsupportedCapability(
                f"provider {route!r} does not honor a depth limit, and the request set one."
            )

        if request.depth >= request.max_depth:
            raise SubagentError(
                f"delegation depth limit reached ({request.max_depth}). "
                f"Do this work directly instead of delegating further."
            )

        # A cycle guard is not the same as a depth limit: three agents delegating in a ring
        # never exceed depth 1 individually while looping forever.
        if request.label and request.label in request.parents:
            raise SubagentError(
                f"delegation cycle: {request.label!r} is already an ancestor of this task "
                f"({' -> '.join(request.parents)})"
            )

        self._ctx.emit("subagent/start", request, route)
        result = impl.start(self._ctx, request)
        self._ctx.emit("subagent/end", request, result)
        return result


def apply(ctx, config=None) -> None:
    ctx.provide("subagents", SubagentsService(ctx))
