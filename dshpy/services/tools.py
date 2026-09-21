"""`ctx.tools` — the tool registry and its guarded execution pipeline.

This is the showcase for the whole pattern. DeepSeek Harness runs every tool call through a
fixed sequence of extension points, and each stage is an event that plugins listen on:

    tools/pre-execute   (waterfall)  policy: allow / deny / ask
      guards            (monotonic)  a final deny nothing downstream can undo
        tools/execute   (waterfall)  around-dispatch: timeout, retry, metrics
          <the tool body>
        tools/post-execute (waterfall) transform, block, or annotate the result
    tools/result        (emit)       observe the immutable outcome

WHY THERE ARE BOTH A GATE AND A GUARD

    They look redundant and are not, and getting this wrong is how permission systems grow
    holes.

    `tools/pre-execute` is a *reorderable policy layer*. Listeners compose: an outer one may
    allow what an inner one questioned, because that is what "policy" means — the last word
    depends on configuration.

    `ctx.tools.guard()` is a *monotonic final deny*. Once any guard denies, nothing later can
    allow. Invariants belong here — path confinement, "never touch this directory" — precisely
    because they must NOT be overridable by a plugin mounted later or by a user's config order.

    A permission gate that only has the reorderable layer is one badly-ordered profile away
    from being bypassed.

WHY THE LOOP DOESN'T KNOW ABOUT ANY OF THIS

    The agent loop calls `ctx.tools.execute(call)`. That's it. Every stage above is optional
    and arrives by mounting a plugin. Compare stages/s03_loop.py, where adding a permission
    prompt means editing the loop itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from dshpy.cancel import Cancelled, CancelToken
from dshpy.core.scope import ScopedRegistry

name = "tools"
inject: list[str] = []


# --- the vocabulary -------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolDefinition:
    """A tool as both the model and the registry see it."""

    name: str
    description: str
    parameters: dict[str, dict]  # {arg: {type, required?, description?, enum?}}
    execute: Callable[[dict, "ToolExecution"], Any]

    def schema(self) -> dict:
        """The neutral schema. Adapters translate this into their provider's wire shape.

        Deliberately NOT OpenAI's `{type:"function", function:{...}}` nor Anthropic's
        `{name, input_schema}` — if the registry spoke either, it would have picked a side and
        the adapter seam would leak up into tool authoring.
        """
        properties = {}
        required = []
        for arg, spec in self.parameters.items():
            prop = {"type": spec.get("type", "string")}
            if "description" in spec:
                prop["description"] = spec["description"]
            if "enum" in spec:
                prop["enum"] = spec["enum"]
            properties[arg] = prop
            if spec.get("required"):
                required.append(arg)
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        }


@dataclass
class ToolExecution:
    """One in-flight call. Identity is fixed; policy plugins read it to make decisions.

    `token` is the cooperative cancellation signal (phase 6). A tool body that can run long
    MUST poll it -- `exec.token.check()` inside its loop -- or it cannot be interrupted and
    `plugins/timeout.py` can only report a deadline it was unable to enforce.
    """

    call_id: str
    name: str
    arguments: dict
    definition: ToolDefinition | None = None
    token: CancelToken | None = None
    # Delegation position (phase 9). A child agent's executions carry depth+1 and the label
    # chain, so a nested `task` call is measured from where it actually sits rather than from
    # zero -- which is what makes the depth limit and the cycle guard mean anything at all.
    depth: int = 0
    parents: tuple = ()


@dataclass
class ToolResult:
    """The outcome of one call, as the loop and the adapters will see it."""

    call_id: str
    name: str
    content: str
    is_error: bool = False
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Deny:
    """Returned by a pre-execute listener or a guard to refuse a call."""

    reason: str


def define_tool(name: str, description: str, parameters: dict, execute: Callable) -> ToolDefinition:
    return ToolDefinition(name=name, description=description,
                          parameters=parameters, execute=execute)


# --- the service ----------------------------------------------------------------------------


class ToolsService:
    """The registry. Tools may be global or bound to one agent's scope (phase 9).

    Scoping lives here rather than in the loop because visibility is a property of the
    registry: "which tools exist for this caller" is the question `schemas()` answers, and a
    subagent asking it must get a different answer than its parent.
    """

    def __init__(self, ctx) -> None:
        self._ctx = ctx
        self._registry: ScopedRegistry[ToolDefinition] = ScopedRegistry()
        self._guards: list[Callable[[ToolExecution], Deny | None]] = []
        # (scope, allowed_names) pairs, compared by identity. See restrict().
        self._filters: list[tuple[Any, frozenset[str]]] = []

    # -- registration (effect-based, like everything else) --------------------------------

    def register(self, definition: ToolDefinition,
                 *, scope: Any = None) -> Callable[[], None]:
        """Register a tool, globally or into one scope. Disposing the plugin unregisters it.

        `scope` is explicit rather than inferred from the calling context. Inferring it was the
        first design and it cannot work: one ToolsService instance serves every context, so it
        holds the ctx it was *created* with and would have scoped every registration to that
        one regardless of who called. Magic that silently reads the wrong value is worse than
        an argument.

        The duplicate check looks at the SAME scope only, not at what is visible from it —
        global tools are visible inside every scope, so checking visibility would make it
        impossible to shadow a global tool with a scoped one.
        """
        if any(t.name == definition.name for t in self._registry.in_scope(scope)):
            raise RuntimeError(f"tool {definition.name!r} is already registered in this scope")

        return self._registry.add(definition, scope=scope)

    def restrict(self, scope: Any, allowed: list[str] | None) -> Callable[[], None]:
        """Limit what `scope` can see to `allowed`. Returns a disposer.

        This is how a subagent gets a smaller tool set, and it is deliberately NOT done by
        re-registering a filtered copy of every tool into the child's scope. That approach
        collides with the originals, doubles the registry, and has to be undone item by item.
        A filter is one entry, is exact, and removing it restores the child's full view.

        Scope visibility is additive by design — a child must not lose the tools everyone
        already had just by existing — so subtraction needs its own mechanism. This is it.
        """
        if allowed is None:
            return lambda: None
        entry = (scope, frozenset(allowed))
        self._filters.append(entry)

        def dispose() -> None:
            if entry in self._filters:
                self._filters.remove(entry)

        return dispose

    def _allowed(self, scope: Any) -> frozenset[str] | None:
        for filter_scope, names in self._filters:
            if filter_scope is scope:
                return names
        return None

    def guard(self, fn: Callable[[ToolExecution], Deny | None]) -> Callable[[], None]:
        """Register a monotonic final deny. Return a Deny to refuse, or None to abstain.

        Unlike a `tools/pre-execute` listener, a guard cannot be talked out of its answer by
        anything registered later.
        """
        self._guards.append(fn)

        def dispose() -> None:
            if fn in self._guards:
                self._guards.remove(fn)

        return dispose

    # -- introspection ----------------------------------------------------------------------

    def visible(self, scope: Any = None) -> list[ToolDefinition]:
        """Tools this scope can see: everything global plus its own, minus any filter.

        A scoped tool shadows a global one of the same name, so a child can replace a tool
        rather than only add or remove.
        """
        allowed = self._allowed(scope)
        seen: dict[str, ToolDefinition] = {}
        for tool in self._registry.visible(scope):
            seen[tool.name] = tool          # later (scoped) wins over earlier (global)
        tools = list(seen.values())
        if allowed is not None:
            tools = [t for t in tools if t.name in allowed]
        return tools

    def schemas(self, scope: Any = None) -> list[dict]:
        """Neutral schemas for every tool visible to `scope`; the loop hands these to the adapter."""
        return [t.schema() for t in self.visible(scope)]

    def names(self, scope: Any = None) -> list[str]:
        return [t.name for t in self.visible(scope)]

    def lookup(self, name: str, scope: Any = None) -> ToolDefinition | None:
        for tool in self.visible(scope):
            if tool.name == name:
                return tool
        return None

    # -- the pipeline -----------------------------------------------------------------------

    def execute(self, call_id: str, tool_name: str, arguments: dict,
                token: CancelToken | None = None, scope: Any = None,
                depth: int = 0, parents: tuple = ()) -> ToolResult:
        """Run one tool call through the full pipeline. The only method the loop calls.

        Resolution is scope-aware: a tool the caller cannot see is "unknown", not "forbidden".
        That is the right answer — the model was never shown it, so being told it does not
        exist is both true from the caller's position and less informative to probe with.
        """
        definition = self.lookup(tool_name, scope)
        exec_ = ToolExecution(call_id=call_id, name=tool_name, arguments=arguments,
                              definition=definition, token=token,
                              depth=depth, parents=parents)

        if definition is None:
            # An unknown tool is a normal event, not a crash: models hallucinate tool names,
            # and being told so is more useful than an exception. Same reasoning as rule 3 in
            # stages/s03_loop.py -- a failure the model can read is a failure it can recover from.
            return self._finish(exec_, ToolResult(
                call_id, tool_name,
                f"Error: unknown tool {tool_name!r}. "
                f"Available: {', '.join(self.names(scope)) or 'none'}",
                is_error=True,
            ))

        # Stage 1: reorderable policy.
        decision = self._ctx.waterfall("tools/pre-execute", exec_, final=lambda _e: None)
        if isinstance(decision, Deny):
            return self._finish(exec_, ToolResult(
                call_id, tool_name, f"Error: {decision.reason}", is_error=True,
                meta={"denied_by": "tools/pre-execute"},
            ))

        # Stage 2: monotonic guards. Checked AFTER policy so that policy cannot pre-empt them,
        # and so a guard's denial is the one that survives.
        for guard in self._guards:
            verdict = guard(exec_)
            if isinstance(verdict, Deny):
                return self._finish(exec_, ToolResult(
                    call_id, tool_name, f"Error: {verdict.reason}", is_error=True,
                    meta={"denied_by": "guard"},
                ))

        # Stage 3: around-dispatch, wrapping the tool body itself.
        def run_body(e: ToolExecution) -> ToolResult:
            try:
                value = e.definition.execute(e.arguments, e)
                content = value if isinstance(value, str) else json.dumps(value, default=str)
                return ToolResult(call_id, tool_name, content)
            except Cancelled as exc:
                # Cancellation is not a tool bug -- it is the outcome the caller asked for.
                # It still comes back as a result so the model learns why nothing happened.
                return ToolResult(call_id, tool_name, f"Error: {exc.reason}", is_error=True,
                                  meta={"cancelled": True})
            except Exception as exc:  # noqa: BLE001 - tool failures are results, not crashes
                return ToolResult(call_id, tool_name, f"Error: {exc}", is_error=True)

        result = self._ctx.waterfall("tools/execute", exec_, final=run_body)

        # A wrapper that returns something other than a ToolResult (a timeout plugin returning
        # a bare string, say) would otherwise corrupt the loop's message history downstream.
        if not isinstance(result, ToolResult):
            result = ToolResult(call_id, tool_name, str(result))

        # Stage 4: transform / block / annotate.
        result = self._ctx.waterfall("tools/post-execute", exec_, result, final=lambda _e, r: r)
        return self._finish(exec_, result)

    def _finish(self, exec_: ToolExecution, result: ToolResult) -> ToolResult:
        """Stage 5: observation. Listeners see the final outcome and cannot change it."""
        self._ctx.emit("tools/result", exec_, result)
        return result


def apply(ctx, config=None) -> None:
    ctx.provide("tools", ToolsService(ctx))
