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
    """One in-flight call. Identity is fixed; policy plugins read it to make decisions."""

    call_id: str
    name: str
    arguments: dict
    definition: ToolDefinition | None = None


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
    def __init__(self, ctx) -> None:
        self._ctx = ctx
        self._tools: dict[str, ToolDefinition] = {}
        self._guards: list[Callable[[ToolExecution], Deny | None]] = []

    # -- registration (effect-based, like everything else) --------------------------------

    def register(self, definition: ToolDefinition) -> Callable[[], None]:
        """Register a tool. Disposing the owning plugin unregisters it."""
        if definition.name in self._tools:
            raise RuntimeError(f"tool {definition.name!r} is already registered")
        self._tools[definition.name] = definition

        def dispose() -> None:
            if self._tools.get(definition.name) is definition:
                del self._tools[definition.name]

        return dispose

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

    def schemas(self) -> list[dict]:
        """Neutral schemas for every registered tool; the loop hands these to the adapter."""
        return [t.schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    # -- the pipeline -----------------------------------------------------------------------

    def execute(self, call_id: str, tool_name: str, arguments: dict) -> ToolResult:
        """Run one tool call through the full pipeline. The only method the loop calls."""
        definition = self._tools.get(tool_name)
        exec_ = ToolExecution(call_id=call_id, name=tool_name,
                              arguments=arguments, definition=definition)

        if definition is None:
            # An unknown tool is a normal event, not a crash: models hallucinate tool names,
            # and being told so is more useful than an exception. Same reasoning as rule 3 in
            # stages/s03_loop.py -- a failure the model can read is a failure it can recover from.
            return self._finish(exec_, ToolResult(
                call_id, tool_name,
                f"Error: unknown tool {tool_name!r}. Available: {', '.join(self.names()) or 'none'}",
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
