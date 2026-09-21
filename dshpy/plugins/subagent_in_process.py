"""An in-process subagent provider: a child agent sharing the runtime, with its own session.

WHAT THE CHILD SHARES AND WHAT IT DOES NOT

    Shares:  the runtime, the services, the LLM adapter, the event bus.
    Its own: a session log, a step budget, and -- the interesting part -- a TOOL SCOPE.

    The tool scope is the first real customer of `core/scope.py`. The child gets a forked
    context whose `scope_key` is the child's own object identity, and the parent's tools are
    re-registered into that scope minus anything the request excluded. So the child sees a
    restricted set while the parent's own view is untouched, and both exist at the same time --
    which is exactly what unmount-and-remount could never do.

WHY THE CHILD GETS A FRESH SESSION

    The point of delegating is context isolation: the parent hands over a task description and
    gets back a summary, without the child's twenty tool calls landing in the parent's history.
    Sharing the log would delegate the work and keep all the context cost, which is the one
    thing delegation is for.
"""

from __future__ import annotations

from dshpy.cancel import NEVER
from dshpy.services.agent_loop import AgentLoopService
from dshpy.services.sessions import SessionsService
from dshpy.services.subagents import (
    Capabilities,
    SubagentProvider,
    SubagentRequest,
    SubagentResult,
)

name = "subagent-in-process"
inject = ["subagents", "tools", "llm"]


class _Child:
    """An opaque identity, used as the scope key. The scope primitive never inspects it."""

    def __init__(self, label: str) -> None:
        self.label = label

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Child {self.label}>"


class InProcessSubagent(SubagentProvider):
    name = "in-process"
    capabilities = Capabilities(depth_limit=True, tool_filter=True, cancellation=True)

    def __init__(self, model: str = "", provider: str | None = None,
                 system: str | None = None) -> None:
        self.model = model
        self.provider = provider
        self.system = system or (
            "You are a focused sub-agent. Do exactly the task you are given, using the tools "
            "available, and end with a short factual summary of what you found or changed. "
            "You cannot ask questions -- if something is ambiguous, state the assumption you "
            "made and proceed."
        )

    def start(self, ctx, request: SubagentRequest) -> SubagentResult:
        child = _Child(request.label or "subagent")
        child_ctx = ctx.fork(child)   # same services, registrations bound to `child`

        # Restrict what the child's scope can see. One filter entry, not a filtered COPY of
        # every tool re-registered into the child's scope -- that was the first design and it
        # collided with the originals, since global tools are visible inside every scope.
        restore = ctx.tools.restrict(child, request.allowed_tools)

        # Its own session: the child's tool calls must not land in the parent's history.
        child_sessions = SessionsService(child_ctx)
        loop = AgentLoopService(child_ctx, {
            "model": self.model or _parent_model(ctx),
            "provider": self.provider or _parent_provider(ctx),
            "system": self.system,
            "max_steps": request.max_steps,
        })
        loop._sessions_override = child_sessions   # read by AgentLoopService.run
        loop._scope = child
        # The child sits one level deeper, and its ancestry grows by its own label. A
        # nested `task` call from inside the child is then measured correctly.
        loop._depth = request.depth + 1
        loop._parents = request.parents + (child.label,)

        try:
            output = loop.run(request.task, token=NEVER)
            steps = sum(1 for e in child_sessions.events() if e.kind == "assistant/message")
            return SubagentResult(label=child.label, output=output, steps=steps,
                                  events=child_sessions.events())
        except Exception as exc:  # noqa: BLE001 - a child failure is the parent's result
            return SubagentResult(label=child.label, output=f"subagent failed: {exc}",
                                  is_error=True)
        finally:
            restore()


def _parent_model(ctx) -> str:
    return ctx.agent_loop.model if ctx.has("agent_loop") else ""


def _parent_provider(ctx):
    return ctx.agent_loop.provider if ctx.has("agent_loop") else None


def apply(ctx, config=None) -> None:
    config = config or {}
    provider = InProcessSubagent(model=config.get("model", ""),
                                 provider=config.get("provider"),
                                 system=config.get("system"))
    ctx.effect(ctx.subagents.register_provider(provider))
