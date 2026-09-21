"""The `task` tool -- the model-facing consumer of `ctx.subagents`.

WHY DELEGATION IS WORTH A TOOL AT ALL

    Context. A parent asked to "find every place we parse dates" would otherwise read thirty
    files into its own history and then still have the real work to do, with most of its
    context spent on files it will never mention again.

    Delegating that search gives the parent back one paragraph instead of thirty files. The
    child's twenty tool calls land in the child's log and are discarded with it.

    So the description below tells the model *when* delegating is worth it, not just that it
    can. A `task` tool with a vague description gets used for everything or nothing.

THE ANCESTRY CHAIN

    Each request carries its parents, so a cycle can be detected. A depth limit alone does not
    catch three agents delegating in a ring -- each stays at depth 1 while the ring runs
    forever.
"""

from __future__ import annotations

from dshpy.services.subagents import SubagentError, SubagentRequest
from dshpy.services.tools import define_tool

name = "tool-subagent"
inject = ["tools", "subagents"]


def apply(ctx, config=None) -> None:
    config = config or {}
    max_depth = config.get("max_depth", 3)
    default_tools = config.get("allowed_tools")

    def task(args, exec):
        label = args.get("label") or args["task"][:40]
        requested = args.get("tools")
        allowed = requested if requested is not None else default_tools

        request = SubagentRequest(
            task=args["task"],
            label=label,
            allowed_tools=allowed,
            max_depth=max_depth,
            # Depth and ancestry travel with the execution, so a child delegating again is
            # measured from where it actually sits rather than from zero.
            depth=getattr(exec, "depth", 0),
            parents=getattr(exec, "parents", ()),
            max_steps=int(args.get("max_steps", config.get("max_steps", 8))),
        )

        try:
            result = ctx.subagents.start(request, provider=args.get("provider"))
        except SubagentError as exc:
            # A refusal the model can read and act on, rather than a crash.
            return f"Delegation refused: {exc}"

        prefix = "[subagent failed] " if result.is_error else ""
        return f"{prefix}{result.label} ({result.steps} steps):\n{result.output}"

    ctx.effect(ctx.tools.register(define_tool(
        name="task",
        description=(
            "Delegate a self-contained piece of work to a sub-agent and get back a short "
            "summary. Worth it when the work needs to read a lot to produce a little — "
            "searching a codebase, checking many files, investigating an unfamiliar area. "
            "The sub-agent has its own context and cannot ask you questions, so state the task "
            "completely. Not worth it for a single tool call you could make yourself."
        ),
        parameters={
            "task": {"type": "string", "required": True,
                     "description": "The complete, self-contained instruction for the sub-agent."},
            "label": {"type": "string", "description": "A short name for this sub-task."},
            "tools": {"type": "array",
                      "description": "Restrict the sub-agent to these tool names."},
            "max_steps": {"type": "number"},
        },
        execute=task,
    )))
