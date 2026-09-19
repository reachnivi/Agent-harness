"""`ctx.agent_loop` — the agent loop, mounted as a plugin like everything else.

This is the claim that makes dsh's architecture more than a tool registry: **the loop itself is
a plugin.** It provides a service, declares what it injects, and can be unmounted and replaced
from configuration without patching anything.

WHAT IS AND ISN'T HERE

    Compare `run_agent()` in stages/s03_loop.py, which knew: the wire format, the tool list, how
    to dispatch a tool by name, and how to format tool results for one specific provider.

    This loop knows none of that. It reads schemas from `ctx.tools`, calls `ctx.llm.generate()`,
    and hands results to `ctx.tools.execute()`. Swapping the provider, adding a tool, gating a
    tool behind a permission prompt, or putting a timeout around dispatch are all things that
    happen *elsewhere* and require no edit here.

    The four rules from stage 3 are still in force — they just moved. Rules 1 and 2 (how results
    are shaped and returned) are now the adapter's problem, because they differ per protocol.
    Rule 3 (failures are results) lives in the tools pipeline. Only rule 4 — bound the loop —
    is still visibly the loop's job, because only the loop knows what a step is.
"""

from __future__ import annotations

from dshpy.services.llm import Message

name = "agent-loop"
inject = ["llm", "tools", "sessions"]

DEFAULT_MAX_STEPS = 8


class AgentLoopService:
    def __init__(self, ctx, config: dict) -> None:
        self._ctx = ctx
        self.model: str = config.get("model", "qwen3-coder")
        self.provider: str | None = config.get("provider")
        self.max_steps: int = config.get("max_steps", DEFAULT_MAX_STEPS)
        self.max_tokens: int | None = config.get("max_tokens")
        self.system: str | None = config.get("system")

    def run(self, user_input: str) -> str:
        ctx = self._ctx
        sessions, tools, llm = ctx.sessions, ctx.tools, ctx.llm

        if self.system and not any(e.kind == "system/message" for e in sessions.events()):
            sessions.append("system/message", content=self.system)
        sessions.append("user/message", content=user_input)

        ctx.emit("turn/start", user_input)
        final_text = ""

        for step in range(self.max_steps):  # rule 4: bounded
            ctx.emit("step/start", step)

            completion = llm.generate(
                sessions.derive_messages(),
                tools=tools.schemas(),
                model=self.model,
                provider=self.provider,
                max_tokens=self.max_tokens,
            )

            sessions.append(
                "assistant/message",
                content=completion.text,
                tool_calls=[vars(tc) for tc in completion.tool_calls],
            )
            final_text = completion.text

            if not completion.tool_calls:
                break

            for call in completion.tool_calls:
                sessions.append("tool/call", call_id=call.id,
                                name=call.name, arguments=call.arguments)
                # Everything interesting -- policy, guards, timeouts, result rewriting --
                # happens inside this one call, contributed by plugins the loop never names.
                result = tools.execute(call.id, call.name, call.arguments)
                sessions.append("tool/result", call_id=result.call_id, name=result.name,
                                content=result.content, is_error=result.is_error)

            ctx.emit("step/end", step)
        else:
            ctx.emit("turn/truncated", self.max_steps)

        ctx.emit("turn/end", final_text)
        return final_text


def apply(ctx, config=None) -> None:
    ctx.provide("agent_loop", AgentLoopService(ctx, config or {}))
