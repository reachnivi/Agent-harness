"""An LLM-summarizing compaction backend, plus the auto-trigger.

WHY THE SUMMARY PROMPT IS SO SPECIFIC

    A generic "summarize this" produces prose, and prose is the wrong output. What survives
    compaction is the only thing the agent will still know, so the summary has to preserve the
    operational facts -- which files were touched, what the user actually asked for, what was
    tried and failed -- and may freely discard the narration.

    A summary that reads well and omits "the user said do NOT touch config.py" is worse than no
    compaction at all, because the agent now believes it has the full picture.

WHERE THE AUTO-TRIGGER LIVES

    On `turn/end`, not inside the loop. Compaction between steps would change the history
    underneath a turn that is still in flight -- the assistant message asking for tools could
    be shadowed before its results arrive, which both wire protocols reject. A turn boundary is
    the only safe moment, which is the same reason fork uses it.
"""

from __future__ import annotations

from dshpy.services.compaction import CompactionBackend
from dshpy.services.llm import GenerateOptions, Message

name = "compaction-basic"
inject = ["compaction", "sessions", "llm"]

SUMMARY_PROMPT = """\
Summarize the conversation below so another assistant can continue the work without having \
read it. This summary REPLACES the original: anything you leave out is lost.

Preserve, concretely:
  - what the user asked for, including any constraints or prohibitions
  - files read or modified, by path
  - decisions made and why
  - what was tried and failed, so it is not retried
  - anything still outstanding

Discard narration, pleasantries, and tool output that no longer matters.
Write compact prose or bullets. Do not invent anything not present below.

CONVERSATION:
"""


class LlmSummarizer(CompactionBackend):
    def __init__(self, model: str | None = None, provider: str | None = None,
                 max_tokens: int = 1024) -> None:
        self.model = model
        self.provider = provider
        self.max_tokens = max_tokens

    def summarize(self, ctx, events: list, *, reason: str) -> str:
        transcript = "\n".join(_render(e) for e in events if _render(e))

        loop = ctx.agent_loop if ctx.has("agent_loop") else None
        model = self.model or (loop.model if loop else "")
        provider = self.provider or (loop.provider if loop else None)

        completion = ctx.llm.generate(
            [Message(role="user", content=SUMMARY_PROMPT + transcript)],
            tools=[],                    # a summarizer must not call tools
            model=model,
            provider=provider,
            max_tokens=self.max_tokens,
        )
        if completion.finish_reason in {"error", "aborted"}:
            # Fail loudly: silently compacting to an error string would delete the history and
            # replace it with nothing, which is the worst possible outcome here.
            raise RuntimeError(f"summarization failed: {completion.error}")
        return completion.text.strip() or "(the summarizer returned nothing)"


def _render(event) -> str:
    data = event.data
    if event.kind == "user/message":
        return f"USER: {data.get('content','')}"
    if event.kind == "assistant/message":
        calls = ", ".join(f"{c['name']}({c['arguments']})" for c in data.get("tool_calls", []))
        text = data.get("content", "")
        return f"ASSISTANT: {text}" + (f"\n  calls: {calls}" if calls else "")
    if event.kind == "tool/result":
        flag = " [error]" if data.get("is_error") else ""
        return f"TOOL {data.get('name')}{flag}: {str(data.get('content',''))[:500]}"
    return ""


def apply(ctx, config=None) -> None:
    config = config or {}
    backend = LlmSummarizer(model=config.get("model"), provider=config.get("provider"),
                            max_tokens=config.get("max_tokens", 1024))
    ctx.effect(ctx.compaction.register_backend(backend))

    threshold = config.get("threshold_tokens")
    keep_last_turns = config.get("keep_last_turns", 2)
    if not threshold:
        return

    def maybe_compact(_final_text) -> None:
        # Turn boundary only: compacting between steps could shadow an assistant message whose
        # tool results have not arrived yet, which both wire protocols reject as malformed.
        if ctx.sessions.estimate_tokens() < threshold:
            return
        try:
            result = ctx.compaction.compact(keep_last_turns=keep_last_turns, reason="auto")
        except Exception as exc:  # noqa: BLE001 - a failed compaction must not kill the turn
            print(f"[compaction] skipped: {exc}")
            return
        if result:
            print(f"[compaction] {result.shadowed_events} events summarized, "
                  f"~{result.tokens_before} -> ~{result.tokens_after} tokens")

    ctx.on("turn/end", maybe_compact)
