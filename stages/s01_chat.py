"""Stage 1 — Conversation state.

CONCEPT — the single most important idea in the whole harness:

    The API is STATELESS. It remembers nothing between calls.

    The `messages` list IS the agent's entire memory, and your code owns it. Every feature you
    will ever read about — "memory", "context window management", "compaction", "sessions" —
    is something done to this list. There is no server-side conversation to fetch.

    Stage 0 sent a one-item list. A chat is the same call with a longer list. That's the whole
    difference, and it's worth sitting with for a second, because it means the model is not
    "having a conversation" — it is being handed a transcript and asked what comes next.

RUN
    uv run stages/s01_chat.py

    Tell it your name. Talk about something else for three turns. Then ask what your name is.

TRY BREAKING IT
    1. Change `messages.append({"role": "assistant", "content": resp.content})` to send only
       the *last* two messages each turn. Watch the memory disappear. That is compaction done
       badly, and it's the failure mode every context-management strategy is trying to avoid.
    2. Type `dump` at the prompt to print the raw list. Do this at least once — it's the one
       time you should look directly at the structure everything else is built on.
    3. Comment out the assistant append entirely. The model will repeat itself forever, because
       from its side every turn looks like the first.
"""

from __future__ import annotations

import sys

from harness.client import MAX_TOKENS, MODEL, PreflightError, build_client, preflight


def main() -> int:
    try:
        preflight()
    except PreflightError as exc:
        print(f"\nPreflight failed:\n{exc}\n", file=sys.stderr)
        return 1

    client = build_client()

    # THIS LIST IS THE AGENT. Everything else in this file is plumbing around it.
    messages: list[dict] = []

    print("Chat with the model. Ctrl-D or 'quit' to exit, 'dump' to inspect raw state.\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except EOFError:
            print()
            return 0

        if user_input in {"quit", "exit"}:
            return 0
        if not user_input:
            continue
        if user_input == "dump":
            _dump(messages)
            continue

        # 1. Your turn goes on the end of the list.
        messages.append({"role": "user", "content": user_input})

        # 2. Send the WHOLE list. Not just the new message — the whole thing, every time.
        #    This is why long conversations get expensive: you re-send the transcript each turn.
        resp = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=messages,
        )

        # 3. The reply goes on the end of the list too, and THIS is the line to pay attention to.
        #
        #    We append `resp.content` — the LIST OF BLOCKS — not `resp.content[0].text`.
        #
        #    Right now those look equivalent: a plain text reply is one text block, so pulling
        #    out the string would work fine and you'd never notice. From stage 2 onward that
        #    list also carries `tool_use` blocks, and flattening it to a string silently drops
        #    them. The next request then references a tool call that isn't in the history, and
        #    the API rejects it with an error that does not obviously point back here.
        #
        #    Store the structure. Render the string. Never confuse the two.
        messages.append({"role": "assistant", "content": resp.content})

        print(f"bot> {_text_of(resp.content)}\n")


def _text_of(content: list) -> str:
    """Render blocks for display. Note this is a *view* — we never store the flattened form."""
    return "".join(block.text for block in content if block.type == "text")


def _dump(messages: list[dict]) -> None:
    print(f"\n--- {len(messages)} messages ---")
    for i, msg in enumerate(messages):
        content = msg["content"]
        if isinstance(content, str):
            summary = repr(content[:70])
        else:
            # Assistant turns are lists of block objects, not strings. Show their types.
            summary = f"[{', '.join(type(b).__name__ for b in content)}]"
        print(f"  {i}: {msg['role']:<9} {summary}")
    print("---\n")


if __name__ == "__main__":
    raise SystemExit(main())
