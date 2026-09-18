"""Stage 0 — Hello, model.

CONCEPT
    There is no "agents API". Everything — tool use, streaming, thinking, agents that run for an
    hour — goes through a single endpoint: POST /v1/messages. Every capability you add over the
    next seven stages is structure *you* build around that one call.

    Get this straight now and you'll stop looking for a feature that doesn't exist.

WHAT THIS STAGE PROVES
    That your setup works. That's all. It's boring on purpose: if the connection or the model is
    wrong, every later stage fails with a much less obvious error.

RUN
    ollama serve            # in another terminal
    uv run stages/s00_hello.py

TRY BREAKING IT
    1. Set MODEL=<something-not-pulled> in .env and re-run. Read the preflight error.
    2. Set MODEL to a model without tool support (e.g. an embedding model) and re-run —
       this is the failure that would have cost you an hour at stage 2.
    3. Delete `max_tokens` from the call below. The SDK will refuse: it's a required argument,
       because an unbounded generation is an unbounded bill.
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

    # The entire Claude API, in one call. Note what's being passed:
    #
    #   model       - which model answers.
    #   max_tokens  - a hard ceiling on the reply. Required. Lowballing it truncates mid-sentence.
    #   messages    - a LIST. Right now it has one entry. By stage 3 this list is the agent's
    #                 entire memory, and managing it is most of what a harness does.
    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": "Say hello in exactly five words."}],
    )

    # `resp.content` is a LIST OF BLOCKS, not a string.
    #
    # Taking content[0].text works fine here, because a plain text reply is a single text block.
    # It stops working in stage 2, where the list also carries `tool_use` blocks. Remember that
    # you got away with it this once.
    print(resp.content[0].text)

    # Worth a look: stop_reason is "end_turn" here, meaning "I'm done talking". In stage 2 you'll
    # see "tool_use" instead, and that one value is what drives the entire agent loop.
    print(f"\n[stop_reason: {resp.stop_reason}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
