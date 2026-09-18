"""Stage 2 — One tool, one round trip.

CONCEPT:

    The model NEVER executes anything.

    It emits a structured *request* to run something. Your code decides whether to run it, runs
    it, and hands back the result. The entire trust boundary of an agent lives in that gap —
    which is why stage 6 (permissions) is possible at all.

    If you internalise one thing here: a "tool call" is just the model producing some JSON. All
    the power, and all the danger, is in what your code chooses to do with that JSON.

THE SHAPE OF A TOOL ROUND TRIP (four messages, two API calls):

    1. user       "what time is it?"
    2. assistant  [tool_use  id=abc123  name=get_time  input={}]      <- stop_reason="tool_use"
    3. user       [tool_result  tool_use_id=abc123  content="..."]    <- YOU produce this
    4. assistant  "It's 3pm."                                         <- stop_reason="end_turn"

RUN
    uv run stages/s02_one_tool.py            # asks "what time is it?"
    uv run stages/s02_one_tool.py "what day of the week is it?"

TRY BREAKING IT
    1. Corrupt the id: change `tool_use_id=block.id` to `tool_use_id="wrong"`. Read the API
       error. You want to recognise it instantly later, because you WILL cause it again.
    2. Put the tool_result in an {"role": "assistant"} message instead of "user". Also an error.
       Tool results are user-role. It feels backwards; it's the protocol.
    3. In stage 1's terms: change the assistant append to `resp.content[0].text` and watch this
       break. That's the gotcha from stage 1 finally biting — the tool_use block gets dropped,
       so message 3 refers to a call that no longer exists in the history.
    4. Delete the `description` from the tool definition. It will get called less reliably, or
       with worse arguments. The description is a prompt, not documentation.
"""

from __future__ import annotations

import sys
from datetime import datetime

from harness.client import MAX_TOKENS, MODEL, PreflightError, build_client, preflight

# ---------------------------------------------------------------------------
# The tool, as the MODEL sees it: a name, a description, and a JSON Schema.
#
# This dict is sent to the model on every request. It is prompt text with extra steps, so the
# `description` field genuinely matters — it's how the model decides *when* to call this.
# "Get the current time" gets used; "time utility" doesn't.
#
# `input_schema` is JSON Schema. This tool takes no arguments, hence the empty properties.
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "get_time",
        "description": (
            "Get the current date and time in ISO 8601 format. "
            "Use this whenever the user asks about the current time, date, or day of the week."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    }
]


# The tool, as YOUR CODE sees it: an ordinary function. The model cannot reach this. It can only
# ask, by name, for it to be run.
def get_time() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def main(argv: list[str]) -> int:
    try:
        preflight()
    except PreflightError as exc:
        print(f"\nPreflight failed:\n{exc}\n", file=sys.stderr)
        return 1

    client = build_client()
    question = argv[1] if len(argv) > 1 else "What time is it?"

    messages: list[dict] = [{"role": "user", "content": question}]
    print(f"you> {question}\n")

    # --- API CALL 1: ask, with tools available ------------------------------------------------
    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        tools=TOOLS,  # <- the only difference from stage 1
        messages=messages,
    )

    # `stop_reason` is the signal that drives everything.
    #   "tool_use" -> the model wants something run before it can answer
    #   "end_turn" -> it's done talking
    print(f"[stop_reason: {resp.stop_reason}]")

    if resp.stop_reason != "tool_use":
        # The model answered directly without calling the tool. On a local model this is common
        # for weaker prompts — it's not necessarily a bug, but if it NEVER calls the tool, check
        # that your model really supports tools (stage 0's preflight).
        print(f"bot> {_text_of(resp.content)}")
        return 0

    # Keep the assistant turn — INCLUDING the tool_use block — in the history.
    messages.append({"role": "assistant", "content": resp.content})

    # --- Execute --------------------------------------------------------------------------
    # Find the tool_use block. Note we iterate: `resp.content` may also hold a text block where
    # the model narrates what it's about to do. Never assume content[0] is the one you want.
    tool_use = next(b for b in resp.content if b.type == "tool_use")
    print(f"[model requested: {tool_use.name}({tool_use.input}) id={tool_use.id}]")

    result = get_time()  # stage 3 replaces this hardcoded call with real dispatch
    print(f"[we ran it, got: {result}]")

    # --- Hand the result back ---------------------------------------------------------------
    # Three things that trip everyone up, all in this one block:
    #   1. role is "user", not "assistant". Tool results come from your side of the conversation.
    #   2. tool_use_id must match the id from the tool_use block exactly. It is the correlation
    #      key; get it wrong and the API rejects the whole request.
    #   3. content is a STRING. Whatever your function returned, serialise it.
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result,
                }
            ],
        }
    )

    # --- API CALL 2: same call, longer list --------------------------------------------------
    # Note this is identical to call 1. The model now sees the tool result in its history and
    # can answer. That symmetry is the hint that this wants to be a loop — which is stage 3.
    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        tools=TOOLS,
        messages=messages,
    )
    print(f"[stop_reason: {resp.stop_reason}]")
    print(f"\nbot> {_text_of(resp.content)}")
    return 0


def _text_of(content: list) -> str:
    return "".join(block.text for block in content if block.type == "text")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
