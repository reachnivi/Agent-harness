"""A scripted stand-in for `anthropic.Anthropic`, for testing the loop without a model.

WHY THIS EXISTS

    Agent loops are miserable to debug against a real model. Every run is non-deterministic, so
    a failure might be your bug or might be the model having an off day, and you can't tell
    which. You end up changing code and re-rolling the dice.

    This fake removes the model from the equation. You script exactly what comes back, then
    assert on the message history your loop built. Failures become deterministic and instant,
    and the four rules from stage 3 become things you can actually *check* rather than things
    you hope you remembered.

    This is a general technique, not a toy for this repo: the seam is `client.messages.create`,
    and any agent harness worth the name keeps that seam injectable. That's exactly why
    `run_agent()` takes `client` as an argument.

WHAT IT MIMICS
    Just enough of the SDK's response shape for the loop: objects with `.content` (a list of
    blocks carrying `.type`, and then `.text` or `.id`/`.name`/`.input`) and `.stop_reason`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class FakeResponse:
    content: list
    stop_reason: str


# --- convenience builders, so tests read like the conversation they describe ---------------


def says(text: str) -> FakeResponse:
    """A final answer — the model is done talking."""
    return FakeResponse(content=[TextBlock(text)], stop_reason="end_turn")


def wants(*calls: tuple[str, str, dict], narration: str = "") -> FakeResponse:
    """A turn requesting one or more tool calls.

    Each call is (id, name, input). Passing several models *parallel* tool use — one assistant
    turn carrying multiple tool_use blocks, which is what rule 1 is about.
    """
    content: list[Any] = [TextBlock(narration)] if narration else []
    content += [ToolUseBlock(id=i, name=n, input=inp) for i, n, inp in calls]
    return FakeResponse(content=content, stop_reason="tool_use")


class FakeMessages:
    def __init__(self, script: list[FakeResponse]) -> None:
        self._script = list(script)
        self.calls: list[dict] = []  # every kwargs dict the loop sent us

    def create(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        if not self._script:
            raise AssertionError(
                f"loop made {len(self.calls)} API calls but only "
                f"{len(self.calls) - 1} responses were scripted"
            )
        return self._script.pop(0)


class FakeClient:
    """Drop-in for `anthropic.Anthropic` as far as `run_agent()` is concerned."""

    def __init__(self, script: list[FakeResponse]) -> None:
        self.messages = FakeMessages(script)

    @property
    def calls(self) -> list[dict]:
        return self.messages.calls

    def last_messages(self) -> list[dict]:
        """The `messages` list as it looked on the most recent API call."""
        return self.messages.calls[-1]["messages"]
