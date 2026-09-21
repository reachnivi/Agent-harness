"""The default profile — a *set* of plugins, not a boot sequence.

In dsh, a **profile** is a named composition of plugin rows. This is the small version of that
idea: a list of `{plugin, config}` dicts that `Runtime.mount_all()` mounts.

THE THING WORTH NOTICING

    The order below is for human readability only. `inject` decides activation order, so you
    can shuffle these rows freely and the harness boots identically — there is a test for it.
    That is the practical difference between a plugin system and a boot script.

SWAPPING THE PROVIDER

    Change `PROVIDER` below (or pass `--provider`). The loop, the tools, the permission policy
    and the telemetry are all untouched; only which adapter claims `ctx.llm`'s route changes,
    and the two adapters speak entirely different wire protocols.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from dshpy.plugins import (
    llm_anthropic,
    llm_openai,
    llm_retry,
    permission,
    stream_ui,
    telemetry,
    timeout,
    token_meter,
    tool_core,
)
from dshpy.services import agent_loop, llm, sessions, tools

load_dotenv()

# DS_* rather than ANTHROPIC_*/OPENAI_*: never reuse an SDK's own variable names, or a value
# already set in your shell for something else silently redirects this project. (That is not a
# hypothetical — it happened while writing harness/client.py on the other branch.)
PROVIDER = os.environ.get("DS_PROVIDER", "openai")     # "openai" | "anthropic"
MODEL = os.environ.get("DS_MODEL", "qwen3-coder")
API_KEY = os.environ.get("DS_API_KEY", "ollama")
BASE_URL = os.environ.get("DS_BASE_URL", "")           # blank -> per-adapter default

SYSTEM_PROMPT = (
    "You are a concise assistant with tools. Use a tool when it is the right way to get an "
    "exact answer; never guess at arithmetic or the current time."
)


def _adapter_row() -> dict:
    """Exactly one adapter claims `ctx.llm`'s route. This is the swap."""
    if PROVIDER == "anthropic":
        return {
            "plugin": llm_anthropic,
            "config": {
                "routes": ["anthropic"],
                # Ollama serves the Anthropic Messages API at the root, not under /v1.
                "base_url": BASE_URL or "http://localhost:11434",
                "api_key": API_KEY,
            },
        }
    return {
        "plugin": llm_openai,
        "config": {
            "routes": ["openai"],
            # ...but the OpenAI-compatible path IS under /v1. The one asymmetry worth knowing.
            "base_url": BASE_URL or "http://localhost:11434/v1",
            "api_key": API_KEY,
        },
    }


def rows() -> list[dict]:
    return [
        # --- services ---------------------------------------------------------------------
        {"plugin": llm},
        {"plugin": tools},
        {"plugin": sessions},

        # --- the model adapter: the swappable one -----------------------------------------
        _adapter_row(),

        # --- capabilities -----------------------------------------------------------------
        {"plugin": tool_core},

        # --- resilience and accounting, both llm/stream listeners -------------------------
        {"plugin": llm_retry},
        {"plugin": token_meter},

        # --- policy and observation, none of which the loop knows about --------------------
        {"plugin": permission, "config": {"default": "ask"}},
        {"plugin": timeout, "config": {"seconds": 30.0}},
        {"plugin": telemetry, "config": {"verbose": True}},
        {"plugin": stream_ui},

        # --- the loop, mounted like anything else -----------------------------------------
        {"plugin": agent_loop, "config": {
            "model": MODEL,
            "provider": PROVIDER,
            "system": SYSTEM_PROMPT,
            "max_steps": 8,
            "max_tokens": 1024,
        }},
    ]
