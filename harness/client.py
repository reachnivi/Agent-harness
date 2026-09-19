"""Provider configuration — the one place in this project that knows where the model lives.

Every stage imports `build_client()` and `MODEL` from here, and that is deliberate. By the end of
stage 6 you will have a working agent harness, and the *only* thing standing between "runs against
a local Ollama model" and "runs against Claude in the cloud" will be three environment variables.

That is the first real lesson of the project:

    The harness is the loop. It is not the provider.

Why the official `anthropic` SDK against a local model?
    Ollama v0.14+ implements Anthropic's Messages API. Same SDK, same request shape, same
    `tool_use` / `tool_result` content blocks. So nothing you learn in these stages is
    Ollama-specific — you are learning the real API, just with a free local model behind it.

What Ollama's compatibility layer does NOT implement (flagged again where it matters):
    - `tool_choice`            — we never force a tool, so this never bites us.
    - extended thinking        — stage 5: don't pass `thinking=`.
    - `count_tokens` endpoint  — stage 7: token counts are approximations.
    - prompt caching           — irrelevant at this scale.
    - server-side tools        — stage 7: `web_search` needs a real Anthropic key.
"""

from __future__ import annotations

import os

import anthropic
from dotenv import load_dotenv

from common.ollama import PreflightError, preflight as _preflight

# Must run before the os.environ reads below, or a .env file would be ignored.
load_dotenv()

# Defaults point at a local Ollama. Override in .env to use the hosted Anthropic API instead.
#
# Why HARNESS_* and not ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY?
#   Because the SDK reads those itself, and they are often already set globally (a corporate
#   proxy, another project, a CI runner). If we used those names, a stray value in your shell
#   would silently redirect this project somewhere else and you'd debug the wrong thing.
#   This is not hypothetical — it happened while writing this file.
#   Own your config namespace. It is two extra characters and it removes a whole class of bug.
BASE_URL = os.environ.get("HARNESS_BASE_URL", "http://localhost:11434")
API_KEY = os.environ.get("HARNESS_API_KEY", "ollama")  # required by the SDK, ignored by Ollama
MODEL = os.environ.get("HARNESS_MODEL", "qwen3-coder")

# Local models are slower than the hosted API, and nothing in stages 0-6 needs a long reply.
# Against the real API you'd want ~16000 here (or ~64000 once streaming, in stage 5).
MAX_TOKENS = int(os.environ.get("HARNESS_MAX_TOKENS", "1024"))



def build_client() -> anthropic.Anthropic:
    """The Anthropic SDK client, pointed wherever the environment says.

    Note there is no `if ollama: ... else: ...` here. There doesn't need to be — that is the
    whole point of Ollama speaking the Messages API.
    """
    return anthropic.Anthropic(base_url=BASE_URL, api_key=API_KEY)


def preflight() -> None:
    """Check the local setup. Implementation lives in `common/ollama.py`.

    It moved there because it is not Anthropic-specific: it talks to Ollama's own native
    endpoints, which are the same whichever protocol you then speak. The plugin harness in
    `dshpy/` uses the identical function.
    """
    _preflight(BASE_URL, MODEL)


__all__ = ["BASE_URL", "API_KEY", "MODEL", "MAX_TOKENS",
           "build_client", "preflight", "PreflightError"]
