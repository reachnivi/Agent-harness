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

import json
import os
import urllib.error
import urllib.request

import anthropic
from dotenv import load_dotenv

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

# Only meaningful when pointed at a local Ollama; used by preflight() for its native endpoints.
_IS_OLLAMA = "localhost" in BASE_URL or "127.0.0.1" in BASE_URL


def build_client() -> anthropic.Anthropic:
    """The Anthropic SDK client, pointed wherever the environment says.

    Note there is no `if ollama: ... else: ...` here. There doesn't need to be — that is the
    whole point of Ollama speaking the Messages API.
    """
    return anthropic.Anthropic(base_url=BASE_URL, api_key=API_KEY)


class PreflightError(RuntimeError):
    """Setup is wrong in a way that would otherwise surface as a confusing bug later."""


def _ollama_get(path: str, payload: dict | None = None) -> dict:
    """Call one of Ollama's *native* endpoints (not the Anthropic-compatible ones)."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def preflight() -> None:
    """Fail loudly *now* instead of mysteriously at stage 2.

    The check that earns its keep is the tool-capability one. A model without tool support does
    not raise an error when you pass `tools=[...]` — it just answers in prose and never emits a
    `tool_use` block. That looks exactly like a bug in your agent loop, and you can lose an hour
    to it. Ten lines here, one hour saved there.
    """
    if not _IS_OLLAMA:
        print(f"[preflight] Using a remote API at {BASE_URL} — skipping local checks.")
        return

    # 1. Is the server even up?
    try:
        version = _ollama_get("/api/version").get("version", "unknown")
    except (urllib.error.URLError, OSError) as exc:
        raise PreflightError(
            f"Can't reach Ollama at {BASE_URL} ({exc}).\n"
            f"  Start it with:  ollama serve"
        ) from exc
    print(f"[preflight] Ollama {version} is up at {BASE_URL}")

    # 2. Is the model actually pulled? Ollama tags include a ':latest' suffix we must tolerate.
    tags = _ollama_get("/api/tags").get("models", [])
    names = [m["name"] for m in tags]
    if not any(n == MODEL or n.split(":")[0] == MODEL for n in names):
        available = "\n    ".join(names) or "(none)"
        raise PreflightError(
            f"Model {MODEL!r} isn't pulled.\n"
            f"  Pull it with:  ollama pull {MODEL}\n"
            f"  Or set HARNESS_MODEL in .env to one you already have:\n    {available}"
        )

    # 3. Does it support tool calling? Stages 2+ are meaningless without this.
    try:
        capabilities = _ollama_get("/api/show", {"model": MODEL}).get("capabilities", [])
    except (urllib.error.URLError, OSError, KeyError):
        print(f"[preflight] WARNING: couldn't read capabilities for {MODEL!r}; continuing anyway.")
        return

    if "tools" not in capabilities:
        raise PreflightError(
            f"Model {MODEL!r} does not support tool calling (capabilities: {capabilities}).\n"
            f"  Stages 2+ need a tool-capable model. Try one of:\n"
            f"    ollama pull qwen3-coder\n"
            f"    ollama pull qwen2.5\n"
            f"    ollama pull llama3.1\n"
            f"  Then set HARNESS_MODEL in your .env accordingly."
        )
    print(f"[preflight] Model {MODEL!r} supports: {', '.join(capabilities)}")
