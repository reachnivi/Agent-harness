"""Ollama setup checks, shared by every harness in this repo.

Moved here from `harness/client.py`. Nothing in it is provider-specific in the *wire* sense — it
talks to Ollama's own native endpoints (`/api/version`, `/api/tags`, `/api/show`), which exist
regardless of whether you then speak the Anthropic or the OpenAI protocol at it.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class PreflightError(RuntimeError):
    """Setup is wrong in a way that would otherwise surface as a confusing bug later."""


def is_local(base_url: str) -> bool:
    return "localhost" in base_url or "127.0.0.1" in base_url


def _native_root(base_url: str) -> str:
    """Ollama's native API lives at the root, but the OpenAI-compatible path is under /v1.

    So a caller configured as `http://localhost:11434/v1` still needs `http://localhost:11434`
    for the checks below. Strip the suffix rather than making every caller remember to.
    """
    return base_url.rstrip("/").removesuffix("/v1")


def _get(base_url: str, path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        f"{_native_root(base_url)}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def preflight(base_url: str, model: str, *, quiet: bool = False) -> None:
    """Fail loudly *now* instead of mysteriously once tools are involved.

    The check that earns its keep is the tool-capability one. A model without tool support does
    not raise an error when you pass tools — it just answers in prose and never emits a tool
    call. That looks exactly like a bug in your agent loop, and you can lose an hour to it.
    """
    say = (lambda _: None) if quiet else print

    if not is_local(base_url):
        say(f"[preflight] Using a remote API at {base_url} — skipping local checks.")
        return

    # 1. Is the server even up?
    try:
        version = _get(base_url, "/api/version").get("version", "unknown")
    except (urllib.error.URLError, OSError) as exc:
        raise PreflightError(
            f"Can't reach Ollama at {_native_root(base_url)} ({exc}).\n"
            f"  Start it with:  ollama serve"
        ) from exc
    say(f"[preflight] Ollama {version} is up at {_native_root(base_url)}")

    # 2. Is the model actually pulled? Ollama tags carry a ':latest' suffix we must tolerate.
    names = [m["name"] for m in _get(base_url, "/api/tags").get("models", [])]
    if not any(n == model or n.split(":")[0] == model for n in names):
        available = "\n    ".join(names) or "(none)"
        raise PreflightError(
            f"Model {model!r} isn't pulled.\n"
            f"  Pull it with:  ollama pull {model}\n"
            f"  Or pick one you already have:\n    {available}"
        )

    # 3. Does it support tool calling? Everything past a plain chat is meaningless without this.
    try:
        capabilities = _get(base_url, "/api/show", {"model": model}).get("capabilities", [])
    except (urllib.error.URLError, OSError, KeyError):
        say(f"[preflight] WARNING: couldn't read capabilities for {model!r}; continuing anyway.")
        return

    if "tools" not in capabilities:
        raise PreflightError(
            f"Model {model!r} does not support tool calling (capabilities: {capabilities}).\n"
            f"  Try one of:\n"
            f"    ollama pull qwen3-coder\n"
            f"    ollama pull qwen2.5\n"
            f"    ollama pull llama3.1"
        )
    say(f"[preflight] Model {model!r} supports: {', '.join(capabilities)}")
