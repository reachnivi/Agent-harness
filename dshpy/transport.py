"""Raw HTTP. No LLM SDK, anywhere in this package.

An LLM API is a POST with a JSON body. That's the whole thing. Every SDK you've used is
convenience on top of these thirty lines: auth headers, retries, JSON encode/decode, typed
objects, streaming reassembly. Useful conveniences — but none of them is the agent loop, and
confusing "I know the SDK" with "I know how this works" is the gap this package exists to close.

Stdlib `urllib` rather than `httpx`, deliberately: the only HTTP library in this venv is
`httpx2`, which is the `anthropic` SDK's own dependency. Reaching for it would tie the no-SDK
harness back to the SDK it is meant to do without. `urllib` adds nothing to install.

(DeepSeek's own reference adapter, `packages/llm/llm-deepseek`, is direct HTTP too. Raw HTTP is
faithful to the source pattern, not a shortcut around it.)
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class HttpError(RuntimeError):
    """A non-2xx response, carrying the body.

    `urllib` throws the response body away by default, which is a small tragedy: for every LLM
    API the body is where the useful message lives ("model not found", "invalid tool schema"),
    while the status code alone tells you almost nothing. Reading it before raising is the
    difference between a five-second fix and a twenty-minute one.
    """

    def __init__(self, status: int, body: str, url: str) -> None:
        super().__init__(f"HTTP {status} from {url}: {body[:500]}")
        self.status = status
        self.body = body
        self.url = url


def post_json(url: str, payload: dict, *, api_key: str | None = None,
              headers: dict[str, str] | None = None, timeout: float = 120.0) -> dict:
    """POST a JSON body, return the parsed JSON response."""
    body = json.dumps(payload).encode()
    request_headers = {"Content-Type": "application/json"}
    if api_key:
        request_headers["Authorization"] = f"Bearer {api_key}"
    request_headers.update(headers or {})

    req = urllib.request.Request(url, data=body, headers=request_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise HttpError(exc.code, exc.read().decode(errors="replace"), url) from None
