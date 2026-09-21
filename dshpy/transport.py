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
from typing import Iterator

from dshpy.cancel import NEVER, CancelToken


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


def _build_request(url: str, payload: dict, api_key: str | None,
                   headers: dict[str, str] | None) -> urllib.request.Request:
    request_headers = {"Content-Type": "application/json"}
    if api_key:
        request_headers["Authorization"] = f"Bearer {api_key}"
    request_headers.update(headers or {})
    return urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=request_headers, method="POST"
    )


def post_json(url: str, payload: dict, *, api_key: str | None = None,
              headers: dict[str, str] | None = None, timeout: float = 120.0) -> dict:
    """POST a JSON body, return the parsed JSON response."""
    try:
        with urllib.request.urlopen(
            _build_request(url, payload, api_key, headers), timeout=timeout
        ) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raise HttpError(exc.code, exc.read().decode(errors="replace"), url) from None


def post_stream(url: str, payload: dict, *, api_key: str | None = None,
                headers: dict[str, str] | None = None, timeout: float = 120.0,
                token: CancelToken | None = None,
                chunk_size: int = 1024) -> Iterator[bytes]:
    """POST and yield the response body in chunks, as it arrives.

    Streaming over `urllib` is just *not calling `.read()` with no argument*. `read(n)` returns
    as soon as n bytes are available rather than waiting for the whole body, and that is the
    entire mechanism — there is nothing else to streaming at the transport layer.

    The cancellation check sits at the top of the read loop, which is the cheapest correct
    place: it runs between socket reads, so a cancelled request stops at the next chunk
    boundary rather than after the model has finished generating. A token you never poll is a
    token that does nothing, and this loop is the main thing worth polling in the whole system.
    """
    token = token or NEVER
    try:
        resp = urllib.request.urlopen(
            _build_request(url, payload, api_key, headers), timeout=timeout
        )
    except urllib.error.HTTPError as exc:
        # Read the body before raising: for every LLM API it holds the actual reason.
        raise HttpError(exc.code, exc.read().decode(errors="replace"), url) from None

    with resp:
        while True:
            token.check()
            chunk = resp.read(chunk_size)
            if not chunk:
                return
            yield chunk
