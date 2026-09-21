"""Retry failed model requests, as a plugin on the `llm/stream` waterfall.

dsh ships this as its own package (`llm-retry`) and splits it in a way worth copying exactly:

    the POLICY lives on the adapter's config; the EXECUTOR is this plugin.

Why the split? Because how many times to retry a 429 is a property of the *provider* (its rate
limits, its reliability, your contract with it), while *whether retrying happens at all* is a
property of the deployment. Putting both in the adapter means every adapter reimplements
backoff; putting both here means one policy for providers with wildly different behavior.

WHAT MUST NOT BE RETRIED

    Only transient failures: 429, 5xx, and transport errors. A 400 means the request is wrong
    and will be exactly as wrong next time -- retrying it burns quota to produce the identical
    error, three times slower. A cancelled request must not be retried either: the user asked
    for it to stop, and "stop" is not a transient failure.

WHERE THE RETRY HAPPENS

    At the stream boundary, before any chunk has been yielded downstream. Once a consumer has
    seen half a reply, replacing the stream would duplicate text it already rendered. So this
    buffers the first chunk to learn whether the attempt got off the ground, and stops being
    able to retry the moment real content has flowed. dsh describes the same constraint as
    retrying "at durable agent-step boundaries".
"""

from __future__ import annotations

import time
from typing import Iterator

from dshpy.cancel import Cancelled
from dshpy.services.llm import Finish, GenerateOptions, LlmAdapter, StreamChunk
from dshpy.transport import HttpError

name = "llm-retry"
inject = ["llm"]

DEFAULT_POLICY = {"attempts": 3, "base_delay": 0.5, "max_delay": 8.0}
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, Cancelled):
        return False  # the user said stop; that is not a transient failure
    if isinstance(exc, HttpError):
        return exc.status in RETRYABLE_STATUS
    return isinstance(exc, (OSError, TimeoutError))  # transport-level


def apply(ctx, config=None) -> None:
    config = config or {}
    sleep = config.get("sleep", time.sleep)  # injectable so tests don't actually wait

    def retrying(options: GenerateOptions, adapter: LlmAdapter, next) -> Iterator[StreamChunk]:
        policy = {**DEFAULT_POLICY, **(adapter.retry_policy or {}), **config.get("policy", {})}
        attempts = max(1, int(policy["attempts"]))

        def run() -> Iterator[StreamChunk]:
            last_error: BaseException | None = None
            for attempt in range(attempts):
                try:
                    stream = next()
                    # Pull the first chunk here so a transport failure surfaces NOW, while
                    # retrying is still safe. A generator that has yielded nothing has shown
                    # the consumer nothing, so replacing it is invisible.
                    first = _first(stream)
                    if first is _EMPTY:
                        return
                    yield from _replay(first, stream)
                    return
                except Exception as exc:  # noqa: BLE001 - classified by is_retryable
                    last_error = exc
                    if attempt == attempts - 1 or not is_retryable(exc):
                        break
                    delay = min(policy["base_delay"] * (2 ** attempt), policy["max_delay"])
                    ctx.emit("llm/retry", attempt + 1, attempts, exc, delay)
                    sleep(delay)

            # Out of attempts. Normalize to a terminal finish rather than raising, so the
            # consumer sees one failure shape (the same rule the service applies).
            reason = "aborted" if isinstance(last_error, Cancelled) else "error"
            yield Finish(reason=reason,
                         error=f"{type(last_error).__name__}: {last_error}")

        return run()

    ctx.on("llm/stream", retrying)


_EMPTY = object()


def _first(stream: Iterator[StreamChunk]):
    for chunk in stream:
        return chunk
    return _EMPTY


def _replay(first, rest: Iterator[StreamChunk]) -> Iterator[StreamChunk]:
    yield first
    yield from rest
