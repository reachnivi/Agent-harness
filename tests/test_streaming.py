"""Tests for the streaming service layer: the llm/stream waterfall, retry, metering, cancel.

These exercise the seam rather than either adapter, so they use a fake adapter that emits
scripted chunks. That's the point of the seam: the retry and metering plugins were written
without knowing which provider is underneath, and these tests prove it by using neither.
"""

from __future__ import annotations

import pytest

from dshpy.cancel import CancelToken, Cancelled
from dshpy.core.context import Runtime
from dshpy.plugins import llm_retry, token_meter
from dshpy.services import llm as llm_service
from dshpy.services.llm import (
    BlockStart,
    Finish,
    GenerateOptions,
    LlmAdapter,
    Message,
    TextDelta,
    ToolCallDelta,
    Usage,
    UsageChunk,
)
from dshpy.transport import HttpError


class ScriptedAdapter(LlmAdapter):
    """Emits canned chunks, or raises a canned sequence of errors first."""

    provider = "scripted"

    def __init__(self, chunks=None, errors=(), retry_policy=None):
        self.chunks = chunks or [TextDelta(index=0, text="ok"), Finish(reason="stop")]
        self.errors = list(errors)
        self.retry_policy = retry_policy
        self.attempts = 0

    def stream(self, options):
        self.attempts += 1
        if self.errors:
            raise self.errors.pop(0)
        yield from self.chunks


def runtime(adapter, *plugins):
    rt = Runtime()
    rt.mount(llm_service)
    rt.services["llm"].register_adapter(["scripted"], adapter)
    for plugin_mod, cfg in plugins:
        rt.mount(plugin_mod, cfg)
    return rt


def opts(**kw):
    return GenerateOptions(messages=[Message("user", "hi")], model="m", **kw)


# --- the service --------------------------------------------------------------------------


def test_generate_is_a_fold_over_the_stream():
    rt = runtime(ScriptedAdapter())
    assert rt.services["llm"].generate([Message("user", "hi")], tools=[], model="m",
                                       provider="scripted").text == "ok"


def test_an_adapter_that_raises_becomes_a_terminal_finish_not_an_exception():
    """dsh's rule: consumers handle ONE failure shape, so the runtime normalizes a throw."""
    rt = runtime(ScriptedAdapter(errors=[RuntimeError("socket died")]))
    completion = rt.services["llm"].generate([Message("user", "hi")], tools=[], model="m",
                                             provider="scripted")
    assert completion.finish_reason == "error"
    assert "socket died" in (completion.raw or "") or completion.text == ""


def test_a_partial_response_survives_the_failure_that_ends_it():
    """A bare exception would throw away text the model already produced."""

    class DiesHalfway(LlmAdapter):
        provider = "scripted"

        def stream(self, options):
            yield BlockStart(index=0, block_type="text")
            yield TextDelta(index=0, text="I got this far")
            raise ConnectionError("boom")

    rt = runtime(DiesHalfway())
    completion = rt.services["llm"].generate([Message("user", "hi")], tools=[], model="m",
                                             provider="scripted")
    assert completion.text == "I got this far"
    assert completion.finish_reason == "error"


def test_the_stream_waterfall_lets_a_plugin_wrap_without_the_service_knowing():
    rt = runtime(ScriptedAdapter())
    seen = []

    def spy(options, adapter, next):
        def wrapped():
            for chunk in next():
                seen.append(type(chunk).__name__)
                yield chunk
        return wrapped()

    rt.mount(type("m", (), {"name": "spy", "inject": ["llm"],
                            "apply": staticmethod(lambda ctx, config=None:
                                                  ctx.on("llm/stream", spy))}))
    rt.services["llm"].generate([Message("user", "hi")], tools=[], model="m",
                                provider="scripted")
    assert "TextDelta" in seen and "Finish" in seen


# --- retry ------------------------------------------------------------------------------------


def test_retry_recovers_from_a_transient_failure():
    adapter = ScriptedAdapter(errors=[HttpError(503, "overloaded", "u")])
    rt = runtime(adapter, (llm_retry, {"sleep": lambda _: None}))

    completion = rt.services["llm"].generate([Message("user", "hi")], tools=[], model="m",
                                             provider="scripted")
    assert completion.text == "ok"
    assert adapter.attempts == 2


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_retryable_statuses(status):
    adapter = ScriptedAdapter(errors=[HttpError(status, "x", "u")])
    rt = runtime(adapter, (llm_retry, {"sleep": lambda _: None}))
    rt.services["llm"].generate([Message("user", "hi")], tools=[], model="m",
                                provider="scripted")
    assert adapter.attempts == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_a_4xx_is_never_retried(status):
    """It will be exactly as wrong next time; retrying burns quota to produce it three times."""
    adapter = ScriptedAdapter(errors=[HttpError(status, "bad request", "u")] * 3)
    rt = runtime(adapter, (llm_retry, {"sleep": lambda _: None}))

    completion = rt.services["llm"].generate([Message("user", "hi")], tools=[], model="m",
                                             provider="scripted")
    assert adapter.attempts == 1
    assert completion.finish_reason == "error"


def test_cancellation_is_not_retried():
    """The user said stop. That is not a transient failure."""
    adapter = ScriptedAdapter(errors=[Cancelled("user cancelled")] * 3)
    rt = runtime(adapter, (llm_retry, {"sleep": lambda _: None}))

    completion = rt.services["llm"].generate([Message("user", "hi")], tools=[], model="m",
                                             provider="scripted")
    assert adapter.attempts == 1
    assert completion.finish_reason == "aborted"


def test_retry_gives_up_after_the_configured_attempts():
    adapter = ScriptedAdapter(errors=[HttpError(503, "x", "u")] * 10)
    rt = runtime(adapter, (llm_retry, {"sleep": lambda _: None, "policy": {"attempts": 3}}))

    completion = rt.services["llm"].generate([Message("user", "hi")], tools=[], model="m",
                                             provider="scripted")
    assert adapter.attempts == 3
    assert completion.finish_reason == "error"


def test_backoff_delays_grow_exponentially():
    delays = []
    adapter = ScriptedAdapter(errors=[HttpError(503, "x", "u")] * 10)
    rt = runtime(adapter, (llm_retry, {"sleep": delays.append,
                                       "policy": {"attempts": 4, "base_delay": 1.0}}))
    rt.services["llm"].generate([Message("user", "hi")], tools=[], model="m",
                                provider="scripted")
    assert delays == [1.0, 2.0, 4.0]


def test_the_policy_can_come_from_the_adapter():
    """dsh's split: policy is a property of the provider, the executor is this plugin."""
    adapter = ScriptedAdapter(errors=[HttpError(503, "x", "u")] * 10,
                              retry_policy={"attempts": 2})
    rt = runtime(adapter, (llm_retry, {"sleep": lambda _: None}))
    rt.services["llm"].generate([Message("user", "hi")], tools=[], model="m",
                                provider="scripted")
    assert adapter.attempts == 2


# --- token metering ----------------------------------------------------------------------------


def test_the_meter_accumulates_usage_across_calls():
    adapter = ScriptedAdapter(chunks=[
        TextDelta(index=0, text="hi"),
        UsageChunk(usage=Usage(input_tokens=10, output_tokens=5)),
        Finish(reason="stop"),
    ])
    rt = runtime(adapter, (token_meter, {}))

    for _ in range(3):
        rt.services["llm"].generate([Message("user", "hi")], tools=[], model="m",
                                    provider="scripted")

    meter = rt.services["tokens"]
    assert meter.calls == 3
    assert meter.total.input_tokens == 30
    assert meter.total.total == 45


def test_metering_does_not_consume_the_stream():
    """The obvious way to write this -- draining to count -- breaks streaming for everyone."""
    adapter = ScriptedAdapter(chunks=[
        TextDelta(index=0, text="still"),
        TextDelta(index=0, text=" here"),
        UsageChunk(usage=Usage(1, 1)),
        Finish(reason="stop"),
    ])
    rt = runtime(adapter, (token_meter, {}))
    completion = rt.services["llm"].generate([Message("user", "hi")], tools=[], model="m",
                                             provider="scripted")
    assert completion.text == "still here"


def test_retry_and_metering_compose():
    adapter = ScriptedAdapter(
        chunks=[UsageChunk(usage=Usage(7, 3)), Finish(reason="stop")],
        errors=[HttpError(503, "x", "u")],
    )
    rt = runtime(adapter, (llm_retry, {"sleep": lambda _: None}), (token_meter, {}))
    rt.services["llm"].generate([Message("user", "hi")], tools=[], model="m",
                                provider="scripted")
    assert adapter.attempts == 2
    assert rt.services["tokens"].total.total == 10, "usage counted once, after the retry"


# --- cancellation ---------------------------------------------------------------------------


def test_a_cancelled_token_stops_a_streaming_adapter():
    token = CancelToken()

    class Endless(LlmAdapter):
        provider = "scripted"

        def stream(self, options):
            for i in range(10_000):
                options.token.check()
                if i == 3:
                    token.cancel("user pressed ctrl-c")
                yield TextDelta(index=0, text="x")

    rt = runtime(Endless())
    completion = rt.services["llm"].generate([Message("user", "hi")], tools=[], model="m",
                                             provider="scripted", token=token)
    assert completion.finish_reason == "aborted"
    assert completion.text == "xxxx", "should stop at the check after cancellation"


def test_a_deadline_token_sets_itself_without_a_timer_thread():
    token = CancelToken(timeout=0.0)
    assert token.is_set()
    with pytest.raises(Cancelled):
        token.check()


def test_a_child_token_is_cancelled_by_its_parent():
    parent = CancelToken()
    child = parent.child(timeout=60)
    assert not child.is_set()
    parent.cancel("outer stop")
    assert child.is_set()
    assert child.reason == "outer stop"


def test_cancelling_a_child_does_not_cancel_its_parent():
    parent = CancelToken()
    child = parent.child()
    child.cancel()
    assert child.is_set() and not parent.is_set()
