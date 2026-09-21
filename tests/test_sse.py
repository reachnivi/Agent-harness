"""Tests for the hand-written SSE parser.

The two tests that matter here are `split_across_reads` and `multibyte_split_across_reads`.
Both describe input that is completely valid and that a naive parser mangles, and both only
show up against a real server — one under load, one when the reply happens to contain an emoji.
Canned byte streams make them deterministic.
"""

from __future__ import annotations

from dshpy.sse import parse_sse


def events(*chunks: bytes, **kwargs):
    return list(parse_sse(iter(chunks), **kwargs))


# --- the basics -----------------------------------------------------------------------------


def test_one_event():
    [event] = events(b'data: {"a":1}\n\n')
    assert event.data == '{"a":1}'
    assert event.event is None


def test_named_events_carry_their_name():
    """Anthropic's dialect; OpenAI's omits the name entirely."""
    [event] = events(b"event: content_block_delta\ndata: {}\n\n")
    assert event.event == "content_block_delta"


def test_several_events_in_one_read():
    got = events(b'data: 1\n\ndata: 2\n\ndata: 3\n\n')
    assert [e.data for e in got] == ["1", "2", "3"]


def test_multiple_data_lines_join_with_newline():
    [event] = events(b"data: line one\ndata: line two\n\n")
    assert event.data == "line one\nline two"


def test_done_sentinel_ends_the_stream():
    got = events(b'data: 1\n\ndata: [DONE]\n\ndata: 2\n\n')
    assert [e.data for e in got] == ["1"], "nothing after [DONE] should be yielded"


def test_a_different_dialect_can_opt_out_of_the_sentinel():
    """Anthropic never sends [DONE], and '[DONE]' could legitimately be model output."""
    got = events(b'data: [DONE]\n\n', done_sentinel="\x00never\x00")
    assert [e.data for e in got] == ["[DONE]"]


def test_comments_and_keepalives_are_ignored():
    got = events(b": keep-alive\n\ndata: real\n\n")
    assert [e.data for e in got] == ["real"]


def test_crlf_line_endings():
    [event] = events(b'data: {"a":1}\r\n\r\n')
    assert event.data == '{"a":1}'


def test_optional_leading_space_is_stripped_only_once():
    [event] = events(b"data:  two spaces\n\n")
    assert event.data == " two spaces"


def test_a_final_event_without_a_trailing_blank_line_is_still_dispatched():
    """Servers do close streams without the final delimiter; dropping that event loses text."""
    got = events(b"data: last\n")
    assert [e.data for e in got] == ["last"]


# --- the two that actually break people -----------------------------------------------------


def test_frame_split_across_reads():
    """A socket read boundary has nothing to do with a line boundary."""
    got = events(b'data: {"te', b'xt":"hi"}', b"\n\n")
    assert [e.data for e in got] == ['{"text":"hi"}']


def test_event_split_between_name_and_data():
    got = events(b"event: content_bl", b"ock_delta\nda", b"ta: {}\n\n")
    assert got[0].event == "content_block_delta"


def test_multibyte_character_split_across_reads():
    """A naive chunk.decode() raises UnicodeDecodeError on this, and the input is valid.

    'é' is two UTF-8 bytes; an emoji is four. A read can land between them.
    """
    payload = 'data: {"text":"café \U0001f600"}\n\n'.encode()
    # Split in the middle of the 4-byte emoji.
    split = payload.rfind(b"\xf0\x9f") + 2
    got = events(payload[:split], payload[split:])
    assert [e.data for e in got] == ['{"text":"café 😀"}']


def test_byte_at_a_time_gives_the_same_result_as_one_read():
    """The strongest form of the buffering test: every possible split point at once."""
    payload = 'event: x\ndata: {"t":"héllo 🌍"}\n\ndata: {"t":"2"}\n\n'.encode()
    one_read = [(e.event, e.data) for e in events(payload)]
    byte_wise = [(e.event, e.data) for e in events(*[bytes([b]) for b in payload])]
    assert byte_wise == one_read


def test_empty_stream_yields_nothing():
    assert events() == []
    assert events(b"") == []
