"""A Server-Sent Events parser, written by hand.

SSE is what every streaming LLM API speaks, and it is a genuinely small format — which is worth
knowing, because "streaming" sounds like the part you need a library for and it isn't.

THE FORMAT, IN FULL

    Text, UTF-8, line-oriented. A blank line dispatches the accumulated event:

        event: content_block_delta          <- optional name; OpenAI omits it entirely
        data: {"type":"text_delta",...}     <- may repeat; lines join with "\\n"
                                            <- blank line: DISPATCH
        data: [DONE]                        <- OpenAI's end sentinel (not part of the spec)

    Lines beginning `:` are comments, used as keep-alives. `id:` and `retry:` exist and no LLM
    API uses them.

THE TWO THINGS THAT ACTUALLY BREAK PEOPLE

    1. **Frames split across reads.** A chunk boundary from the socket has nothing to do with a
       line boundary; `data: {"te` and `xt":"hi"}\\n\\n` routinely arrive as separate reads. You
       must buffer across reads, and you must not assume one read is one frame.

    2. **Multi-byte characters split across reads.** A single emoji or CJK character is several
       UTF-8 bytes, and those bytes can land in different reads. Naive `chunk.decode()` raises
       `UnicodeDecodeError` on perfectly valid input. The fix is an incremental decoder that
       holds the partial character until its remaining bytes arrive — which is why this parser
       takes *bytes* and owns the decoding rather than accepting `str`.

    Both are tested in tests/test_sse.py. Both are the kind of bug that only shows up against a
    real server, under load, in the reply that happened to contain an emoji.
"""

from __future__ import annotations

import codecs
from dataclasses import dataclass, field
from typing import Iterable, Iterator


@dataclass
class SSEEvent:
    """One dispatched event. `data` is the joined data lines, `event` the optional name."""

    data: str
    event: str | None = None
    raw_lines: list[str] = field(default_factory=list)


def parse_sse(chunks: Iterable[bytes], *, done_sentinel: str = "[DONE]") -> Iterator[SSEEvent]:
    """Turn a stream of arbitrarily-sized byte chunks into dispatched SSE events.

    Stops when `done_sentinel` arrives as a data payload (OpenAI's convention; the Anthropic
    dialect never sends one and simply ends the stream instead).
    """
    decoder = codecs.getincrementaldecoder("utf-8")()  # holds partial multi-byte characters
    buffer = ""
    event_name: str | None = None
    data_lines: list[str] = []
    raw_lines: list[str] = []

    for chunk in chunks:
        buffer += decoder.decode(chunk)

        # Keep the trailing fragment: a frame split across reads leaves an incomplete last line.
        *lines, buffer = buffer.split("\n")

        for line in lines:
            line = line.rstrip("\r")  # servers differ on CRLF vs LF
            raw_lines.append(line)

            if line == "":  # blank line: dispatch whatever has accumulated
                if data_lines:
                    data = "\n".join(data_lines)
                    if data.strip() == done_sentinel:
                        return
                    yield SSEEvent(data=data, event=event_name, raw_lines=list(raw_lines))
                event_name, data_lines, raw_lines = None, [], []
                continue

            if line.startswith(":"):  # comment / keep-alive
                continue

            field_name, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value  # one optional leading space

            if field_name == "data":
                data_lines.append(value)
            elif field_name == "event":
                event_name = value
            # `id:` and `retry:` are spec fields no LLM API uses; ignore them.

    # A stream that ends without a trailing blank line still has a final event to dispatch.
    buffer += decoder.decode(b"", final=True)
    if buffer.strip():
        field_name, _, value = buffer.rstrip("\r").partition(":")
        if field_name == "data":
            data_lines.append(value[1:] if value.startswith(" ") else value)
    if data_lines:
        data = "\n".join(data_lines)
        if data.strip() != done_sentinel:
            yield SSEEvent(data=data, event=event_name, raw_lines=list(raw_lines))
