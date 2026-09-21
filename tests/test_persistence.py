"""Tests for persistence, resume, fork and compaction.

These are the payoff tests for a decision made back in phase 3: separating the append-only log
from the derived model view. Each capability here is a few dozen lines *because* of that split,
and each test says which property it depends on.
"""

from __future__ import annotations

import json

import pytest

from dshpy.core.context import Runtime
from dshpy.plugins import compaction_basic, persistence_jsonl
from dshpy.services import (
    compaction as compaction_service,
    llm as llm_service,
    persistence as persistence_service,
    sessions as sessions_service,
)
from dshpy.services.llm import Completion, Finish, LlmAdapter, TextDelta


class FakeSummarizerAdapter(LlmAdapter):
    provider = "fake"

    def __init__(self, summary="SUMMARY: user wanted X; edited a.py; tests pass."):
        self.summary = summary
        self.prompts = []

    def stream(self, options):
        self.prompts.append(options.messages[-1].content)
        yield TextDelta(index=0, text=self.summary)
        yield Finish(reason="stop")


@pytest.fixture
def rt(tmp_path):
    runtime = Runtime()
    runtime.mount(sessions_service)
    runtime.mount(persistence_service)
    runtime.mount(persistence_jsonl, {"root": str(tmp_path / "sessions")})
    return runtime


def conversation(sessions, turns=2, prefix="q"):
    """Append `turns` complete turns, each ending at a turn/end boundary."""
    for i in range(turns):
        sessions.append("turn/start")
        sessions.append("user/message", content=f"{prefix}{i}")
        sessions.append("assistant/message", content=f"a{i}", tool_calls=[])
        sessions.append("turn/end")


# --- persistence ------------------------------------------------------------------------------


def test_every_event_is_written_as_it_happens(rt, tmp_path):
    sessions = rt.services["sessions"]
    sessions.append("user/message", content="hello")

    path = tmp_path / "sessions" / f"{sessions.session_id}.jsonl"
    assert path.exists()
    [line] = path.read_text().strip().splitlines()
    assert json.loads(line)["data"]["content"] == "hello"


def test_a_round_trip_preserves_every_event_exactly(rt):
    sessions = rt.services["sessions"]
    conversation(sessions, turns=3)
    before = [e.to_json() for e in sessions.events()]

    loaded = rt.services["persistence"].load(sessions.session_id)
    assert [e.to_json() for e in loaded] == before


def test_a_session_survives_a_process_boundary(tmp_path):
    """Write in one runtime, read in a completely separate one."""
    root = str(tmp_path / "sessions")

    first = Runtime()
    first.mount(sessions_service)
    first.mount(persistence_service)
    first.mount(persistence_jsonl, {"root": root})
    conversation(first.services["sessions"], turns=2)
    session_id = first.services["sessions"].session_id
    expected = [m.content for m in first.services["sessions"].derive_messages()]
    first.dispose()

    second = Runtime()
    second.mount(sessions_service)
    second.mount(persistence_service)
    second.mount(persistence_jsonl, {"root": root})
    count = second.services["persistence"].resume(session_id)

    assert count > 0
    assert [m.content for m in second.services["sessions"].derive_messages()] == expected


def test_resuming_does_not_rewrite_the_log(rt, tmp_path):
    """restore() must not re-emit session/event, or resume would duplicate the history."""
    sessions = rt.services["sessions"]
    conversation(sessions, turns=2)
    session_id = sessions.session_id
    path = tmp_path / "sessions" / f"{session_id}.jsonl"
    lines_before = len(path.read_text().strip().splitlines())

    rt.services["persistence"].resume(session_id)
    assert len(path.read_text().strip().splitlines()) == lines_before


def test_a_torn_final_line_loses_one_event_not_the_session(rt, tmp_path):
    """The expected shape of a crash mid-write."""
    sessions = rt.services["sessions"]
    conversation(sessions, turns=2)
    path = tmp_path / "sessions" / f"{sessions.session_id}.jsonl"

    with path.open("a") as fh:
        fh.write('{"kind": "user/mess')  # torn

    loaded = rt.services["persistence"].load(sessions.session_id)
    assert len(loaded) == len(sessions.events()), "a torn line should not cost a whole session"


@pytest.mark.parametrize("bad", ["../../escape", "a/b", "..", ".", "", ".hidden"])
def test_a_suspicious_session_id_is_refused_not_sanitized(rt, bad):
    """Sanitizing silently rewrites the id, so the caller reads a different session than it
    asked for and nothing reports it. Refusal is the correct response."""
    with pytest.raises(ValueError, match="unusable session id"):
        rt.services["persistence"].provider._path(bad)


def test_an_ordinary_session_id_still_works(rt):
    assert rt.services["persistence"].provider._path("abc123").name == "abc123.jsonl"


def test_list_sessions(rt):
    rt.services["sessions"].append("user/message", content="x")
    assert rt.services["sessions"].session_id in rt.services["persistence"].list_sessions()


# --- fork -------------------------------------------------------------------------------------


def test_fork_copies_history_up_to_a_turn_boundary(rt):
    sessions = rt.services["sessions"]
    conversation(sessions, turns=3)
    boundaries = sessions.turn_boundaries()

    child = rt.services["persistence"].fork(sessions.session_id, at_seq=boundaries[0])
    child_events = rt.services["persistence"].load(child)

    assert max(e.seq for e in child_events) == boundaries[0]
    assert len(child_events) < len(sessions.events())


def test_a_fork_diverges_without_touching_the_parent(rt):
    sessions = rt.services["sessions"]
    conversation(sessions, turns=2)
    parent_id = sessions.session_id
    parent_before = [e.to_json() for e in rt.services["persistence"].load(parent_id)]

    child = rt.services["persistence"].fork(parent_id, at_seq=sessions.turn_boundaries()[0])
    rt.services["persistence"].resume(child)
    rt.services["sessions"].append("user/message", content="a different direction")

    assert [e.to_json() for e in rt.services["persistence"].load(parent_id)] == parent_before


def test_fork_defaults_to_the_last_completed_turn(rt):
    sessions = rt.services["sessions"]
    conversation(sessions, turns=2)
    child = rt.services["persistence"].fork(sessions.session_id)
    assert max(e.seq for e in rt.services["persistence"].load(child)) == \
        sessions.turn_boundaries()[-1]


def test_forking_mid_turn_is_refused(rt):
    """A fork mid-turn leaves a tool_use with no result, which both protocols reject."""
    sessions = rt.services["sessions"]
    conversation(sessions, turns=2)
    mid = sessions.turn_boundaries()[0] - 1

    with pytest.raises(ValueError, match="not a turn boundary"):
        rt.services["persistence"].fork(sessions.session_id, at_seq=mid)


def test_forking_a_session_with_no_completed_turn_is_refused(rt):
    rt.services["sessions"].append("user/message", content="just started")
    with pytest.raises(ValueError, match="no completed turn"):
        rt.services["persistence"].fork(rt.services["sessions"].session_id)


# --- compaction ---------------------------------------------------------------------------------


@pytest.fixture
def compact_rt(tmp_path):
    runtime = Runtime()
    runtime.mount(sessions_service)
    runtime.mount(llm_service)
    runtime.mount(compaction_service)
    adapter = FakeSummarizerAdapter()
    runtime.services["llm"].register_adapter(["fake"], adapter)
    runtime.mount(compaction_basic, {"model": "m", "provider": "fake"})
    return runtime, adapter


def test_compaction_shortens_the_model_view_but_not_the_log(compact_rt):
    """The headline property: the projection changes, the record does not."""
    rt, _ = compact_rt
    sessions = rt.services["sessions"]
    sessions.append("system/message", content="be helpful")
    conversation(sessions, turns=4)

    events_before = len(sessions.events())
    messages_before = len(sessions.derive_messages())

    result = rt.services["compaction"].compact(keep_last_turns=1)

    assert result is not None
    assert len(sessions.derive_messages()) < messages_before, "the model view did not shrink"
    assert len(sessions.events()) > events_before, "events were deleted rather than shadowed"


def test_the_summary_reaches_the_model_as_a_user_message(compact_rt):
    rt, _ = compact_rt
    conversation(rt.services["sessions"], turns=4)
    rt.services["compaction"].compact(keep_last_turns=1)

    messages = rt.services["sessions"].derive_messages()
    assert any(m.role == "user" and "summarized" in m.content for m in messages)


def test_the_system_prompt_is_never_shadowed(compact_rt):
    """Summarizing away an instruction silently changes behavior for the rest of the session."""
    rt, _ = compact_rt
    sessions = rt.services["sessions"]
    sessions.append("system/message", content="NEVER touch config.py")
    conversation(sessions, turns=4)

    rt.services["compaction"].compact(keep_last_turns=1)

    system = [m for m in sessions.derive_messages() if m.role == "system"]
    assert system and system[0].content == "NEVER touch config.py"


def test_compaction_events_are_log_only(compact_rt):
    """The bookkeeping must not reach the model."""
    rt, _ = compact_rt
    conversation(rt.services["sessions"], turns=4)
    rt.services["compaction"].compact(keep_last_turns=1)

    kinds = {e.kind for e in rt.services["sessions"].events()}
    assert {"compaction/start", "compaction/summary", "compaction/end"} <= kinds

    rendered = " ".join(m.content for m in rt.services["sessions"].derive_messages())
    assert "compaction/" not in rendered


def test_the_lock_is_released_last(compact_rt):
    """A crash mid-operation must leave a detectable orphan, not a false 'finished'."""
    rt, _ = compact_rt
    conversation(rt.services["sessions"], turns=4)
    rt.services["compaction"].compact(keep_last_turns=1)

    kinds = [e.kind for e in rt.services["sessions"].events()]
    assert kinds.index("compaction/start") < kinds.index("compaction/summary")
    assert kinds.index("compaction/summary") < kinds.index("compaction/end")
    assert not rt.services["compaction"].is_locked()


def test_a_failed_summarization_releases_the_lock_and_records_the_error(compact_rt):
    rt, adapter = compact_rt
    conversation(rt.services["sessions"], turns=4)

    def boom(options):
        raise RuntimeError("model unavailable")
        yield  # pragma: no cover

    adapter.stream = boom

    with pytest.raises(RuntimeError):
        rt.services["compaction"].compact(keep_last_turns=1)

    assert not rt.services["compaction"].is_locked(), "a failure left the lock held"
    end = [e for e in rt.services["sessions"].events() if e.kind == "compaction/end"][-1]
    assert end.data.get("error")


def test_an_orphaned_lock_is_detected_and_clearable(compact_rt):
    rt, _ = compact_rt
    sessions = rt.services["sessions"]
    conversation(sessions, turns=4)
    sessions.append("compaction/start", reason="crashed before finishing")

    assert rt.services["compaction"].is_locked()
    with pytest.raises(RuntimeError, match="orphaned lock"):
        rt.services["compaction"].compact()

    assert rt.services["compaction"].clear_orphaned_lock()
    assert not rt.services["compaction"].is_locked()


def test_compaction_declines_when_there_is_nothing_worth_doing(compact_rt):
    """A speculative auto-compaction call must not be an error."""
    rt, _ = compact_rt
    conversation(rt.services["sessions"], turns=1)
    assert rt.services["compaction"].compact(keep_last_turns=2) is None


def test_the_summarizer_is_given_the_real_transcript(compact_rt):
    rt, adapter = compact_rt
    sessions = rt.services["sessions"]
    conversation(sessions, turns=4, prefix="do the thing ")
    rt.services["compaction"].compact(keep_last_turns=1)

    assert adapter.prompts, "the backend never called the model"
    assert "do the thing 0" in adapter.prompts[0]


def test_a_compacted_session_is_still_forkable_at_an_earlier_turn(tmp_path):
    """The reason nothing is deleted: full fidelity survives a shortened projection."""
    rt = Runtime()
    rt.mount(sessions_service)
    rt.mount(llm_service)
    rt.mount(compaction_service)
    rt.mount(persistence_service)
    rt.mount(persistence_jsonl, {"root": str(tmp_path / "s")})
    rt.services["llm"].register_adapter(["fake"], FakeSummarizerAdapter())
    rt.mount(compaction_basic, {"model": "m", "provider": "fake"})

    sessions = rt.services["sessions"]
    conversation(sessions, turns=4)
    early = sessions.turn_boundaries()[0]
    rt.services["compaction"].compact(keep_last_turns=1)

    child = rt.services["persistence"].fork(sessions.session_id, at_seq=early)
    child_events = rt.services["persistence"].load(child)
    assert [e.kind for e in child_events].count("user/message") == 1


def test_auto_compaction_fires_on_a_turn_boundary(tmp_path):
    rt = Runtime()
    rt.mount(sessions_service)
    rt.mount(llm_service)
    rt.mount(compaction_service)
    rt.services["llm"].register_adapter(["fake"], FakeSummarizerAdapter())
    rt.mount(compaction_basic, {"model": "m", "provider": "fake",
                                "threshold_tokens": 1, "keep_last_turns": 1})

    sessions = rt.services["sessions"]
    conversation(sessions, turns=4)
    rt.events.emit("turn/end", "done")

    assert any(e.kind == "compaction/summary" for e in sessions.events())
