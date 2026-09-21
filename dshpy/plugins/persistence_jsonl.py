"""JSONL persistence: one line per event, one file per session.

WHY JSONL AND NOT JSON

    A JSON array must be rewritten in full on every append and is unreadable until the closing
    bracket arrives. JSONL appends one line and stays valid at every instant, so a crash loses
    at most the line being written -- and a half-written final line is detectable and skippable
    rather than corrupting the file.

    For an append-only log this is not a close call. It is also why `tail -f` works on it,
    which matters more than it sounds when you are debugging an agent.

HOW IT HOOKS IN

    By listening to `session/event`. The sessions service does not know persistence exists, and
    unmounting this plugin stops the writing with no other change -- the difference between a
    plugin and a feature.
"""

from __future__ import annotations

import json
from pathlib import Path

from dshpy.services.persistence import PersistenceProvider
from dshpy.services.sessions import SessionEvent

name = "persistence-jsonl"
inject = ["persistence"]


class JsonlPersistence(PersistenceProvider):
    def __init__(self, root: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        """Reject a suspicious session id rather than sanitizing it.

        The first version took `Path(session_id).name` -- a basename, which is safe in that it
        cannot escape the directory. But `Path("../../escape").name` is `"escape"`, so the id
        is silently REWRITTEN and the caller reads and writes a different session than the one
        it asked for, with no error anywhere. Silent correction of a suspicious input is worse
        than refusal: it turns an attack into a bug report nobody can reproduce.
        """
        if (not session_id or "/" in session_id or "\\" in session_id
                or session_id in {".", ".."} or session_id.startswith(".")):
            raise ValueError(f"unusable session id {session_id!r}")
        return self.root / f"{session_id}.jsonl"

    def append(self, session_id: str, event: SessionEvent) -> None:
        with self._path(session_id).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event.to_json(), default=str) + "\n")

    def load(self, session_id: str) -> list[SessionEvent]:
        path = self._path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"no stored session {session_id!r} in {self.root}")

        events = []
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(SessionEvent.from_json(json.loads(line)))
            except json.JSONDecodeError:
                # A torn last line is the expected shape of a crash mid-write. Dropping it
                # loses one event; refusing to load would lose the entire session.
                print(f"[persistence] skipping unreadable line {lineno} of {path.name}")
        return events

    def write_all(self, session_id: str, events: list[SessionEvent]) -> None:
        path = self._path(session_id)
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text(
            "".join(json.dumps(e.to_json(), default=str) + "\n" for e in events),
            encoding="utf-8",
        )
        tmp.replace(path)  # atomic within the directory

    def list_sessions(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.jsonl"))


def apply(ctx, config=None) -> None:
    config = config or {}
    provider = JsonlPersistence(config.get("root", ".dshpy/sessions"))
    ctx.effect(ctx.persistence.register_provider(provider))

    # Write through on every event. The sessions service never learns this is happening.
    ctx.on("session/event", lambda event, session_id: provider.append(session_id, event))
