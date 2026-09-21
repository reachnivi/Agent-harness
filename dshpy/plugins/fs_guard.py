"""Filesystem invariants: path confinement and read-before-edit.

WHY THIS MOVED OUT OF permission.py

    Phase 5 put path confinement in the generic permission plugin, because that was the only
    place a guard existed. It belongs next to the filesystem: it is an invariant *about paths*,
    it needs to understand symlinks and roots, and it must apply to every fs consumer — not
    only to tools whose argument happens to be named `path`.

    It stays a **monotonic guard**, not a `tools/pre-execute` listener. Nothing mounted later
    and no profile ordering may switch it off. That distinction is the whole gate/guard split.

THE SYMLINK CASE, WHICH THE OBVIOUS CHECK MISSES

    The usual confinement check is:

        Path(p).resolve().is_relative_to(root)

    and it is correct, *because* `resolve()` follows symlinks. The bug is the version people
    write instead — comparing before resolving, or using `os.path.normpath`, which collapses
    `..` textually without touching the filesystem. Then a symlink inside the project pointing
    at `/etc` passes the check and reads whatever is behind it.

    So: resolve first, compare second, and test it with a real symlink. There is a test here
    that creates one, because a confinement check nobody tested against a symlink is a
    confinement check that probably does not hold.

READ-BEFORE-EDIT

    The guard tracks which files have been read, and with which version. A write or edit to a
    file that was never read, or that changed since it was read, is rejected.

    This is not bureaucracy. A model that read a file several tool calls ago is working from a
    stale copy, and a blind overwrite silently destroys whatever changed in between — very
    often the user's own edit. The rejection message tells the model to re-read, which is a
    recovery it can perform on its own.
"""

from __future__ import annotations

from pathlib import Path

from dshpy.services.fs import VersionConflict

name = "fs-guard"
inject = ["fs"]


class PathEscape(PermissionError):
    """A path resolved to somewhere outside the allowed root."""


def apply(ctx, config=None) -> None:
    config = config or {}
    root = Path(config.get("root", Path.cwd())).resolve()
    require_read_before_write = config.get("require_read_before_write", True)

    # path -> version at the time it was last read
    seen: dict[str, str] = {}

    def confine(path: str) -> Path:
        """Resolve, THEN compare. resolve() follows symlinks; normpath would not."""
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            resolved = candidate.resolve()
        except (OSError, RuntimeError) as exc:  # RuntimeError: symlink loop
            raise PathEscape(f"unusable path {path!r}: {exc}") from None

        if resolved != root and root not in resolved.parents:
            raise PathEscape(
                f"path {path!r} resolves to {resolved}, which is outside the project root "
                f"({root}). This is refused even for a symlink inside the project."
            )
        return resolved

    # --- reads: confine, and remember the version for the write check ---------------------

    def on_read(intent, next):
        intent.path = str(confine(intent.path))
        content, version = next()
        seen[intent.path] = version
        return content, version

    # --- mutations: confine, and require a current read -----------------------------------

    def enforce(intent) -> None:
        """Confine the path and impose the version recorded at read time.

        The intent is MUTATED and then delegated -- dsh's cooperative-listener idiom. That is
        what lets this guard require a version the caller never supplied, which is the whole
        mechanism of read-before-edit.
        """
        resolved = confine(intent.path)
        intent.path = str(resolved)

        if not require_read_before_write:
            return
        if not resolved.exists():
            return  # creating a new file needs no prior read

        recorded = seen.get(str(resolved))
        if recorded is None:
            raise VersionConflict(
                f"{resolved.name} has not been read in this session. Read it before writing, "
                f"so you are not overwriting changes you cannot see."
            )
        # Impose it even when the caller passed nothing: a caller that forgets to supply a
        # version must not thereby escape the check.
        intent.expected_version = intent.expected_version or recorded

    def on_write(intent, next):
        enforce(intent)
        new_version = next()
        seen[intent.path] = new_version
        return new_version

    def on_edit(intent, next):
        enforce(intent)
        new_version, made = next()
        seen[intent.path] = new_version
        return new_version, made

    ctx.on("fs/read-intent", on_read)
    ctx.on("fs/write-intent", on_write)
    ctx.on("fs/edit-intent", on_edit)

    # Expose the read set so a test or a UI can inspect what the agent has actually looked at.
    ctx.provide("fs_guard", type("Guard", (), {
        "seen": seen, "root": root, "confine": staticmethod(confine)})())
