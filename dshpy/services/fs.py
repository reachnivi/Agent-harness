"""`ctx.fs` — the filesystem seam.

dsh splits this into `dsh-fs` (the contract), `fs-local` / `fs-sandbox` (providers), and
`tool-fs` (the model-facing tools). Three packages for "read a file" looks like ceremony until
you want the same tools pointed at a container, a remote host, or a fake — at which point the
split is the only thing that makes it possible.

THE VERSION GUARD — the interesting part of this interface

    `write_text` and `edit` take an optional `expected_version`. If the file has changed since
    the caller read it, the write is rejected.

    dsh makes this **optional at the provider level** so a backend works without policy, and
    lets a *caller* supply the guard. That separation is what turns a plain filesystem into
    read-before-edit enforcement (`plugins/fs_guard.py`) without the filesystem knowing what a
    policy is.

    Why it matters for an agent specifically: a model that read a file 40 seconds and three
    tool calls ago is working from a stale copy. Overwriting blind loses whatever changed in
    between — often the user's own edit, or the model's own earlier write it has forgotten.
    A version check turns silent data loss into an error the model can read and recover from.

    "Version" here is `(mtime_ns, size)` rather than a hash: cheap to take on every read, and
    good enough to catch the realistic case. A hash would be stronger and would cost a full
    read on every check; noted rather than pretended away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

name = "fs"
inject: list[str] = []


class FsError(RuntimeError):
    """Any filesystem failure the model should be told about rather than crash on."""


class VersionConflict(FsError):
    """The file changed since the caller last read it."""


@dataclass(frozen=True)
class FileInfo:
    path: str
    size: int
    version: str          # opaque to callers; compare, never parse
    is_dir: bool = False


class FsProvider:
    """What a filesystem backend implements. `fs_local` is the reference."""

    def read_text(self, path: str, *, max_bytes: int | None = None) -> tuple[str, str]:
        """Return (content, version)."""
        raise NotImplementedError

    def write_text(self, path: str, content: str, *, expected_version: str | None = None) -> str:
        """Write atomically; return the new version."""
        raise NotImplementedError

    def edit(self, path: str, old: str, new: str, *,
             expected_version: str | None = None, count: int = 1) -> tuple[str, int]:
        """Literal replace; return (new_version, replacements_made)."""
        raise NotImplementedError

    def list_dir(self, path: str) -> list[FileInfo]:
        raise NotImplementedError

    def stat(self, path: str) -> FileInfo:
        raise NotImplementedError

    def glob(self, pattern: str, *, root: str | None = None, limit: int = 500) -> list[str]:
        raise NotImplementedError


@dataclass
class ReadIntent:
    path: str
    max_bytes: int | None = None


@dataclass
class WriteIntent:
    path: str
    content: str
    expected_version: str | None = None


@dataclass
class EditIntent:
    path: str
    old: str
    new: str
    expected_version: str | None = None
    count: int = 1


class FsService:
    """The seam. Holds one provider and fires the `fs/*` intent waterfalls around mutations.

    The intents are separate events rather than one `fs/access` because they carry different
    risk. A read is cheap to allow and expensive to block; a write is the opposite. Collapsing
    them would force every policy plugin to re-derive the distinction from the arguments.

    WHY EACH INTENT IS A MUTABLE OBJECT AND NOT POSITIONAL ARGUMENTS

        `next()` in a waterfall takes no arguments, so a listener cannot pass different values
        downstream by calling `next(new_path)`. dsh solves this the same way: "cooperative
        listeners usually mutate a shared request or decision object and then delegate."

        So a listener that wants to rewrite a path, or impose a version the caller did not
        supply, mutates the intent and calls `next()`. `fs_guard.py` does exactly that, and the
        first version of this file needed an awkward helper precisely because the intents were
        positional. The shape of the event is doing real work here.
    """

    def __init__(self, ctx) -> None:
        self._ctx = ctx
        self._provider: FsProvider | None = None

    def register_provider(self, provider: FsProvider) -> Callable[[], None]:
        if self._provider is not None:
            raise RuntimeError("an fs provider is already registered")
        self._provider = provider

        def dispose() -> None:
            if self._provider is provider:
                self._provider = None

        return dispose

    @property
    def provider(self) -> FsProvider:
        if self._provider is None:
            raise FsError("no filesystem provider is mounted — check your profile")
        return self._provider

    # --- reads -------------------------------------------------------------------------------

    def read_text(self, path: str, *, max_bytes: int | None = None) -> tuple[str, str]:
        intent = ReadIntent(path=path, max_bytes=max_bytes)

        def do(i: ReadIntent):
            return self.provider.read_text(i.path, max_bytes=i.max_bytes)

        return self._ctx.waterfall("fs/read-intent", intent, final=do)

    def list_dir(self, path: str) -> list[FileInfo]:
        return self.provider.list_dir(path)

    def stat(self, path: str) -> FileInfo:
        return self.provider.stat(path)

    def glob(self, pattern: str, *, root: str | None = None, limit: int = 500) -> list[str]:
        return self.provider.glob(pattern, root=root, limit=limit)

    # --- mutations ---------------------------------------------------------------------------

    def write_text(self, path: str, content: str, *,
                   expected_version: str | None = None) -> str:
        intent = WriteIntent(path=path, content=content, expected_version=expected_version)

        def do(i: WriteIntent):
            return self.provider.write_text(i.path, i.content,
                                            expected_version=i.expected_version)

        return self._ctx.waterfall("fs/write-intent", intent, final=do)

    def edit(self, path: str, old: str, new: str, *,
             expected_version: str | None = None, count: int = 1) -> tuple[str, int]:
        intent = EditIntent(path=path, old=old, new=new,
                            expected_version=expected_version, count=count)

        def do(i: EditIntent):
            return self.provider.edit(i.path, i.old, i.new,
                                      expected_version=i.expected_version, count=i.count)

        return self._ctx.waterfall("fs/edit-intent", intent, final=do)


def apply(ctx, config=None) -> None:
    ctx.provide("fs", FsService(ctx))
