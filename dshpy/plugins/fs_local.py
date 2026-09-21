"""Local filesystem provider for `ctx.fs`.

Nothing here knows about policy, permissions, or the project root. That is deliberate: dsh's
providers implement the *mechanism*, and the guard (`plugins/fs_guard.py`) supplies the policy.
Mixing them is how you end up with a backend you cannot reuse in a sandbox.

THE ATOMIC WRITE

    `write_text` writes to a temp file in the same directory and `os.replace`s it into place.
    `os.replace` is atomic on POSIX and Windows, so a reader never sees a half-written file and
    a crash mid-write leaves the original intact.

    The same-directory detail matters: `os.replace` across filesystems is not atomic and may
    fail outright, so a temp file in /tmp would quietly break the guarantee on any machine
    where the project lives on a different mount.
"""

from __future__ import annotations

import fnmatch
import os
import tempfile
from pathlib import Path

from dshpy.services.fs import FileInfo, FsError, FsProvider, VersionConflict

name = "fs-local"
inject = ["fs"]

DEFAULT_MAX_BYTES = 256 * 1024  # a model does not benefit from a 40MB file in its context


class LocalFs(FsProvider):
    def __init__(self, root: str | None = None, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self.root = Path(root or Path.cwd()).resolve()
        self.max_bytes = max_bytes

    def _p(self, path: str) -> Path:
        # Resolve relative paths against the root, so the model can say "README.md".
        p = Path(path).expanduser()
        return (p if p.is_absolute() else self.root / p).resolve()

    @staticmethod
    def _version(p: Path) -> str:
        """An opaque version token. Callers compare it; they must not parse it.

        (mtime_ns, size) is cheap enough to take on every read. A content hash would be
        stronger and costs a full read per check — a real trade, not an oversight.
        """
        st = p.stat()
        return f"{st.st_mtime_ns}:{st.st_size}"

    # --- reads -------------------------------------------------------------------------------

    def read_text(self, path: str, *, max_bytes: int | None = None) -> tuple[str, str]:
        p = self._p(path)
        if not p.exists():
            raise FsError(f"no such file: {path}")
        if p.is_dir():
            raise FsError(f"{path} is a directory, not a file")

        cap = max_bytes or self.max_bytes
        version = self._version(p)
        data = p.read_bytes()
        truncated = len(data) > cap
        text = data[:cap].decode("utf-8", errors="replace")
        if truncated:
            # Say so in the content. A silently truncated file is one the model will reason
            # about as if it were complete.
            text += f"\n\n[truncated at {cap} bytes of {len(data)}]"
        return text, version

    def list_dir(self, path: str) -> list[FileInfo]:
        p = self._p(path)
        if not p.is_dir():
            raise FsError(f"not a directory: {path}")
        out = []
        for child in sorted(p.iterdir()):
            try:
                st = child.stat()
                out.append(FileInfo(path=str(child), size=st.st_size,
                                    version=f"{st.st_mtime_ns}:{st.st_size}",
                                    is_dir=child.is_dir()))
            except OSError:
                continue  # a broken symlink or a race; listing should not fail wholesale
        return out

    def stat(self, path: str) -> FileInfo:
        p = self._p(path)
        if not p.exists():
            raise FsError(f"no such path: {path}")
        st = p.stat()
        return FileInfo(path=str(p), size=st.st_size, version=self._version(p),
                        is_dir=p.is_dir())

    def glob(self, pattern: str, *, root: str | None = None, limit: int = 500) -> list[str]:
        base = self._p(root) if root else self.root
        matches = []
        for dirpath, dirnames, filenames in os.walk(base):
            # Skip the directories nobody wants in an agent's search results.
            dirnames[:] = [d for d in dirnames
                           if d not in {".git", "__pycache__", ".venv", "node_modules"}]
            for filename in filenames:
                full = Path(dirpath) / filename
                rel = full.relative_to(base)
                if fnmatch.fnmatch(str(rel), pattern) or fnmatch.fnmatch(filename, pattern):
                    matches.append(str(full))
                    if len(matches) >= limit:
                        return sorted(matches)
        return sorted(matches)

    # --- mutations ---------------------------------------------------------------------------

    def _check_version(self, p: Path, expected: str | None) -> None:
        if expected is None:
            return
        actual = self._version(p) if p.exists() else None
        if actual != expected:
            raise VersionConflict(
                f"{p.name} changed since you read it "
                f"(expected version {expected}, found {actual}). Re-read it before writing."
            )

    def write_text(self, path: str, content: str, *,
                   expected_version: str | None = None) -> str:
        p = self._p(path)
        self._check_version(p, expected_version)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Temp file in the SAME directory: os.replace is only atomic within a filesystem.
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(content)
            os.replace(tmp, p)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return self._version(p)

    def edit(self, path: str, old: str, new: str, *,
             expected_version: str | None = None, count: int = 1) -> tuple[str, int]:
        p = self._p(path)
        if not p.exists():
            raise FsError(f"no such file: {path}")
        self._check_version(p, expected_version)

        content = p.read_text(encoding="utf-8")
        occurrences = content.count(old)
        if occurrences == 0:
            raise FsError(f"the text to replace was not found in {path}")
        if count == 1 and occurrences > 1:
            # Ambiguity is an error, not a coin flip. Replacing "the first one" when the model
            # meant a different one is a silent wrong edit, which is worse than a refusal.
            raise FsError(
                f"the text to replace appears {occurrences} times in {path}; "
                f"include more surrounding context to make it unique, or pass count=0 for all"
            )

        replaced = content.replace(old, new) if count == 0 else content.replace(old, new, count)
        made = occurrences if count == 0 else min(count, occurrences)
        return self.write_text(str(p), replaced), made


def apply(ctx, config=None) -> None:
    config = config or {}
    provider = LocalFs(root=config.get("root"), max_bytes=config.get("max_bytes",
                                                                    DEFAULT_MAX_BYTES))
    ctx.effect(ctx.fs.register_provider(provider))
