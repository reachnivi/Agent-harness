"""Filesystem tools, as a plugin.

Note what these do NOT do: no path checking, no permission logic, no version tracking. Each is
a thin translation from "what the model asked for" to "a call on `ctx.fs`". Everything
dangerous is enforced underneath them, by `fs_guard` on the `fs/*` intents.

That is the payoff of the seam, and it is worth checking against the alternative: if these
tools did their own confinement, then every future tool would have to remember to, and the
first one that forgot would be the hole. Putting it under the interface means a tool CANNOT
forget, because the tool never sees a raw path operation at all.

WHY read_file RETURNS LINE NUMBERS

    A model asked to edit a file needs to refer to a location. Line numbers in the read output
    give it a vocabulary for that, and make a later `edit_file` far more likely to quote a
    unique string. It costs a few tokens and removes a whole class of failed edit.
"""

from __future__ import annotations


from dshpy.services.fs import FsError
from dshpy.services.tools import define_tool

name = "tool-fs"
inject = ["tools", "fs"]


def apply(ctx, config=None) -> None:
    config = config or {}
    max_bytes = config.get("max_bytes")
    fs = ctx.fs

    def read_file(args, exec):
        content, version = fs.read_text(args["path"], max_bytes=max_bytes)
        numbered = "\n".join(f"{i:>5}  {line}"
                             for i, line in enumerate(content.splitlines(), start=1))
        return f"{args['path']} (version {version})\n{numbered}"

    def write_file(args, exec):
        version = fs.write_text(args["path"], args["content"])
        return f"wrote {args['path']} ({len(args['content'])} bytes, version {version})"

    def edit_file(args, exec):
        version, made = fs.edit(args["path"], args["old_text"], args["new_text"],
                                count=int(args.get("count", 1)))
        return f"replaced {made} occurrence(s) in {args['path']} (version {version})"

    def list_dir(args, exec):
        entries = fs.list_dir(args.get("path", "."))
        return "\n".join(
            f"{'dir ' if e.is_dir else 'file'}  {e.size:>9}  {e.path}" for e in entries
        ) or "(empty)"

    def glob_files(args, exec):
        matches = fs.glob(args["pattern"], root=args.get("path"),
                          limit=int(args.get("limit", 200)))
        return "\n".join(matches) or "(no matches)"

    def grep(args, exec):
        """Search file contents. Deliberately goes through ctx.fs rather than os.walk, so it
        obeys the same confinement as everything else."""
        import re

        try:
            pattern = re.compile(args["pattern"])
        except re.error as exc:
            raise FsError(f"invalid regex: {exc}") from None

        limit = int(args.get("limit", 100))
        hits = []
        for path in fs.glob(args.get("glob", "*"), root=args.get("path"), limit=2000):
            if exec.token:
                exec.token.check()  # searching a big tree can run long
            try:
                content, _ = fs.read_text(path)
            except Exception:  # noqa: BLE001 - an unreadable file must not stop the search
                continue
            for lineno, line in enumerate(content.splitlines(), start=1):
                if pattern.search(line):
                    hits.append(f"{path}:{lineno}: {line.strip()[:200]}")
                    if len(hits) >= limit:
                        return "\n".join(hits) + f"\n[stopped at {limit} matches]"
        return "\n".join(hits) or "(no matches)"

    tools = [
        define_tool(
            name="read_file",
            description=("Read a text file. Returns the content with line numbers, and a "
                         "version token. You must read a file before writing or editing it."),
            parameters={"path": {"type": "string", "required": True,
                                 "description": "Path, absolute or relative to the project root."}},
            execute=read_file,
        ),
        define_tool(
            name="write_file",
            description=("Write a text file, replacing its entire contents. Creates it if it "
                         "does not exist. For a small change to an existing file prefer "
                         "edit_file, which does not require sending the whole file back."),
            parameters={
                "path": {"type": "string", "required": True},
                "content": {"type": "string", "required": True},
            },
            execute=write_file,
        ),
        define_tool(
            name="edit_file",
            description=("Replace an exact string in a file. The old_text must appear exactly "
                         "once unless count is 0 (replace all) — include surrounding lines to "
                         "make it unique."),
            parameters={
                "path": {"type": "string", "required": True},
                "old_text": {"type": "string", "required": True},
                "new_text": {"type": "string", "required": True},
                "count": {"type": "number",
                          "description": "1 (default) for a unique match, 0 to replace all."},
            },
            execute=edit_file,
        ),
        define_tool(
            name="list_dir",
            description="List the entries of a directory.",
            parameters={"path": {"type": "string", "description": "Defaults to the root."}},
            execute=list_dir,
        ),
        define_tool(
            name="glob",
            description="Find files by glob pattern, e.g. '*.py' or 'src/**/test_*.py'.",
            parameters={
                "pattern": {"type": "string", "required": True},
                "path": {"type": "string", "description": "Directory to search from."},
                "limit": {"type": "number"},
            },
            execute=glob_files,
        ),
        define_tool(
            name="grep",
            description="Search file contents with a regular expression. Returns path:line: text.",
            parameters={
                "pattern": {"type": "string", "required": True},
                "glob": {"type": "string", "description": "Restrict to files matching this."},
                "path": {"type": "string"},
                "limit": {"type": "number"},
            },
            execute=grep,
        ),
    ]

    for tool in tools:
        ctx.effect(ctx.tools.register(tool))
