"""The coding-agent profile: everything in `default`, plus a filesystem and a shell.

A profile is a SET of plugin rows, so this is literally the default list with four more entries
and a different system prompt. That it composes this cleanly is the architecture's claim being
cashed: turning a chat agent into a coding agent adds rows, it does not fork the loop.
"""

from __future__ import annotations

import os
from pathlib import Path

from dshpy.plugins import (
    fs_guard,
    fs_local,
    shell_local,
    subagent_in_process,
    tool_bash,
    tool_fs,
    tool_subagent,
)
from dshpy.profiles import default
from dshpy.services import fs, shell, subagents

ROOT = os.environ.get("DS_ROOT", str(Path.cwd()))

SYSTEM_PROMPT = (
    "You are a coding assistant working inside a project directory. "
    "Read a file before editing it — you will be refused otherwise, because a blind overwrite "
    "destroys changes you cannot see. Prefer edit_file over write_file for small changes. "
    "Use glob and grep to find things rather than guessing at paths. "
    "For work that needs to read a lot to produce a little — searching the codebase, checking "
    "many files — delegate with the task tool so its output does not fill your context. "
    "Run tests with bash when you have changed something."
)


def rows() -> list[dict]:
    base = [r for r in default.rows()
            if getattr(r["plugin"], "name", "") != "agent-loop"]

    return base + [
        # --- the two new seams and their local providers -----------------------------------
        {"plugin": fs},
        {"plugin": shell},
        {"plugin": fs_local, "config": {"root": ROOT}},
        {"plugin": shell_local, "config": {"cwd": ROOT}},

        # --- the invariants, next to the filesystem they protect ---------------------------
        {"plugin": fs_guard, "config": {"root": ROOT}},

        # --- the capabilities ---------------------------------------------------------------
        {"plugin": tool_fs},
        {"plugin": tool_bash},

        # --- delegation: a child agent with its own session and a restricted tool scope -----
        {"plugin": subagents},
        {"plugin": subagent_in_process},
        {"plugin": tool_subagent, "config": {
            "max_depth": 2,
            # A researcher should not be able to write. The restriction is enforced by the
            # tool scope, not by asking the sub-agent nicely.
            "allowed_tools": ["read_file", "list_dir", "glob", "grep", "get_time"],
        }},

        # --- the loop, last for readability only; inject decides the real order -------------
        {"plugin": __import__("dshpy.services.agent_loop", fromlist=["x"]), "config": {
            "model": default.MODEL,
            "provider": default.PROVIDER,
            "system": SYSTEM_PROMPT,
            "max_steps": 20,      # a coding task needs more steps than a chat turn
            "max_tokens": 4096,
        }},
    ]
