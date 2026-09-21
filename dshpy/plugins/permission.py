"""Permission policy, as a plugin. The showcase for the whole architecture.

In stages/s03_loop.py, adding a permission prompt meant editing the loop. Here it is this file,
mounted beside the others, and the loop is untouched and unaware.

TWO MECHANISMS, DELIBERATELY NOT ONE

    `tools/pre-execute` is REORDERABLE POLICY. It composes: a plugin mounted later, or a
    listener registered with prepend=True, can decline to delegate and thereby allow something
    this gate would have questioned. That is correct for policy, whose whole nature is that the
    final answer depends on configuration.

    `ctx.tools.guard()` is a MONOTONIC FINAL DENY. Once a guard denies, nothing can allow.

    Put invariants in the guard and preferences in the gate. Put an invariant in the gate by
    mistake and you are one badly-ordered profile away from it being bypassed -- which is how
    permission systems grow holes. tests/test_pipeline.py asserts the distinction holds.

A DENIAL IS A RESULT, NOT AN EXCEPTION

    The pipeline turns a Deny into an is_error tool result, which goes back to the model as
    ordinary content. The model reads "user declined" and picks another approach. Raising
    instead would kill that recovery path -- rule 3 from stage 3, now enforced structurally
    rather than by remembering to write a try/except.
"""

from __future__ import annotations

from pathlib import Path

from dshpy.services.tools import Deny

name = "permission"
inject = ["tools"]

# Read-only tools allow; anything that mutates the world or runs code asks.
#
# This is the line where the gate stops being a demo. `bash` can do anything the user can, and
# `write_file` can destroy work -- so the default answer for both is "ask a human", and a
# deployment that wants otherwise has to say so explicitly. Defaulting to allow and relying on
# the model's judgement is a decision, and it should be one somebody made on purpose.
DEFAULT_POLICY = {
    # harmless
    "get_time": "allow",
    "calculate": "allow",
    # read-only: cheap to allow, expensive to interrupt for
    "read_file": "allow",
    "list_dir": "allow",
    "glob": "allow",
    "grep": "allow",
    # mutating or arbitrary-code: ask
    "write_file": "ask",
    "edit_file": "ask",
    "bash": "ask",
}


def apply(ctx, config=None) -> None:
    config = config or {}
    policy: dict[str, str] = {**DEFAULT_POLICY, **config.get("policy", {})}
    default = config.get("default", "ask")
    ask_fn = config.get("ask", _prompt)
    root = Path(config.get("root", Path.cwd())).resolve()

    # --- reorderable policy ---------------------------------------------------------------
    def gate(exec_, next):
        verdict = policy.get(exec_.name, default)
        if verdict == "allow":
            return next()
        if verdict == "deny":
            return Deny(f"tool {exec_.name!r} is denied by policy")
        if ask_fn(exec_):
            return next()
        return Deny("user declined to run this tool")

    ctx.on("tools/pre-execute", gate)

    # --- monotonic invariant --------------------------------------------------------------
    #
    # Path confinement MOVED to plugins/fs_guard.py in phase 7. It belongs next to the
    # filesystem: it must understand symlinks and roots, and it must protect every fs consumer
    # rather than only tools whose argument happens to be named `path`. It is still a guard,
    # not a gate -- nothing mounted later may switch it off.
    #
    # A generic argument-shaped fallback is kept here for tools that take a path but do NOT go
    # through ctx.fs (a future tool nobody has written yet). Defence in depth: the real check
    # is in fs_guard, this one catches the tool that forgot to use the seam.
    def confine_stray_paths(exec_):
        for key, value in exec_.arguments.items():
            if not isinstance(value, str) or key not in {"path", "file", "filename", "dir"}:
                continue
            try:
                candidate = Path(value).expanduser()
                # Resolve a relative path against the ROOT, not the process cwd. Getting this
                # wrong makes the guard reject `read_file("f.txt")` as an escape, because
                # Path("f.txt").resolve() silently anchors to wherever the process happens to
                # be running -- which is not where the agent's root is.
                if not candidate.is_absolute():
                    candidate = root / candidate
                resolved = candidate.resolve()
            except (OSError, RuntimeError):
                return Deny(f"unusable path in {key!r}")
            if resolved != root and root not in resolved.parents:
                return Deny(f"path {value!r} is outside the project root")
        return None

    ctx.effect(ctx.tools.guard(confine_stray_paths))


def _prompt(exec_) -> bool:
    print(f"\n  {exec_.name}({exec_.arguments})")
    try:
        return input("  allow? [y/N] ").strip().lower().startswith("y")
    except EOFError:
        return False
