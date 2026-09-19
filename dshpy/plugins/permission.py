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

DEFAULT_POLICY = {
    "get_time": "allow",
    "calculate": "allow",
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
    def confine_paths(exec_):
        """Any argument that looks like a path must stay inside the project root.

        Schema validation is NOT path validation: a schema saying "string" happily accepts
        '../../.ssh/authorized_keys'. And this belongs in a guard rather than the gate above
        precisely because no configuration should be able to switch it off.
        """
        for key, value in exec_.arguments.items():
            if not isinstance(value, str) or key not in {"path", "file", "filename", "dir"}:
                continue
            try:
                resolved = Path(value).expanduser().resolve()
            except (OSError, RuntimeError):
                return Deny(f"unusable path in {key!r}")
            if not resolved.is_relative_to(root):
                return Deny(f"path {value!r} is outside the project root")
        return None

    ctx.effect(ctx.tools.guard(confine_paths))


def _prompt(exec_) -> bool:
    print(f"\n  {exec_.name}({exec_.arguments})")
    try:
        return input("  allow? [y/N] ").strip().lower().startswith("y")
    except EOFError:
        return False
