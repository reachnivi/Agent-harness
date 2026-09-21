"""The bash tool.

The most powerful tool in the harness and the one with the least code, which is the right ratio:
it delegates to `ctx.shell` and formats the result. The safety is the permission gate in front
of it (`permission.py` defaults it to "ask") and the provider's own cancellation and output
bounds -- not anything this file does.

WHY THE OUTPUT FORMAT MATTERS MORE THAN IT LOOKS

    A model reading a command's output needs to know three things: did it succeed, what did it
    print, and was anything withheld. So the result always states the exit code explicitly, and
    always says when output was truncated or the command was killed.

    The failure mode this avoids: a build that printed 200MB, got silently cut to the first
    100KB, and left the model concluding the build passed because the error was in the part it
    never saw.
"""

from __future__ import annotations

from dshpy.cancel import NEVER
from dshpy.services.tools import define_tool

name = "tool-bash"
inject = ["tools", "shell"]


def apply(ctx, config=None) -> None:
    config = config or {}
    default_timeout = config.get("timeout", 120.0)
    max_output = config.get("max_output", 100_000)

    def bash(args, exec):
        result = ctx.shell.run(
            args["command"],
            cwd=args.get("cwd"),
            timeout=float(args.get("timeout", default_timeout)),
            token=exec.token or NEVER,
            max_output=max_output,
        )

        parts = [f"$ {result.command}", f"exit code: {result.exit_code}"]
        if result.timed_out:
            parts.append("[killed: timed out]")
        if result.stdout:
            parts.append(f"stdout:\n{result.stdout}")
        if result.stderr:
            parts.append(f"stderr:\n{result.stderr}")
        if not result.stdout and not result.stderr:
            parts.append("(no output)")
        if result.truncated:
            parts.append(f"[output truncated at {max_output} bytes -- it was longer]")
        return "\n".join(parts)

    ctx.effect(ctx.tools.register(define_tool(
        name="bash",
        description=("Run a shell command and return its exit code, stdout and stderr. "
                     "Supports pipes, globs and && since it runs through a shell."),
        parameters={
            "command": {"type": "string", "required": True},
            "cwd": {"type": "string", "description": "Working directory."},
            "timeout": {"type": "number", "description": "Seconds before the command is killed."},
        },
        execute=bash,
    )))
