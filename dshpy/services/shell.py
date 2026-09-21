"""`ctx.shell` — the subprocess seam.

dsh splits this as `shell` (contract) + `bash-local` / `bash-sandbox` (providers) +
`tool-bash` (the model-facing tool). Same three-role shape as the filesystem, and for the same
reason: pointing the agent at a container should be a config change, not a fork of the tool.

Unlike `ctx.llm`, this seam allows ONE provider at a time. dsh makes the same distinction --
several LLM adapters coexist because a request names its provider, but "run this command" has
no such selector, so a second executor would be ambiguous rather than useful.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from dshpy.cancel import NEVER, CancelToken

name = "shell"
inject: list[str] = []


@dataclass
class ShellResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    truncated: bool = False
    timed_out: bool = False


class ShellProvider:
    def run(self, command: str, *, cwd: str | None = None, timeout: float | None = None,
            token: CancelToken = NEVER, max_output: int = 100_000) -> ShellResult:
        raise NotImplementedError


class ShellService:
    def __init__(self, ctx) -> None:
        self._ctx = ctx
        self._provider: ShellProvider | None = None

    def register_provider(self, provider: ShellProvider) -> Callable[[], None]:
        if self._provider is not None:
            raise RuntimeError("a shell provider is already registered")
        self._provider = provider

        def dispose() -> None:
            if self._provider is provider:
                self._provider = None

        return dispose

    def run(self, command: str, **kwargs) -> ShellResult:
        if self._provider is None:
            raise RuntimeError("no shell provider is mounted — check your profile")

        def do(cmd: str):
            return self._provider.run(cmd, **kwargs)

        # A waterfall so a sandbox plugin can rewrite argv before it is spawned, exactly as
        # dsh describes ("consumers wrap argv before spawning").
        return self._ctx.waterfall("shell/exec-intent", command, final=do)


def apply(ctx, config=None) -> None:
    ctx.provide("shell", ShellService(ctx))
