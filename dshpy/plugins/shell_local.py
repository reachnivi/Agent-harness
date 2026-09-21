"""Local subprocess provider for `ctx.shell`.

TWO THINGS THIS GETS RIGHT THAT A NAIVE subprocess.run DOES NOT

    1. **It polls the cancellation token, and kills the whole process GROUP.**
       `subprocess.run(timeout=...)` blocks until its own timeout and cannot be interrupted by
       an outer cancellation, so this polls in a loop instead.

       Killing the group rather than the process is the part that is easy to get wrong, and I
       did get it wrong first. With `shell=True` the child is `/bin/sh -c "<command>"`, and the
       shell may *fork* the real command rather than exec it. Terminating the shell then leaves
       a grandchild alive — and that grandchild still holds the stdout/stderr pipes open, so
       the subsequent `communicate()` blocks waiting for an EOF that will not come until the
       orphan finishes on its own. The symptom is a "timeout" that reports success at killing
       the process and still takes the full 30 seconds to return.

       So: `start_new_session=True` puts the child in its own process group, and
       `os.killpg` takes down the shell and everything it spawned.

    2. **It bounds the output.** A command that prints 200MB would otherwise go straight into
       the model's context. The cap is applied per stream and the result says it truncated,
       because a silently shortened build log is one the model will misread.

`shell=True` is used deliberately: the model writes shell syntax (pipes, globs, &&) and the
tool's whole purpose is to run it. That makes this an intentionally powerful tool, which is
exactly why it defaults to "ask" in the permission policy -- the safety here is the gate in
front of it, not an illusion of argument sanitization.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time

from dshpy.cancel import NEVER, CancelToken
from dshpy.services.shell import ShellProvider, ShellResult

name = "shell-local"
inject = ["shell"]

POLL_INTERVAL = 0.02
GRACE_PERIOD = 0.5  # between SIGTERM and SIGKILL
POSIX = sys.platform != "win32"

# The tool is called `bash`, so run bash where it exists. On this container /bin/sh is dash,
# which has no brace expansion, no [[ ]], no arrays -- a command the model wrote expecting bash
# fails in ways that look like the model's mistake rather than the harness's.
SHELL = shutil.which("bash") or "/bin/sh"


class LocalShell(ShellProvider):
    def __init__(self, cwd: str | None = None, default_timeout: float = 120.0) -> None:
        self.cwd = cwd
        self.default_timeout = default_timeout

    def run(self, command: str, *, cwd: str | None = None, timeout: float | None = None,
            token: CancelToken = NEVER, max_output: int = 100_000) -> ShellResult:
        limit = timeout if timeout is not None else self.default_timeout
        deadline = time.monotonic() + limit

        proc = subprocess.Popen(
            command, shell=True, executable=SHELL if POSIX else None, cwd=cwd or self.cwd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            # Its own process group, so we can kill the shell AND anything it forked.
            start_new_session=POSIX,
        )

        timed_out = False
        while proc.poll() is None:
            if token.is_set() or time.monotonic() > deadline:
                timed_out = not token.is_set()
                self._kill_group(proc)
                break
            time.sleep(POLL_INTERVAL)

        # Safe now: the group is gone, so nothing is still holding the pipes open.
        stdout, stderr = proc.communicate()
        stdout, stderr = stdout or "", stderr or ""
        truncated = len(stdout) > max_output or len(stderr) > max_output

        return ShellResult(
            command=command,
            exit_code=proc.returncode if proc.returncode is not None else -1,
            stdout=stdout[:max_output],
            stderr=stderr[:max_output],
            truncated=truncated,
            timed_out=timed_out,
        )


    @staticmethod
    def _kill_group(proc: subprocess.Popen) -> None:
        """SIGTERM the group, then SIGKILL what ignored it."""
        if not POSIX:
            proc.terminate()
            try:
                proc.wait(timeout=GRACE_PERIOD)
            except subprocess.TimeoutExpired:
                proc.kill()
            return

        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            return  # it exited between the poll and here

        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                return
            try:
                proc.wait(timeout=GRACE_PERIOD)
                return
            except subprocess.TimeoutExpired:
                continue  # it ignored SIGTERM; escalate


def apply(ctx, config=None) -> None:
    config = config or {}
    provider = LocalShell(cwd=config.get("cwd"),
                          default_timeout=config.get("timeout", 120.0))
    ctx.effect(ctx.shell.register_provider(provider))
