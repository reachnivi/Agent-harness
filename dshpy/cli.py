"""Run the plugin harness.

    uv run -m dshpy.cli "what is 17 * 23, and what time is it?"
    uv run -m dshpy.cli --provider anthropic "same question, different wire protocol"
    uv run -m dshpy.cli --dump-config        # show the mounted tree, run nothing
    uv run -m dshpy.cli --yes "..."          # don't prompt for tool permission

Note how little this file does. It composes a profile, mounts it, and calls `ctx.agent_loop`.
There is no place here to put a special case for a provider, a tool, or a policy — which is the
symptom you want from an architecture like this one.
"""

from __future__ import annotations

import argparse
import sys

from common.ollama import PreflightError, is_local, preflight
from dshpy.core.context import Runtime
from dshpy.profiles import coding as coding_profile
from dshpy.profiles import default as default_profile


def build_runtime(provider: str | None = None, *, assume_yes: bool = False,
                  quiet: bool = False, profile: str = "default") -> Runtime:
    if provider:
        default_profile.PROVIDER = provider

    rows = (coding_profile if profile == "coding" else default_profile).rows()
    if assume_yes:
        for row in rows:
            if getattr(row["plugin"], "name", "") == "permission":
                # --yes means "answer yes to every prompt", NOT "rewrite the policy". Setting
                # default=allow would be the wrong fix: DEFAULT_POLICY marks write_file and
                # bash as "ask" explicitly, and an explicit entry beats the default -- so
                # --yes would silently do nothing for exactly the tools it matters for.
                row.setdefault("config", {})["ask"] = lambda exec_: True
    if quiet:
        for row in rows:
            if getattr(row["plugin"], "name", "") == "telemetry":
                row.setdefault("config", {})["verbose"] = False

    runtime = Runtime()
    runtime.mount_all(rows)
    return runtime


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dshpy", description=__doc__)
    parser.add_argument("prompt", nargs="?", help="what to ask")
    parser.add_argument("--resume", metavar="SESSION_ID", help="continue a stored session")
    parser.add_argument("--fork", metavar="SESSION_ID",
                        help="branch a stored session at its last completed turn")
    parser.add_argument("--sessions", action="store_true", help="list stored sessions")
    parser.add_argument("--compact", action="store_true",
                        help="summarize earlier history before running")
    parser.add_argument("--profile", choices=["default", "coding"], default="default",
                        help="coding adds a filesystem, a shell, and the tools for them")
    parser.add_argument("--provider", choices=["openai", "anthropic"],
                        help="which wire protocol to speak (default: $DS_PROVIDER or openai)")
    parser.add_argument("--dump-config", action="store_true",
                        help="print the mounted plugin tree and exit")
    parser.add_argument("--yes", action="store_true", help="auto-allow tool calls")
    parser.add_argument("--quiet", action="store_true", help="suppress telemetry lines")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args(argv)

    runtime = build_runtime(args.provider, assume_yes=args.yes, quiet=args.quiet,
                            profile=args.profile)

    if args.dump_config:
        print(runtime.dump())
        return 0

    store = runtime.services.get("persistence")
    if args.sessions:
        for session_id in (store.list_sessions() if store else []):
            print(session_id)
        return 0

    if args.fork:
        child = store.fork(args.fork)
        print(f"forked {args.fork} -> {child}")
        args.resume = child

    if args.resume:
        count = store.resume(args.resume)
        print(f"resumed {args.resume} ({count} events)")

    if not args.prompt and not (args.sessions or args.fork or args.compact):
        parser.error("a prompt is required (or use --dump-config / --sessions)")

    loop = runtime.services.get("agent_loop")
    if loop is None:
        print("agent_loop never activated — here is why:\n", file=sys.stderr)
        print(runtime.dump(), file=sys.stderr)
        return 1

    if not args.skip_preflight:
        adapter_base = _adapter_base_url(runtime)
        if is_local(adapter_base):
            try:
                preflight(adapter_base, loop.model, quiet=args.quiet)
            except PreflightError as exc:
                print(f"\nPreflight failed:\n{exc}\n", file=sys.stderr)
                return 1

    if args.compact:
        result = runtime.services["compaction"].compact(reason="manual")
        print(f"compacted {result.shadowed_events} events" if result else "nothing to compact")

    print(f"\nyou> {args.prompt}\n")
    streaming = "stream-ui" in runtime.plugin_names()
    # Read this BEFORE dispose(): teardown frees every service key, so reaching for
    # `runtime.services["sessions"]` afterwards is a KeyError. Reversible effects are only a
    # good property if you remember they actually reverse.
    session_id = runtime.services["sessions"].session_id
    try:
        answer = loop.run(args.prompt)
    finally:
        runtime.dispose()  # unwinds every plugin's registrations, newest first
    # The stream UI already rendered this token by token; printing it again would show the
    # same answer twice. Whether it did is a property of the mounted profile, not of the CLI.
    if not streaming:
        print(f"\nbot> {answer}")
    print(f"\n[session {session_id} — resume with --resume {session_id}]")
    return 0


def _adapter_base_url(runtime: Runtime) -> str:
    """Ask the mounted adapter where it points, rather than re-deriving it from config."""
    llm = runtime.services.get("llm")
    for route in (llm.providers() if llm else []):
        adapter = llm._adapters[route]
        return getattr(adapter, "base_url", "")
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
