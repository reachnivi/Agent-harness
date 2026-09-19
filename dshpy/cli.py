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
from dshpy.profiles import default as default_profile


def build_runtime(provider: str | None = None, *, assume_yes: bool = False,
                  quiet: bool = False) -> Runtime:
    if provider:
        default_profile.PROVIDER = provider

    rows = default_profile.rows()
    if assume_yes:
        for row in rows:
            if row["plugin"] is not None and getattr(row["plugin"], "name", "") == "permission":
                row.setdefault("config", {})["default"] = "allow"
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
    parser.add_argument("--provider", choices=["openai", "anthropic"],
                        help="which wire protocol to speak (default: $DS_PROVIDER or openai)")
    parser.add_argument("--dump-config", action="store_true",
                        help="print the mounted plugin tree and exit")
    parser.add_argument("--yes", action="store_true", help="auto-allow tool calls")
    parser.add_argument("--quiet", action="store_true", help="suppress telemetry lines")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args(argv)

    runtime = build_runtime(args.provider, assume_yes=args.yes, quiet=args.quiet)

    if args.dump_config:
        print(runtime.dump())
        return 0

    if not args.prompt:
        parser.error("a prompt is required (or use --dump-config)")

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

    print(f"\nyou> {args.prompt}\n")
    try:
        answer = loop.run(args.prompt)
    finally:
        runtime.dispose()  # unwinds every plugin's registrations, newest first
    print(f"\nbot> {answer}")
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
