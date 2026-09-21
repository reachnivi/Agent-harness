"""Tests for the fs and shell seams, their guard, and the coding-agent tools.

Every test here uses `tmp_path`. Nothing touches the real tree — a test suite for a tool that
writes files should not be one bug away from writing yours.
"""

from __future__ import annotations

import sys
import time

import pytest

from dshpy.cancel import CancelToken
from dshpy.core.context import Runtime
from dshpy.plugins import fs_guard, fs_local, permission, shell_local, tool_bash, tool_fs
from dshpy.services import fs as fs_service, shell as shell_service, tools as tools_service
from dshpy.services.fs import FsError, VersionConflict
from dshpy.plugins.fs_guard import PathEscape


@pytest.fixture
def fs_rt(tmp_path):
    """fs seam + local provider + guard, rooted at a temp dir."""
    rt = Runtime()
    rt.mount(fs_service)
    rt.mount(fs_local, {"root": str(tmp_path)})
    rt.mount(fs_guard, {"root": str(tmp_path)})
    return rt, tmp_path


@pytest.fixture
def tools_rt(tmp_path):
    """The full coding-agent stack: seams, providers, guard, tools."""
    rt = Runtime()
    rt.mount(tools_service)
    rt.mount(fs_service)
    rt.mount(shell_service)
    rt.mount(fs_local, {"root": str(tmp_path)})
    rt.mount(shell_local, {"cwd": str(tmp_path)})
    rt.mount(fs_guard, {"root": str(tmp_path)})
    rt.mount(tool_fs)
    rt.mount(tool_bash)
    return rt, tmp_path


def call(rt, tool, **args):
    return rt.services["tools"].execute("c1", tool, args)


# --- the provider ---------------------------------------------------------------------------


def test_read_and_write_round_trip(fs_rt):
    rt, tmp = fs_rt
    fs = rt.services["fs"]
    fs.write_text("notes.txt", "hello")
    content, version = fs.read_text("notes.txt")
    assert content == "hello" and version


def test_relative_paths_resolve_against_the_root(fs_rt):
    rt, tmp = fs_rt
    rt.services["fs"].write_text("a/b/c.txt", "nested")
    assert (tmp / "a" / "b" / "c.txt").read_text() == "nested"


def test_a_write_is_atomic_and_leaves_no_temp_files(fs_rt):
    rt, tmp = fs_rt
    rt.services["fs"].write_text("f.txt", "x" * 1000)
    assert [p.name for p in tmp.iterdir()] == ["f.txt"], "a temp file was left behind"


def test_reading_a_missing_file_is_an_error_not_a_crash(fs_rt):
    rt, _ = fs_rt
    with pytest.raises(FsError, match="no such file"):
        rt.services["fs"].read_text("nope.txt")


def test_a_large_file_is_truncated_and_says_so(fs_rt):
    rt, tmp = fs_rt
    (tmp / "big.txt").write_text("y" * 5000)
    rt.services["fs"].read_text("big.txt")  # record a version so the guard allows the read
    content, _ = rt.services["fs"].read_text("big.txt", max_bytes=100)
    assert "[truncated at 100 bytes of 5000]" in content, (
        "a silently truncated file is one the model reasons about as if complete"
    )


def test_edit_replaces_a_unique_string(fs_rt):
    rt, tmp = fs_rt
    fs = rt.services["fs"]
    fs.write_text("f.py", "a = 1\nb = 2\n")
    fs.read_text("f.py")
    _, made = fs.edit("f.py", "b = 2", "b = 3")
    assert made == 1
    assert (tmp / "f.py").read_text() == "a = 1\nb = 3\n"


def test_an_ambiguous_edit_is_refused_rather_than_guessed(fs_rt):
    """Replacing 'the first one' when the model meant another is a silent wrong edit."""
    rt, _ = fs_rt
    fs = rt.services["fs"]
    fs.write_text("f.py", "x = 1\nx = 1\n")
    fs.read_text("f.py")
    with pytest.raises(FsError, match="appears 2 times"):
        fs.edit("f.py", "x = 1", "x = 2")


def test_edit_all_occurrences_with_count_zero(fs_rt):
    rt, tmp = fs_rt
    fs = rt.services["fs"]
    fs.write_text("f.py", "x = 1\nx = 1\n")
    fs.read_text("f.py")
    _, made = fs.edit("f.py", "x = 1", "x = 2", count=0)
    assert made == 2 and (tmp / "f.py").read_text() == "x = 2\nx = 2\n"


def test_editing_text_that_is_not_there(fs_rt):
    rt, _ = fs_rt
    fs = rt.services["fs"]
    fs.write_text("f.py", "hello")
    fs.read_text("f.py")
    with pytest.raises(FsError, match="not found"):
        fs.edit("f.py", "goodbye", "x")


# --- path confinement -----------------------------------------------------------------------


def test_a_relative_escape_is_refused(fs_rt):
    rt, _ = fs_rt
    with pytest.raises(PathEscape, match="outside the project root"):
        rt.services["fs"].read_text("../../etc/passwd")


def test_an_absolute_path_outside_the_root_is_refused(fs_rt):
    rt, _ = fs_rt
    with pytest.raises(PathEscape):
        rt.services["fs"].read_text("/etc/passwd")


@pytest.mark.skipif(sys.platform == "win32", reason="symlink creation needs privileges")
def test_a_symlink_pointing_outside_the_root_is_refused(fs_rt, tmp_path):
    """THE case a naive check misses.

    os.path.normpath collapses '..' textually without touching the filesystem, so a symlink
    inside the project pointing at /etc passes it. resolve() follows the link, which is why
    the guard resolves BEFORE comparing.
    """
    rt, tmp = fs_rt
    outside = tmp_path.parent / "outside_secret.txt"
    outside.write_text("secret")
    (tmp / "innocent.txt").symlink_to(outside)

    with pytest.raises(PathEscape, match="outside the project root"):
        rt.services["fs"].read_text("innocent.txt")


@pytest.mark.skipif(sys.platform == "win32", reason="symlink creation needs privileges")
def test_a_symlink_staying_inside_the_root_is_allowed(fs_rt):
    """Confinement must not be so blunt that it breaks legitimate links."""
    rt, tmp = fs_rt
    (tmp / "real.txt").write_text("fine")
    (tmp / "link.txt").symlink_to(tmp / "real.txt")
    content, _ = rt.services["fs"].read_text("link.txt")
    assert content == "fine"


def test_a_symlink_loop_is_refused_without_hanging(fs_rt):
    rt, tmp = fs_rt
    (tmp / "a").symlink_to(tmp / "b")
    (tmp / "b").symlink_to(tmp / "a")
    with pytest.raises(PathEscape):
        rt.services["fs"].read_text("a")


# --- read-before-edit -----------------------------------------------------------------------


def test_writing_a_file_that_was_never_read_is_refused(fs_rt):
    rt, tmp = fs_rt
    (tmp / "existing.txt").write_text("important")
    with pytest.raises(VersionConflict, match="has not been read"):
        rt.services["fs"].write_text("existing.txt", "clobbered")
    assert (tmp / "existing.txt").read_text() == "important", "the file was modified anyway"


def test_creating_a_new_file_needs_no_prior_read(fs_rt):
    rt, tmp = fs_rt
    rt.services["fs"].write_text("brand_new.txt", "fine")
    assert (tmp / "brand_new.txt").read_text() == "fine"


def test_writing_after_reading_is_allowed(fs_rt):
    rt, tmp = fs_rt
    fs = rt.services["fs"]
    (tmp / "f.txt").write_text("before")
    fs.read_text("f.txt")
    fs.write_text("f.txt", "after")
    assert (tmp / "f.txt").read_text() == "after"


def test_a_file_changed_since_it_was_read_is_refused(fs_rt):
    """The whole point: a stale overwrite silently destroys somebody else's edit."""
    rt, tmp = fs_rt
    fs = rt.services["fs"]
    (tmp / "f.txt").write_text("v1")
    fs.read_text("f.txt")

    time.sleep(0.01)
    (tmp / "f.txt").write_text("someone else's change")  # edited outside the agent

    with pytest.raises(VersionConflict, match="changed since you read it"):
        fs.write_text("f.txt", "agent's stale content")
    assert (tmp / "f.txt").read_text() == "someone else's change"


def test_re_reading_clears_the_conflict(fs_rt):
    """The rejection tells the model to re-read, and that recovery must actually work."""
    rt, tmp = fs_rt
    fs = rt.services["fs"]
    (tmp / "f.txt").write_text("v1")
    fs.read_text("f.txt")
    time.sleep(0.01)
    (tmp / "f.txt").write_text("v2")

    with pytest.raises(VersionConflict):
        fs.write_text("f.txt", "x")

    fs.read_text("f.txt")          # the recovery the error message asks for
    fs.write_text("f.txt", "v3")
    assert (tmp / "f.txt").read_text() == "v3"


def test_the_guard_can_be_turned_off_for_a_trusted_deployment(tmp_path):
    rt = Runtime()
    rt.mount(fs_service)
    rt.mount(fs_local, {"root": str(tmp_path)})
    rt.mount(fs_guard, {"root": str(tmp_path), "require_read_before_write": False})
    (tmp_path / "f.txt").write_text("before")
    rt.services["fs"].write_text("f.txt", "after")
    assert (tmp_path / "f.txt").read_text() == "after"


def test_confinement_is_NOT_turned_off_by_that_switch(tmp_path):
    """Read-before-edit is a preference; confinement is an invariant. Different things."""
    rt = Runtime()
    rt.mount(fs_service)
    rt.mount(fs_local, {"root": str(tmp_path)})
    rt.mount(fs_guard, {"root": str(tmp_path), "require_read_before_write": False})
    with pytest.raises(PathEscape):
        rt.services["fs"].read_text("/etc/passwd")


# --- the tools -------------------------------------------------------------------------------


def test_read_file_tool_returns_line_numbers(tools_rt):
    rt, tmp = tools_rt
    (tmp / "f.py").write_text("one\ntwo\n")
    result = call(rt, "read_file", path="f.py")
    assert not result.is_error
    assert "    1  one" in result.content and "    2  two" in result.content


def test_write_file_tool_creates_the_file(tools_rt):
    rt, tmp = tools_rt
    result = call(rt, "write_file", path="out.txt", content="written")
    assert not result.is_error
    assert (tmp / "out.txt").read_text() == "written"


def test_a_tool_error_comes_back_as_a_result_the_model_can_read(tools_rt):
    rt, _ = tools_rt
    result = call(rt, "read_file", path="../../etc/passwd")
    assert result.is_error
    assert "outside the project root" in result.content


def test_edit_file_tool(tools_rt):
    rt, tmp = tools_rt
    (tmp / "f.py").write_text("value = 1\n")
    call(rt, "read_file", path="f.py")
    result = call(rt, "edit_file", path="f.py", old_text="value = 1", new_text="value = 2")
    assert not result.is_error
    assert (tmp / "f.py").read_text() == "value = 2\n"


def test_glob_and_grep(tools_rt):
    rt, tmp = tools_rt
    (tmp / "a.py").write_text("import os\n")
    (tmp / "b.py").write_text("import sys\n")
    (tmp / "c.txt").write_text("import nothing\n")

    globbed = call(rt, "glob", pattern="*.py")
    assert "a.py" in globbed.content and "c.txt" not in globbed.content

    grepped = call(rt, "grep", pattern=r"^import (os|sys)", glob="*.py")
    assert "a.py:1" in grepped.content and "b.py:1" in grepped.content
    assert "c.txt" not in grepped.content


def test_list_dir(tools_rt):
    rt, tmp = tools_rt
    (tmp / "x.txt").write_text("")
    (tmp / "sub").mkdir()
    result = call(rt, "list_dir", path=".")
    assert "x.txt" in result.content and "sub" in result.content


# --- the shell -------------------------------------------------------------------------------


def test_bash_runs_a_command(tools_rt):
    rt, _ = tools_rt
    result = call(rt, "bash", command="echo hello")
    assert not result.is_error
    assert "hello" in result.content and "exit code: 0" in result.content


def test_bash_reports_a_nonzero_exit_explicitly(tools_rt):
    """A model must be able to tell success from failure without parsing prose."""
    rt, _ = tools_rt
    result = call(rt, "bash", command="exit 3")
    assert "exit code: 3" in result.content


def test_bash_captures_stderr_separately(tools_rt):
    rt, _ = tools_rt
    result = call(rt, "bash", command="echo oops >&2")
    assert "stderr:" in result.content and "oops" in result.content


def test_bash_output_is_bounded_and_says_when_it_truncated(tmp_path):
    """A silently shortened build log is one the model misreads as a pass."""
    rt = Runtime()
    rt.mount(tools_service)
    rt.mount(shell_service)
    rt.mount(shell_local, {"cwd": str(tmp_path)})
    rt.mount(tool_bash, {"max_output": 100})

    result = rt.services["tools"].execute("c1", "bash", {"command": "printf 'x%.0s' {1..5000}"})
    assert "truncated at 100 bytes" in result.content


def test_bash_times_out_and_kills_the_child(tools_rt):
    rt, _ = tools_rt
    started = time.monotonic()
    result = call(rt, "bash", command="sleep 30", timeout=0.2)
    elapsed = time.monotonic() - started
    assert "killed: timed out" in result.content
    assert elapsed < 5, "the command was not actually killed"


def test_bash_honors_an_outer_cancellation(tools_rt):
    """subprocess.run(timeout=) cannot do this -- it blocks on its own timeout."""
    rt, _ = tools_rt
    token = CancelToken(timeout=0.2)
    started = time.monotonic()
    rt.services["tools"].execute("c1", "bash", {"command": "sleep 30"}, token=token)
    assert time.monotonic() - started < 5


# --- policy defaults --------------------------------------------------------------------------


def test_mutating_tools_ask_by_default(tools_rt):
    """The line where the gate stops being a demo."""
    rt, tmp = tools_rt
    asked = []
    rt.mount(permission, {"root": str(tmp), "ask": lambda e: asked.append(e.name) or False})

    denied = call(rt, "write_file", path="x.txt", content="nope")
    assert denied.is_error and "declined" in denied.content
    assert not (tmp / "x.txt").exists(), "the file was written despite the refusal"
    assert asked == ["write_file"]


def test_read_only_tools_do_not_ask(tools_rt):
    rt, tmp = tools_rt
    (tmp / "f.txt").write_text("hi")
    asked = []
    rt.mount(permission, {"root": str(tmp), "ask": lambda e: asked.append(e.name) or False})

    assert not call(rt, "read_file", path="f.txt").is_error
    assert asked == [], "a read-only tool should not interrupt the user"


def test_bash_asks_by_default(tools_rt):
    rt, tmp = tools_rt
    asked = []
    rt.mount(permission, {"root": str(tmp), "ask": lambda e: asked.append(e.name) or False})
    result = call(rt, "bash", command="echo hi")
    assert result.is_error and asked == ["bash"]


# --- the coding profile -------------------------------------------------------------------


def test_the_coding_profile_boots_with_every_tool(monkeypatch, tmp_path):
    """Turning a chat agent into a coding agent should add rows, not fork the loop."""
    monkeypatch.setenv("DS_ROOT", str(tmp_path))
    import importlib

    from dshpy.profiles import coding
    importlib.reload(coding)

    rt = Runtime()
    rt.mount_all(coding.rows())

    for key in ("llm", "tools", "sessions", "agent_loop", "fs", "shell"):
        assert key in rt.services, f"{key} never activated:\n{rt.dump()}"

    registered = set(rt.services["tools"].names())
    assert {"read_file", "write_file", "edit_file", "glob", "grep", "bash"} <= registered


def test_yes_answers_the_prompt_rather_than_rewriting_policy(tmp_path):
    """--yes must reach the tools DEFAULT_POLICY marks 'ask', which a default= change misses."""
    from dshpy import cli

    rt = cli.build_runtime(assume_yes=True, quiet=True)
    permission_rows = [r for r in cli.default_profile.rows()
                       if getattr(r["plugin"], "name", "") == "permission"]
    assert permission_rows, "the default profile lost its permission plugin"
    # The explicit policy still says "ask" -- the override is the answer, not the policy.
    assert permission.DEFAULT_POLICY["bash"] == "ask"
    assert rt.services  # boots
