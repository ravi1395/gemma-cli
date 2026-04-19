"""Tests for the ``run_linter`` built-in.

Rather than depend on ruff/mypy/eslint being installed on the test
machine, we monkey-patch ``shutil.which`` and the subprocess runner so
the tests are hermetic and fast.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gemma.tools.builtins import lint as _lint_mod
from gemma.tools.registry import get as _get
from gemma.tools.subprocess_runner import RunResult


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Chdir into an empty tmp workspace so safety resolves against it."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run_linter():
    _spec, handler = _get("run_linter")
    return handler


# ---------------------------------------------------------------------------
# Happy path (ruff found, runs, returns linter stdout)
# ---------------------------------------------------------------------------

def test_run_linter_returns_runner_output(workspace, monkeypatch):
    (workspace / "a.py").write_text("x = 1\n")

    # Pretend ruff is installed.
    monkeypatch.setattr(_lint_mod.shutil, "which", lambda exe: "/usr/bin/" + exe)

    captured = {}

    def fake_run(argv, cwd, timeout_s, max_output_bytes):
        captured["argv"] = argv
        return RunResult(
            exit_code=0, stdout="All good.\n", stderr="",
            duration_ms=12, timed_out=False, truncated=False,
            start_error=None,
        )

    monkeypatch.setattr(_lint_mod._runner, "run", fake_run)

    result = _run_linter()(lang="python", path="a.py")
    assert result.ok is True
    assert "All good." in result.content
    # The ``{path}`` template must be substituted with the resolved path.
    assert captured["argv"][0] == "ruff"
    assert captured["argv"][-1].endswith("a.py")


def test_run_linter_surfaces_nonzero_findings(workspace, monkeypatch):
    """A non-zero exit (lint findings) is a success, not an error."""
    (workspace / "a.py").write_text("")
    monkeypatch.setattr(_lint_mod.shutil, "which", lambda exe: "/usr/bin/" + exe)

    def fake_run(argv, cwd, timeout_s, max_output_bytes):
        return RunResult(
            exit_code=1,
            stdout="a.py:1: E501 line too long\n",
            stderr="",
            duration_ms=7, timed_out=False, truncated=False,
            start_error=None,
        )

    monkeypatch.setattr(_lint_mod._runner, "run", fake_run)

    result = _run_linter()(lang="python", path="a.py")
    assert result.ok is True  # linter findings are not a tool failure
    assert "E501" in result.content
    assert result.metadata["exit_code"] == 1


# ---------------------------------------------------------------------------
# Rejection: missing executable
# ---------------------------------------------------------------------------

def test_run_linter_missing_executable_returns_missing_tool(workspace, monkeypatch):
    (workspace / "a.py").write_text("")
    monkeypatch.setattr(_lint_mod.shutil, "which", lambda exe: None)

    result = _run_linter()(lang="python", path="a.py")
    assert result.ok is False
    assert result.error == "missing_tool"


# ---------------------------------------------------------------------------
# Rejection: unsafe path
# ---------------------------------------------------------------------------

def test_run_linter_rejects_path_escape(workspace, monkeypatch):
    # which() still reports true so we know the refusal happens before
    # the subprocess runner is consulted.
    monkeypatch.setattr(_lint_mod.shutil, "which", lambda exe: "/usr/bin/" + exe)

    result = _run_linter()(lang="python", path="../../etc/passwd")
    assert result.ok is False
    assert result.error == "safety"
