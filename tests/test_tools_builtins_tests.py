"""Tests for the ``run_tests`` built-in.

We patch the subprocess runner so the tests are hermetic; ``_detect_runner``
is exercised against a tmp workspace by dropping marker files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gemma.tools.builtins import tests as _t
from gemma.tools.registry import get as _get
from gemma.tools.subprocess_runner import RunResult


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run_tests():
    _spec, handler = _get("run_tests")
    return handler


# ---------------------------------------------------------------------------
# _detect_runner — the marker-file heuristic
# ---------------------------------------------------------------------------

def test_detect_runner_prefers_pytest(workspace):
    (workspace / "pyproject.toml").write_text("")
    (workspace / "package.json").write_text("{}")
    # pytest markers are checked first, so this is the expected winner.
    assert _t._detect_runner(workspace) == "pytest"


def test_detect_runner_falls_through_to_npm(workspace):
    (workspace / "package.json").write_text("{}")
    assert _t._detect_runner(workspace) == "npm"


def test_detect_runner_returns_none_for_empty_dir(workspace):
    assert _t._detect_runner(workspace) is None


# ---------------------------------------------------------------------------
# Happy path — pytest summary extraction
# ---------------------------------------------------------------------------

def test_run_tests_happy_path_returns_summary(workspace, monkeypatch):
    (workspace / "pyproject.toml").write_text("")
    monkeypatch.setattr(_t.shutil, "which", lambda exe: "/usr/bin/" + exe)

    stdout = (
        "collected 5 items\n"
        "tests/test_a.py ....\n"
        "tests/test_b.py .\n"
        "============== 5 passed in 0.12s ==============\n"
    )

    def fake_run(argv, cwd, timeout_s, max_output_bytes):
        return RunResult(
            exit_code=0, stdout=stdout, stderr="",
            duration_ms=120, timed_out=False, truncated=False,
            start_error=None,
        )

    monkeypatch.setattr(_t._runner, "run", fake_run)

    result = _run_tests()()
    assert result.ok is True
    assert result.metadata["runner"] == "pytest"
    # Summary extraction keeps the tail lines around; the summary line
    # must be present in the returned content.
    assert "passed" in result.content


def test_run_tests_failure_surfaces_as_tests_failed(workspace, monkeypatch):
    (workspace / "pyproject.toml").write_text("")
    monkeypatch.setattr(_t.shutil, "which", lambda exe: "/usr/bin/" + exe)

    def fake_run(argv, cwd, timeout_s, max_output_bytes):
        return RunResult(
            exit_code=1,
            stdout="============== 1 failed, 4 passed in 0.4s ==============\n",
            stderr="", duration_ms=400, timed_out=False, truncated=False,
            start_error=None,
        )

    monkeypatch.setattr(_t._runner, "run", fake_run)

    result = _run_tests()()
    assert result.ok is False
    assert result.error == "tests_failed"


def test_run_tests_include_logs_returns_full_output(workspace, monkeypatch):
    (workspace / "pyproject.toml").write_text("")
    monkeypatch.setattr(_t.shutil, "which", lambda exe: "/usr/bin/" + exe)

    full = "\n".join(f"line {i}" for i in range(100))

    def fake_run(argv, cwd, timeout_s, max_output_bytes):
        return RunResult(
            exit_code=0, stdout=full, stderr="",
            duration_ms=1, timed_out=False, truncated=False,
            start_error=None,
        )

    monkeypatch.setattr(_t._runner, "run", fake_run)

    summary = _run_tests()(include_logs=False)
    logs = _run_tests()(include_logs=True)
    # With include_logs=True we get every line; summary mode gets a tail.
    assert len(logs.content.splitlines()) > len(summary.content.splitlines())


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------

def test_run_tests_no_runner_detected(workspace):
    """Empty workspace → no runner marker → refused."""
    result = _run_tests()()
    assert result.ok is False
    assert result.error == "no_runner"


def test_run_tests_missing_runner_executable(workspace, monkeypatch):
    (workspace / "pyproject.toml").write_text("")
    monkeypatch.setattr(_t.shutil, "which", lambda exe: None)
    result = _run_tests()()
    assert result.ok is False
    assert result.error == "missing_tool"


def test_run_tests_rejects_path_target_escape(workspace, monkeypatch):
    """A target that contains a slash is safety-checked before spawn."""
    (workspace / "pyproject.toml").write_text("")
    monkeypatch.setattr(_t.shutil, "which", lambda exe: "/usr/bin/" + exe)
    result = _run_tests()(target="../../etc/passwd::test_x")
    assert result.ok is False
    assert result.error == "safety"
