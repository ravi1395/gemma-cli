"""Tests for the hardened subprocess runner.

Covers env scrub, timeout kill, output cap, missing-executable handling.
Each test uses a tiny inline Python script invoked via ``sys.executable``
so the behaviour is portable and doesn't depend on specific CLI tools.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from gemma.tools import subprocess_runner as _r


def _py_script(body: str) -> list[str]:
    """Build an argv that runs ``body`` as a Python one-liner."""
    return [sys.executable, "-c", body]


def test_happy_path_captures_stdout_and_exit(tmp_path):
    result = _r.run(
        _py_script("print('hello'); import sys; sys.exit(0)"),
        cwd=tmp_path,
        timeout_s=5,
        max_output_bytes=1024,
    )
    assert result.exit_code == 0
    assert "hello" in result.stdout
    assert result.timed_out is False
    assert result.truncated is False


def test_nonzero_exit_surfaced(tmp_path):
    result = _r.run(
        _py_script("import sys; sys.exit(42)"),
        cwd=tmp_path, timeout_s=5, max_output_bytes=1024,
    )
    assert result.exit_code == 42
    assert result.timed_out is False


def test_stderr_captured(tmp_path):
    result = _r.run(
        _py_script("import sys; sys.stderr.write('oops\\n'); sys.exit(1)"),
        cwd=tmp_path, timeout_s=5, max_output_bytes=1024,
    )
    assert result.exit_code == 1
    assert "oops" in result.stderr


def test_timeout_kills_process(tmp_path):
    result = _r.run(
        _py_script("import time; time.sleep(30)"),
        cwd=tmp_path,
        timeout_s=0.5,
        max_output_bytes=1024,
    )
    assert result.timed_out is True
    assert result.exit_code == -2


def test_output_cap_truncates_with_marker(tmp_path):
    big = "A" * 20_000
    result = _r.run(
        _py_script(f"print('{big}')"),
        cwd=tmp_path,
        timeout_s=5,
        max_output_bytes=1000,
    )
    assert result.truncated is True
    assert "truncated" in result.stdout


def test_missing_executable_reports_start_error(tmp_path):
    result = _r.run(
        ["/definitely/does/not/exist/xyzzy"],
        cwd=tmp_path, timeout_s=5, max_output_bytes=1024,
    )
    assert result.exit_code == -1
    assert result.start_error is not None


def test_env_scrub_removes_credentials(monkeypatch, tmp_path):
    # Secret-carrying names that must not reach the child.
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "attack")
    monkeypatch.setenv("OLLAMA_API_KEY", "attack")
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))

    result = _r.run(
        _py_script(
            "import os, json; "
            "print(json.dumps({k: os.environ.get(k, '') "
            "for k in ['AWS_SECRET_ACCESS_KEY', 'OLLAMA_API_KEY', 'PATH']}))"
        ),
        cwd=tmp_path, timeout_s=5, max_output_bytes=1024,
    )
    assert result.exit_code == 0
    import json
    got = json.loads(result.stdout.strip())
    assert got["AWS_SECRET_ACCESS_KEY"] == ""
    assert got["OLLAMA_API_KEY"] == ""
    # PATH survives because it's on the allowlist.
    assert got["PATH"] != ""


def test_build_env_only_contains_allowlisted(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "nope")
    env = _r.build_env(["PATH"])
    assert env == {"PATH": "/usr/bin"}
