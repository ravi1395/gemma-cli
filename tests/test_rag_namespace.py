"""Tests for :func:`gemma.rag.namespace.resolve_namespace`.

We monkeypatch ``subprocess.run`` where necessary so these tests are
independent of whether the test harness is in a git repo.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from gemma.rag import namespace as _ns


# ---------------------------------------------------------------------------
# Determinism + shape
# ---------------------------------------------------------------------------

def test_namespace_is_deterministic(tmp_path):
    a = _ns.resolve_namespace(tmp_path, branch="main")
    b = _ns.resolve_namespace(tmp_path, branch="main")
    assert a == b


def test_namespace_shape_is_hash_colon_branch(tmp_path):
    result = _ns.resolve_namespace(tmp_path, branch="main")
    assert ":" in result
    root_hash, branch = result.split(":", 1)
    assert len(root_hash) == 12
    # Hash should be lowercase hex.
    assert all(c in "0123456789abcdef" for c in root_hash)
    assert branch == "main"


def test_namespace_differs_across_workspaces(tmp_path):
    other = tmp_path.parent / "other"
    other.mkdir(exist_ok=True)
    a = _ns.resolve_namespace(tmp_path, branch="main")
    b = _ns.resolve_namespace(other, branch="main")
    assert a != b  # different paths ⇒ different hashes


def test_namespace_differs_across_branches(tmp_path):
    a = _ns.resolve_namespace(tmp_path, branch="main")
    b = _ns.resolve_namespace(tmp_path, branch="feature/x")
    assert a != b


def test_namespace_relative_and_absolute_are_equivalent(tmp_path, monkeypatch):
    """./foo and /abs/path/foo should resolve to the same hash."""
    monkeypatch.chdir(tmp_path.parent)
    rel = Path(tmp_path.name)  # e.g. "tmp_path_name"
    ns_rel = _ns.resolve_namespace(rel, branch="main")
    ns_abs = _ns.resolve_namespace(tmp_path, branch="main")
    assert ns_rel == ns_abs


# ---------------------------------------------------------------------------
# Branch sanitisation
# ---------------------------------------------------------------------------

def test_sanitise_branch_strips_unsafe_chars():
    assert _ns._sanitize_branch("feature/auth v2") == "feature_auth_v2"
    assert _ns._sanitize_branch("ok.branch-1") == "ok.branch-1"
    assert _ns._sanitize_branch("bad:colon") == "bad_colon"


# ---------------------------------------------------------------------------
# Branch detection
# ---------------------------------------------------------------------------

def _fake_run_returning(stdout: str, returncode: int = 0):
    """Build a fake ``subprocess.run`` that returns the given result."""
    def _run(*args, **kwargs):
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")
    return _run


def test_detect_branch_uses_git_output(tmp_path, monkeypatch):
    monkeypatch.setattr(_ns.subprocess, "run", _fake_run_returning("main\n"))
    assert _ns._detect_branch(tmp_path) == "main"


def test_detect_branch_returns_none_for_detached_head(tmp_path, monkeypatch):
    monkeypatch.setattr(_ns.subprocess, "run", _fake_run_returning("HEAD\n"))
    assert _ns._detect_branch(tmp_path) is None


def test_detect_branch_returns_none_for_non_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(
        _ns.subprocess, "run", _fake_run_returning("", returncode=128),
    )
    assert _ns._detect_branch(tmp_path) is None


def test_detect_branch_handles_missing_git(tmp_path, monkeypatch):
    def _boom(*a, **kw):
        raise FileNotFoundError("git not installed")
    monkeypatch.setattr(_ns.subprocess, "run", _boom)
    assert _ns._detect_branch(tmp_path) is None


def test_resolve_falls_back_to_default_when_no_branch(tmp_path, monkeypatch):
    monkeypatch.setattr(_ns.subprocess, "run", _fake_run_returning("", returncode=128))
    ns = _ns.resolve_namespace(tmp_path)  # no explicit branch
    assert ns.endswith(":_default")
