"""Tests for gemma.clipboard — backend probe, copy pipeline, redaction gate.

All subprocess calls are mocked so the suite runs unchanged in CI where
no clipboard tool is present. Probe caching is reset between tests so
one test's choice of backend doesn't leak into the next.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from typing import List

import pytest

from gemma import clipboard
from gemma import platform as plat


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_probe_cache():
    """Clipboard probe is memoised — reset before/after every test."""
    clipboard._reset_cache()
    yield
    clipboard._reset_cache()


@dataclass
class RunCall:
    """Record of a mocked ``subprocess.run`` invocation."""
    argv: List[str]
    input: bytes
    timeout: float | None


@pytest.fixture
def captured_runs(monkeypatch) -> List[RunCall]:
    """Capture every ``subprocess.run`` call the clipboard module makes.

    Replaces the real function with a no-op that records args and
    returns a ``CompletedProcess`` with rc=0. Tests that want to
    exercise error paths override this per-test with their own patch.
    """
    calls: List[RunCall] = []

    def _fake_run(argv, **kwargs):
        calls.append(RunCall(
            argv=list(argv),
            input=kwargs.get("input", b""),
            timeout=kwargs.get("timeout"),
        ))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(clipboard.subprocess, "run", _fake_run)
    return calls


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

def test_macos_selects_pbcopy(monkeypatch, captured_runs):
    monkeypatch.setattr(plat, "detect_os", lambda: plat.OS.MACOS)
    monkeypatch.setattr(clipboard.shutil, "which", lambda n: f"/usr/bin/{n}")

    result = clipboard.copy("hello")

    assert result.ok
    assert result.backend == "pbcopy"
    assert len(captured_runs) == 1
    assert captured_runs[0].argv == ["/usr/bin/pbcopy"]
    assert captured_runs[0].input == b"hello"


def test_linux_wayland_prefers_wl_copy(monkeypatch, captured_runs):
    monkeypatch.setattr(plat, "detect_os", lambda: plat.OS.LINUX)
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setattr(
        clipboard.shutil, "which",
        lambda n: "/usr/bin/wl-copy" if n == "wl-copy" else None,
    )

    result = clipboard.copy("hi")
    assert result.ok
    assert result.backend == "wl-copy"


def test_linux_x11_falls_back_to_xclip_then_xsel(monkeypatch, captured_runs):
    monkeypatch.setattr(plat, "detect_os", lambda: plat.OS.LINUX)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")

    # wl-copy skipped (Wayland var not set); xclip present.
    monkeypatch.setattr(
        clipboard.shutil, "which",
        lambda n: "/usr/bin/xclip" if n == "xclip" else None,
    )
    result = clipboard.copy("hi")
    assert result.ok
    assert result.backend == "xclip"
    assert captured_runs[0].argv == ["/usr/bin/xclip", "-selection", "clipboard"]


def test_linux_x11_uses_xsel_when_xclip_missing(monkeypatch, captured_runs):
    monkeypatch.setattr(plat, "detect_os", lambda: plat.OS.LINUX)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(
        clipboard.shutil, "which",
        lambda n: "/usr/bin/xsel" if n == "xsel" else None,
    )
    result = clipboard.copy("hi")
    assert result.ok
    assert result.backend == "xsel"


def test_wsl_selects_clip_exe(monkeypatch, captured_runs):
    monkeypatch.setattr(plat, "detect_os", lambda: plat.OS.WSL)
    monkeypatch.setattr(
        clipboard.shutil, "which",
        lambda n: "/mnt/c/Windows/System32/clip.exe" if n == "clip.exe" else None,
    )
    result = clipboard.copy("hi\n")
    assert result.ok
    assert result.backend == "clip.exe"
    # clip.exe trailing-newline quirk: single trailing \n is stripped.
    assert captured_runs[0].input == b"hi"


def test_no_backend_returns_soft_failure(monkeypatch):
    monkeypatch.setattr(plat, "detect_os", lambda: plat.OS.UNKNOWN)
    monkeypatch.setattr(clipboard.shutil, "which", lambda _n: None)
    # Force pyperclip import failure by poisoning sys.modules.
    monkeypatch.setitem(sys.modules, "pyperclip", None)

    result = clipboard.copy("hello")
    assert result.ok is False
    assert result.backend is None
    assert "no clipboard backend" in (result.reason or "")


# ---------------------------------------------------------------------------
# Redaction gate
# ---------------------------------------------------------------------------

_FAKE_AWS_KEY = "AKIAABCDEFGHIJKLMNOP"


def test_secret_in_text_refused_by_default(monkeypatch):
    monkeypatch.setattr(plat, "detect_os", lambda: plat.OS.MACOS)
    monkeypatch.setattr(clipboard.shutil, "which", lambda n: f"/usr/bin/{n}")

    result = clipboard.copy(f"use this: {_FAKE_AWS_KEY}")

    assert result.ok is False
    assert "refused" in (result.reason or "")
    assert any(f.type == "AWS_ACCESS_KEY" for f in result.redaction_findings)
    assert result.backend is None  # never hit the subprocess


def test_secret_copied_when_allow_secrets_true(monkeypatch, captured_runs):
    monkeypatch.setattr(plat, "detect_os", lambda: plat.OS.MACOS)
    monkeypatch.setattr(clipboard.shutil, "which", lambda n: f"/usr/bin/{n}")

    result = clipboard.copy(f"use this: {_FAKE_AWS_KEY}", allow_secrets=True)

    assert result.ok is True
    # Findings still reported so the caller can warn.
    assert any(f.type == "AWS_ACCESS_KEY" for f in result.redaction_findings)
    # Raw secret should have reached the pipe.
    assert _FAKE_AWS_KEY.encode() in captured_runs[0].input


# ---------------------------------------------------------------------------
# Subprocess error translation
# ---------------------------------------------------------------------------

def test_timeout_translated_to_soft_failure(monkeypatch):
    monkeypatch.setattr(plat, "detect_os", lambda: plat.OS.MACOS)
    monkeypatch.setattr(clipboard.shutil, "which", lambda n: f"/usr/bin/{n}")

    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="pbcopy", timeout=5)

    monkeypatch.setattr(clipboard.subprocess, "run", _raise_timeout)

    result = clipboard.copy("hi")
    assert result.ok is False
    assert "timed out" in (result.reason or "")
    assert result.backend == "pbcopy"


def test_nonzero_exit_translated_to_soft_failure(monkeypatch):
    monkeypatch.setattr(plat, "detect_os", lambda: plat.OS.MACOS)
    monkeypatch.setattr(clipboard.shutil, "which", lambda n: f"/usr/bin/{n}")

    def _raise_cpe(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd="pbcopy")

    monkeypatch.setattr(clipboard.subprocess, "run", _raise_cpe)

    result = clipboard.copy("hi")
    assert result.ok is False
    assert "exited non-zero" in (result.reason or "")


# ---------------------------------------------------------------------------
# describe() snapshot
# ---------------------------------------------------------------------------

def test_describe_is_json_safe(monkeypatch):
    import json

    monkeypatch.setattr(plat, "detect_os", lambda: plat.OS.MACOS)
    monkeypatch.setattr(clipboard.shutil, "which", lambda n: f"/usr/bin/{n}")

    snapshot = clipboard.describe()
    # Must serialise cleanly.
    json.dumps(snapshot)
    assert snapshot["selected"] == "pbcopy"
    assert snapshot["os"] == "macos"
    # Probe log contains at least the selected entry.
    names = {entry["backend"] for entry in snapshot["probe_log"]}
    assert "pbcopy" in names


def test_describe_reports_no_backend_cleanly(monkeypatch):
    monkeypatch.setattr(plat, "detect_os", lambda: plat.OS.UNKNOWN)
    monkeypatch.setattr(clipboard.shutil, "which", lambda _n: None)
    monkeypatch.setitem(sys.modules, "pyperclip", None)

    snapshot = clipboard.describe()
    assert snapshot["selected"] is None
    # Every probe entry recorded as skipped, with a reason.
    assert all(entry["ok"] is False for entry in snapshot["probe_log"])


# ---------------------------------------------------------------------------
# Probe cache semantics
# ---------------------------------------------------------------------------

def test_probe_cached_after_first_call(monkeypatch):
    monkeypatch.setattr(plat, "detect_os", lambda: plat.OS.MACOS)

    calls: list[str] = []

    def _counting_which(name):
        calls.append(name)
        return f"/usr/bin/{name}"

    monkeypatch.setattr(clipboard.shutil, "which", _counting_which)

    clipboard.detect_backend()
    clipboard.detect_backend()
    clipboard.detect_backend()

    # Only the first call runs the probe; subsequent calls use the cache.
    assert calls == ["pbcopy"]
