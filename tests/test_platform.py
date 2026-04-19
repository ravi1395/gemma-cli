"""Tests for gemma.platform — OS / shell / terminal detection.

These tests poke at env-var and monkeypatch-driven detection paths. They
do **not** assume the test runner's real OS, so the suite is portable
across macOS, Linux, and WSL CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from gemma import platform as plat


# ---------------------------------------------------------------------------
# OS detection
# ---------------------------------------------------------------------------

def test_detect_os_wsl_via_distro_name(monkeypatch):
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu-22.04")
    # Even on a Mac test runner, WSL env var should win.
    monkeypatch.setattr(sys, "platform", "linux")
    assert plat.detect_os() is plat.OS.WSL


def test_detect_os_wsl_via_interop(monkeypatch):
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.setenv("WSL_INTEROP", "/run/WSL/1_interop")
    monkeypatch.setattr(sys, "platform", "linux")
    assert plat.detect_os() is plat.OS.WSL


def test_detect_os_macos(monkeypatch):
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.delenv("WSL_INTEROP", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    assert plat.detect_os() is plat.OS.MACOS


def test_detect_os_linux(monkeypatch):
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.delenv("WSL_INTEROP", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    assert plat.detect_os() is plat.OS.LINUX


def test_detect_os_windows(monkeypatch):
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.delenv("WSL_INTEROP", raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    assert plat.detect_os() is plat.OS.WINDOWS


def test_detect_os_unknown_platform(monkeypatch):
    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.delenv("WSL_INTEROP", raising=False)
    monkeypatch.setattr(sys, "platform", "emscripten")
    assert plat.detect_os() is plat.OS.UNKNOWN


# ---------------------------------------------------------------------------
# Shell detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "shell_path, expected",
    [
        ("/bin/bash", plat.Shell.BASH),
        ("/usr/local/bin/zsh", plat.Shell.ZSH),
        ("/opt/homebrew/bin/fish", plat.Shell.FISH),
        ("/usr/bin/pwsh", plat.Shell.POWERSHELL),
        ("C:\\Program Files\\PowerShell\\7\\pwsh.exe", plat.Shell.POWERSHELL),
        ("/bin/tcsh", plat.Shell.UNKNOWN),
        ("", plat.Shell.UNKNOWN),
    ],
)
def test_detect_shell_from_shell_env(shell_path, expected):
    assert plat.detect_shell(shell_env=shell_path) is expected


def test_detect_shell_defaults_to_env(monkeypatch):
    monkeypatch.setenv("SHELL", "/bin/zsh")
    monkeypatch.delenv("PSModulePath", raising=False)
    assert plat.detect_shell() is plat.Shell.ZSH


def test_detect_shell_powershell_via_psmodulepath(monkeypatch):
    monkeypatch.delenv("SHELL", raising=False)
    monkeypatch.setenv("PSModulePath", "C:\\Program Files\\PowerShell\\7\\Modules")
    assert plat.detect_shell() is plat.Shell.POWERSHELL


def test_detect_shell_unknown_when_env_missing(monkeypatch):
    monkeypatch.delenv("SHELL", raising=False)
    monkeypatch.delenv("PSModulePath", raising=False)
    assert plat.detect_shell() is plat.Shell.UNKNOWN


# ---------------------------------------------------------------------------
# Terminal environment
# ---------------------------------------------------------------------------

def test_is_ssh_true_when_ssh_tty_set(monkeypatch):
    monkeypatch.setenv("SSH_TTY", "/dev/pts/0")
    assert plat.is_ssh() is True


def test_is_ssh_true_when_ssh_connection_set(monkeypatch):
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.setenv("SSH_CONNECTION", "1.2.3.4 22 5.6.7.8 22")
    assert plat.is_ssh() is True


def test_is_ssh_false_when_neither_set(monkeypatch):
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    assert plat.is_ssh() is False


def test_is_tty_handles_broken_stdout(monkeypatch):
    """Captured-stdout harnesses may not expose isatty; never raise."""

    class _Stub:
        def isatty(self):
            raise ValueError("not a real stdout")

    monkeypatch.setattr(sys, "stdout", _Stub())
    assert plat.is_tty() is False


# ---------------------------------------------------------------------------
# rc_file_for
# ---------------------------------------------------------------------------

def test_rc_file_bash_macos():
    path = plat.rc_file_for(plat.Shell.BASH, os_=plat.OS.MACOS)
    assert path == Path.home() / ".bash_profile"


def test_rc_file_bash_linux():
    path = plat.rc_file_for(plat.Shell.BASH, os_=plat.OS.LINUX)
    assert path == Path.home() / ".bashrc"


def test_rc_file_zsh_is_home_zshrc():
    assert plat.rc_file_for(plat.Shell.ZSH, os_=plat.OS.LINUX) == Path.home() / ".zshrc"


def test_rc_file_fish_uses_completions_dir():
    path = plat.rc_file_for(plat.Shell.FISH, os_=plat.OS.LINUX)
    assert path == Path.home() / ".config" / "fish" / "completions" / "gemma.fish"


def test_rc_file_powershell():
    path = plat.rc_file_for(plat.Shell.POWERSHELL, os_=plat.OS.WINDOWS)
    assert path is not None
    assert path.suffix == ".ps1"


def test_rc_file_unknown_shell_returns_none():
    assert plat.rc_file_for(plat.Shell.UNKNOWN, os_=plat.OS.LINUX) is None


# ---------------------------------------------------------------------------
# describe()
# ---------------------------------------------------------------------------

def test_describe_is_json_safe():
    import json

    snapshot = plat.describe()
    # Must serialise — this is the promise that lets `gemma ... status`
    # emit --json output without a custom encoder.
    json.dumps(snapshot)
    assert {"os", "shell", "tty", "ssh", "os_release"}.issubset(snapshot)
