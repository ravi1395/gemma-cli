"""OS, shell, and terminal-environment detection for gemma-cli.

Consolidates every ``os.name`` / ``$SHELL`` / ``$SSH_TTY`` probe the rest of
the codebase needs so features like clipboard integration (Phase 6.4) and
shell completions (Phase 6.3) don't each re-roll their own detection logic.

Design notes
------------
* The detection functions are **pure** and **cheap** — each reads environment
  variables and returns an enum. They are safe to call repeatedly; no
  caching is needed at this layer.
* ``detect_backend_for_clipboard()`` and other backend probes live in their
  respective modules (e.g. ``gemma/clipboard.py``) — this file answers
  *where am I running*, not *what tools are installed*.
* Windows and WSL are reported as distinct ``OS`` values because their
  clipboard and rc-file behaviour diverges, even though both are "Windows-
  adjacent".
* ``rc_file_for`` returns a default rc file per shell. Bash is particularly
  messy: macOS prefers ``~/.bash_profile``, Linux prefers ``~/.bashrc``.
  Callers that want to do something more sophisticated can inspect the
  returned ``Path`` and fall back themselves.
"""

from __future__ import annotations

import os
import platform as _stdlib_platform
import sys
from enum import Enum
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class OS(str, Enum):
    """Coarse-grained host operating system.

    WSL is distinct from LINUX because:
      * clipboard: WSL uses ``clip.exe`` (a Windows binary), whereas Linux
        uses ``xclip``/``xsel``/``wl-copy``;
      * paths to a Windows host clipboard need Windows-style translation.
    """

    MACOS = "macos"
    LINUX = "linux"
    WSL = "wsl"
    WINDOWS = "windows"
    UNKNOWN = "unknown"


class Shell(str, Enum):
    """The user's interactive shell, as inferred from ``$SHELL``.

    ``UNKNOWN`` means we could not determine the shell — callers should
    either degrade gracefully (completions: ask the user) or fall back to
    a sensible default (clipboard: fine, this enum is irrelevant there).
    """

    BASH = "bash"
    ZSH = "zsh"
    FISH = "fish"
    POWERSHELL = "powershell"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# OS detection
# ---------------------------------------------------------------------------

def detect_os() -> OS:
    """Return the coarse-grained host operating system.

    WSL is detected via the ``WSL_DISTRO_NAME`` / ``WSL_INTEROP`` env vars
    that Microsoft's init injects — more reliable than parsing ``uname -r``.
    """
    # Explicit WSL check first: WSL reports ``sys.platform == "linux"`` so
    # we'd miss it otherwise.
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return OS.WSL

    plat = sys.platform
    if plat == "darwin":
        return OS.MACOS
    if plat.startswith("linux"):
        return OS.LINUX
    if plat in ("win32", "cygwin"):
        return OS.WINDOWS
    return OS.UNKNOWN


def os_release() -> str:
    """Return a human-readable OS release string, for diagnostics only.

    Never used for logic decisions — this is the string that goes into
    ``gemma clipboard status`` output so users can file useful bug reports.
    """
    return f"{_stdlib_platform.system()} {_stdlib_platform.release()}"


# ---------------------------------------------------------------------------
# Shell detection
# ---------------------------------------------------------------------------

def detect_shell(shell_env: Optional[str] = None) -> Shell:
    """Return the user's interactive shell.

    Args:
        shell_env: Optional override for ``$SHELL``. Exposed so tests can
            inject values without mutating the real process environment.

    Returns:
        ``Shell.UNKNOWN`` if ``$SHELL`` is unset or unrecognised.
    """
    raw = shell_env if shell_env is not None else os.environ.get("SHELL", "")
    if not raw:
        # No $SHELL on Windows in PowerShell sessions; probe $PSModulePath.
        if os.environ.get("PSModulePath"):
            return Shell.POWERSHELL
        return Shell.UNKNOWN

    # $SHELL is a path to the shell binary. We only care about the basename.
    # Split on both POSIX and Windows separators because ``Path`` on Linux
    # won't recognise ``\`` as a separator when a user hands us a Windows
    # path (e.g. PowerShell tests running on a Linux CI box).
    base = raw.replace("\\", "/").rsplit("/", 1)[-1].lower()
    if base == "bash":
        return Shell.BASH
    if base == "zsh":
        return Shell.ZSH
    if base == "fish":
        return Shell.FISH
    if base in ("pwsh", "pwsh.exe", "powershell", "powershell.exe"):
        return Shell.POWERSHELL
    return Shell.UNKNOWN


# ---------------------------------------------------------------------------
# Terminal environment
# ---------------------------------------------------------------------------

def is_tty() -> bool:
    """True iff stdout is attached to an interactive terminal.

    Used by features that want to prompt the user before mutating state.
    Non-TTY pipelines (``gemma ask --copy | pbcopy``) must *not* prompt.
    """
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):
        # Some captured-stdout test harnesses raise ValueError on isatty.
        return False


def is_ssh() -> bool:
    """True iff we appear to be running inside an SSH session.

    We check ``$SSH_TTY`` first (most reliable — only set when a TTY was
    allocated), then fall back to ``$SSH_CONNECTION``. OSC52 clipboard
    writes and similar terminal tricks rely on this.
    """
    return bool(os.environ.get("SSH_TTY") or os.environ.get("SSH_CONNECTION"))


# ---------------------------------------------------------------------------
# RC file resolution
# ---------------------------------------------------------------------------

def rc_file_for(shell: Shell, os_: Optional[OS] = None) -> Optional[Path]:
    """Return the default rc file for ``shell`` on ``os_``.

    Args:
        shell: The target shell.
        os_:   The host OS. Defaults to :func:`detect_os()`. Only affects
               bash, where macOS and Linux use different files by tradition.

    Returns:
        A :class:`pathlib.Path` to the rc file (not guaranteed to exist —
        installers are expected to create it), or ``None`` for shells we
        don't know how to target automatically (``Shell.UNKNOWN``).

    Rationale: the fish completion path is the per-completion directory,
    not ``config.fish``, because fish auto-loads that directory and this
    yields an idempotent, self-contained install with no rc-file edits.
    """
    host = os_ if os_ is not None else detect_os()
    home = Path.home()

    if shell is Shell.BASH:
        # macOS Terminal starts login shells, so ``~/.bash_profile`` is
        # sourced but ``~/.bashrc`` is not. Linux typically runs interactive
        # non-login shells and reads ``~/.bashrc``.
        if host is OS.MACOS:
            return home / ".bash_profile"
        return home / ".bashrc"

    if shell is Shell.ZSH:
        return home / ".zshrc"

    if shell is Shell.FISH:
        # Completions directory is auto-loaded. Using it avoids editing
        # config.fish entirely.
        return home / ".config" / "fish" / "completions" / "gemma.fish"

    if shell is Shell.POWERSHELL:
        # ``$PROFILE`` on Windows resolves to
        # ``Documents/PowerShell/Microsoft.PowerShell_profile.ps1``.
        # We emit that default; users with non-default profile paths can
        # use ``gemma completion print`` instead.
        return (
            home
            / "Documents"
            / "PowerShell"
            / "Microsoft.PowerShell_profile.ps1"
        )

    return None


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def describe() -> dict[str, str]:
    """Snapshot of the detection results, for ``gemma ... status`` commands.

    Returns stringified values (not enums) so the dict can be dumped to
    JSON without a custom encoder.
    """
    return {
        "os": detect_os().value,
        "os_release": os_release(),
        "shell": detect_shell().value,
        "tty": str(is_tty()),
        "ssh": str(is_ssh()),
    }
