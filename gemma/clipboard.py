"""Cross-platform clipboard integration for gemma-cli.

Copying the output of `gemma sh`, `gemma commit`, `gemma diff`, etc. to
the system clipboard is one of the highest-leverage UX wins for a
terminal assistant — it removes the mouse from the "ask, copy, paste"
loop entirely. This module is the single entry point: everything
clipboard-related funnels through :func:`copy`.

Design
------
The module is built around a **backend probe chain**. At first call we
walk a platform-ordered list of external tools (``pbcopy``, ``wl-copy``,
``xclip``, ``xsel``, ``clip.exe``) and pick the first one available.
The result is cached for the life of the process — clipboard preference
does not change mid-run. If none of the external tools are present we
fall back to the cross-platform :mod:`pyperclip` package if installed;
if not, we return a failure result rather than raising, so calling
commands can soft-fail without aborting the whole invocation.

Safety
------
Clipboard content is routed through :mod:`gemma.redaction` before
leaving the process. If any secret pattern is matched:

  * the default behaviour is to refuse the copy and return a
    :class:`CopyResult` with ``ok=False`` and populated
    ``redaction_findings``,
  * the caller (CLI flag ``--allow-secrets``) can override to force the
    raw copy.

This is the same posture as the rest of the codebase — we do not
persist secrets silently, and the clipboard is effectively persistence
once the user pastes into Slack / GitHub / a PR.

Public API
----------
``copy(text, *, allow_secrets=False) -> CopyResult``
  The primary entry point. Pure w.r.t. logging — the caller decides
  what to print.
``describe() -> dict[str, str]``
  Introspection helper used by ``gemma clipboard status``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from gemma import platform as _platform
from gemma.redaction import RedactionFinding, redact


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CopyResult:
    """Outcome of a :func:`copy` call.

    ``ok=True`` means the text (possibly redacted) reached the clipboard.
    ``ok=False`` means the copy did not happen; ``reason`` explains why
    so callers can log something useful to the user.

    ``redaction_findings`` is populated whether or not the copy succeeded
    — the caller should always inspect it so the UX can warn "we
    redacted 2 secrets" even on success.
    """

    ok: bool
    backend: Optional[str] = None
    reason: Optional[str] = None
    redaction_findings: List[RedactionFinding] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Backend descriptors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Backend:
    """Internal description of a single clipboard backend.

    ``write`` is a function that takes the text and either completes
    normally or raises. We wrap subprocess backends behind a common
    callable so ``_try_copy`` is backend-agnostic.
    """

    name: str
    write: Callable[[str], None]


def _run_subprocess_copy(argv: List[str], text: str, *, strip_trailing_newline: bool = False) -> None:
    """Pipe ``text`` to ``argv`` via stdin.

    Parameters
    ----------
    argv:
        The already-resolved command list (first element is an absolute
        path — we resolve via :func:`shutil.which` before building the
        Backend, not here, so no PATH search happens at copy time).
    text:
        The content to copy. Encoded as UTF-8.
    strip_trailing_newline:
        ``clip.exe`` on Windows/WSL adds or preserves trailing newlines
        inconsistently — strip a single trailing ``\\n`` for that backend
        so the pasted content matches what the user sees in the panel.
    """
    payload = text
    if strip_trailing_newline and payload.endswith("\n"):
        payload = payload[:-1]

    # 5-second timeout is generous; these tools complete in milliseconds.
    subprocess.run(
        argv,
        input=payload.encode("utf-8"),
        check=True,
        timeout=5,
        # Swallow any stderr output so "no selection owner" noise from
        # xclip on headless runs doesn't pollute the user's terminal.
        stderr=subprocess.DEVNULL,
    )


# ---------------------------------------------------------------------------
# Probe chain
# ---------------------------------------------------------------------------
#
# Probes live in a platform-ordered list. ``_probe`` walks the list and
# returns the first (backend, probe_log) pair that succeeds. The probe
# log is a list of (name, ok, reason) tuples useful for the ``status``
# subcommand's diagnostics — users want to see *why* wl-copy was skipped
# on a Wayland-ish system that doesn't have it installed.

_ProbeLog = List[Tuple[str, bool, str]]


def _probe() -> Tuple[Optional[_Backend], _ProbeLog]:
    """Return the first usable backend plus a log of every candidate.

    Callers should not use this directly — it is invoked once by
    :func:`detect_backend` and its result is cached.
    """
    log: _ProbeLog = []
    host = _platform.detect_os()

    for candidate in _ordered_candidates(host):
        backend, reason = candidate()
        if backend is not None:
            log.append((backend.name, True, "selected"))
            return backend, log
        log.append((candidate.__name__.removeprefix("_cand_"), False, reason))

    return None, log


def _ordered_candidates(host: _platform.OS) -> List[Callable[[], Tuple[Optional[_Backend], str]]]:
    """Return probe callables in the order we should try them for ``host``.

    Ordering rationale:
      * macOS: ``pbcopy`` is always present.
      * WSL: ``clip.exe`` first; Wayland/X11 tools almost never work
        inside WSL's default shell.
      * Linux: Wayland first (growing share, native), then X11 tools.
      * Windows: ``clip.exe`` is the Microsoft-supplied baseline.
      * Unknown: only the library fallback.

    We always end with the :mod:`pyperclip` fallback so users on obscure
    hosts still get *something*.
    """
    pyp = [_cand_pyperclip]

    if host is _platform.OS.MACOS:
        return [_cand_pbcopy] + pyp
    if host is _platform.OS.WSL:
        return [_cand_clip_exe, _cand_wl_copy, _cand_xclip, _cand_xsel] + pyp
    if host is _platform.OS.LINUX:
        return [_cand_wl_copy, _cand_xclip, _cand_xsel] + pyp
    if host is _platform.OS.WINDOWS:
        return [_cand_clip_exe] + pyp
    return pyp


# -- individual candidate probes --------------------------------------------
#
# Each probe returns (backend_or_None, reason). The name convention
# `_cand_<backend>` is used by `_probe` to derive a log label.

def _cand_pbcopy() -> Tuple[Optional[_Backend], str]:
    path = shutil.which("pbcopy")
    if not path:
        return None, "pbcopy not on PATH"
    return _Backend(name="pbcopy",
                    write=lambda t, p=path: _run_subprocess_copy([p], t)), ""


def _cand_wl_copy() -> Tuple[Optional[_Backend], str]:
    # wl-copy requires a running Wayland compositor. The $WAYLAND_DISPLAY
    # var is the canonical signal; its absence means wl-copy would hang
    # or fail with "Compositor doesn't support wl_data_device_manager".
    if not os.environ.get("WAYLAND_DISPLAY"):
        return None, "WAYLAND_DISPLAY unset"
    path = shutil.which("wl-copy")
    if not path:
        return None, "wl-copy not on PATH"
    return _Backend(name="wl-copy",
                    write=lambda t, p=path: _run_subprocess_copy([p], t)), ""


def _cand_xclip() -> Tuple[Optional[_Backend], str]:
    if not os.environ.get("DISPLAY"):
        return None, "DISPLAY unset"
    path = shutil.which("xclip")
    if not path:
        return None, "xclip not on PATH"
    return _Backend(
        name="xclip",
        write=lambda t, p=path: _run_subprocess_copy([p, "-selection", "clipboard"], t),
    ), ""


def _cand_xsel() -> Tuple[Optional[_Backend], str]:
    if not os.environ.get("DISPLAY"):
        return None, "DISPLAY unset"
    path = shutil.which("xsel")
    if not path:
        return None, "xsel not on PATH"
    return _Backend(
        name="xsel",
        write=lambda t, p=path: _run_subprocess_copy([p, "-b", "-i"], t),
    ), ""


def _cand_clip_exe() -> Tuple[Optional[_Backend], str]:
    # ``clip.exe`` lives under System32 on Windows and is surfaced on
    # WSL's PATH by default. We resolve it via `which` so the cached
    # path is absolute.
    path = shutil.which("clip.exe") or shutil.which("clip")
    if not path:
        return None, "clip.exe not on PATH"
    return _Backend(
        name="clip.exe",
        write=lambda t, p=path: _run_subprocess_copy(
            [p], t, strip_trailing_newline=True
        ),
    ), ""


def _cand_pyperclip() -> Tuple[Optional[_Backend], str]:
    try:
        import pyperclip  # type: ignore
    except ImportError:
        return None, "pyperclip not installed"

    def _write(t: str) -> None:
        pyperclip.copy(t)

    return _Backend(name="pyperclip", write=_write), ""


# ---------------------------------------------------------------------------
# Cached detection
# ---------------------------------------------------------------------------

# Cached on first call; reset only by explicit call to :func:`_reset_cache`
# from tests. The cache is a ``(backend, log)`` tuple, not just the
# backend, so ``describe()`` can report the probe log without re-probing.
_DETECTED: Optional[Tuple[Optional[_Backend], _ProbeLog]] = None


def detect_backend() -> Optional[_Backend]:
    """Return the currently selected backend, probing on first call.

    ``None`` means no usable backend was found; callers should surface
    that as a soft failure rather than an error.
    """
    global _DETECTED
    if _DETECTED is None:
        _DETECTED = _probe()
    return _DETECTED[0]


def _reset_cache() -> None:
    """Clear the probe cache — **for tests only**.

    The public API is process-life-cached on purpose; production code
    has no reason to call this.
    """
    global _DETECTED
    _DETECTED = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def copy(text: str, *, allow_secrets: bool = False) -> CopyResult:
    """Copy ``text`` to the system clipboard, after secret redaction.

    Parameters
    ----------
    text:
        The content to place on the clipboard. Empty strings are copied
        as empty strings — some users pipe empty output through ``--copy``
        and expect the previous clipboard to be overwritten rather than
        preserved.
    allow_secrets:
        When True, bypass the redaction-refusal gate. Matched secrets
        are still reported in ``redaction_findings`` for logging, but
        the raw text is copied verbatim.

    Returns
    -------
    :class:`CopyResult`
        Never raises for clipboard-layer errors; subprocess failures are
        translated into ``CopyResult(ok=False, reason=...)``. Callers
        therefore never need a try/except around :func:`copy`.
    """
    # --- redaction gate -------------------------------------------------
    redacted, findings = redact(text)
    if findings and not allow_secrets:
        return CopyResult(
            ok=False,
            backend=None,
            reason=(
                f"refused: {len(findings)} secret(s) detected "
                f"({', '.join(sorted({f.type for f in findings}))}); "
                "pass --allow-secrets to copy anyway"
            ),
            redaction_findings=findings,
        )

    payload = text if allow_secrets else redacted

    # --- backend selection ---------------------------------------------
    backend = detect_backend()
    if backend is None:
        return CopyResult(
            ok=False,
            backend=None,
            reason="no clipboard backend available (try `gemma clipboard status`)",
            redaction_findings=findings,
        )

    # --- actual write --------------------------------------------------
    try:
        backend.write(payload)
    except subprocess.TimeoutExpired:
        return CopyResult(
            ok=False,
            backend=backend.name,
            reason=f"{backend.name} timed out",
            redaction_findings=findings,
        )
    except subprocess.CalledProcessError as exc:
        return CopyResult(
            ok=False,
            backend=backend.name,
            reason=f"{backend.name} exited non-zero ({exc.returncode})",
            redaction_findings=findings,
        )
    except Exception as exc:  # noqa: BLE001 — we deliberately catch broadly
        # Any unexpected error (e.g. pyperclip raising because no display
        # server) becomes a soft failure so the CLI doesn't crash.
        return CopyResult(
            ok=False,
            backend=backend.name,
            reason=f"{backend.name} raised: {exc}",
            redaction_findings=findings,
        )

    return CopyResult(
        ok=True,
        backend=backend.name,
        reason=None,
        redaction_findings=findings,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def describe() -> dict:
    """Return a JSON-safe snapshot for ``gemma clipboard status``.

    Includes:
      * the selected backend (or ``None``),
      * the full probe log so users see *why* alternatives were skipped,
      * the host OS and SSH status (SSH clipboards are the #1 support
        question — if the user is on SSH and our backends failed, that
        is almost certainly the reason).
    """
    # Intentionally force a probe so users see current state even if no
    # copy has happened yet this run.
    backend = detect_backend()
    assert _DETECTED is not None  # set by detect_backend()
    _, log = _DETECTED

    return {
        "os": _platform.detect_os().value,
        "ssh": str(_platform.is_ssh()),
        "selected": backend.name if backend else None,
        "probe_log": [
            {"backend": name, "ok": ok, "reason": reason}
            for (name, ok, reason) in log
        ],
    }
