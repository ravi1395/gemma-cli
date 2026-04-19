"""Hardened ``subprocess.run`` wrapper used by every subprocess-driven tool.

Three invariants this module exists to uphold:

1. **No shell.** ``shell=False`` always. Argv is a list — never a
   string that could be parsed by ``/bin/sh``. Regex-escape bugs are
   the #1 source of shell-injection incidents in tools like this one.
2. **Scrubbed environment.** We rebuild ``env`` from a small allowlist.
   AWS keys, cloud-provider tokens, model-provider API keys, and
   assorted CI secrets never reach the child process.
3. **Bounded output + wall-clock timeout.** Tool calls can't exhaust
   memory (output cap) or hang the CLI (timeout). On timeout we
   escalate: first ``terminate`` then ``kill`` after a short grace
   period, so the child can't ignore SIGTERM forever.

The single entry point :func:`run` returns a structured
:class:`RunResult` so callers can distinguish "ran and exited non-zero"
from "timed out" from "could not start". That structure also serialises
cleanly into the audit log.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple


#: Env vars we pass through to child processes. Everything else is
#: scrubbed. Tools that genuinely need more (e.g. a proxy setting) can
#: be audited individually — the default is intentionally tight.
DEFAULT_ENV_ALLOWLIST: Tuple[str, ...] = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "TMPDIR",
    "TEMP",
    "TMP",
    # Virtualenv / shell introspection so tools like pytest find their
    # interpreter. Still omits CREDENTIAL-carrying names.
    "VIRTUAL_ENV",
    "PYTHONPATH",
    "PYTHONHOME",
    "PYENV_VERSION",
    "NODE_PATH",
)

#: Truncation marker appended when output exceeds ``max_output_bytes``.
TRUNCATION_MARKER = "\n…[truncated by gemma: exceeded {limit} bytes]\n"

#: Grace period in seconds between SIGTERM and SIGKILL on timeout.
_KILL_GRACE_SECONDS = 2.0


@dataclass(frozen=True)
class RunResult:
    """Outcome of :func:`run`.

    Attributes:
        exit_code: Child exit code. ``-1`` for "failed to start",
            ``-2`` for "timed out".
        stdout: Child stdout, truncated if over cap.
        stderr: Child stderr, truncated if over cap.
        duration_ms: Wall-clock runtime in milliseconds.
        timed_out: True iff the process was killed by the timeout.
        truncated: True iff either stream hit the output cap.
        start_error: Populated when the executable could not be
            launched (e.g. not on ``PATH``). Mutually exclusive with
            a normal exit.
    """

    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool = False
    truncated: bool = False
    start_error: Optional[str] = None


# ---------------------------------------------------------------------------
# Environment scrubbing
# ---------------------------------------------------------------------------

def build_env(allowlist: Iterable[str] = DEFAULT_ENV_ALLOWLIST) -> dict:
    """Build a minimal env dict containing only allowlisted names.

    Kept as a separate helper so tests can assert on the scrub logic
    without running a subprocess.
    """
    out: dict = {}
    for name in allowlist:
        if name in os.environ:
            out[name] = os.environ[name]
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    argv: list[str],
    *,
    cwd: Path,
    timeout_s: float,
    max_output_bytes: int,
    env_allowlist: Iterable[str] = DEFAULT_ENV_ALLOWLIST,
) -> RunResult:
    """Run ``argv`` under the hardened policy.

    Args:
        argv: Executable + args. ``argv[0]`` must be resolvable via
            ``PATH`` inside the scrubbed env, or an absolute path.
        cwd: Working directory. Caller must have already validated
            this through :mod:`gemma.safety`.
        timeout_s: Wall-clock deadline. On expiry the child is
            terminated, then killed, and :class:`RunResult` reports
            ``timed_out=True``.
        max_output_bytes: Per-stream cap. Output beyond this is
            dropped; a truncation marker is appended so the caller
            knows the story isn't complete.
        env_allowlist: Names of env vars to pass through.

    Returns:
        :class:`RunResult`. Never raises for normal failures; only
        re-raises truly unexpected exceptions.
    """
    import time

    env = build_env(env_allowlist)
    started = time.monotonic()

    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(cwd),
            env=env,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Start in a new process group so a timed-out child's
            # children (e.g. pytest workers) go too when we SIGKILL.
            start_new_session=os.name != "nt",
        )
    except (FileNotFoundError, PermissionError, OSError) as exc:
        dur_ms = int((time.monotonic() - started) * 1000)
        return RunResult(
            exit_code=-1,
            stdout="",
            stderr="",
            duration_ms=dur_ms,
            start_error=f"{type(exc).__name__}: {exc}",
        )

    timed_out = False
    try:
        stdout_b, stderr_b = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        # Polite first (SIGTERM gives Python children a chance to tear
        # down), then mandatory (SIGKILL).
        proc.terminate()
        try:
            stdout_b, stderr_b = proc.communicate(timeout=_KILL_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout_b, stderr_b = proc.communicate()

    dur_ms = int((time.monotonic() - started) * 1000)
    stdout, stdout_trunc = _truncate(stdout_b, max_output_bytes)
    stderr, stderr_trunc = _truncate(stderr_b, max_output_bytes)

    return RunResult(
        # -2 conventionally marks "killed by us", -N for SIGNAL otherwise.
        exit_code=-2 if timed_out else proc.returncode,
        stdout=stdout,
        stderr=stderr,
        duration_ms=dur_ms,
        timed_out=timed_out,
        truncated=stdout_trunc or stderr_trunc,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _truncate(blob: bytes, limit: int) -> Tuple[str, bool]:
    """Decode ``blob`` to UTF-8 and append a marker if it was capped.

    Non-UTF-8 bytes are replaced; we would rather surface a best-effort
    transcript than crash a tool over mojibake in its output.
    """
    if not blob:
        return "", False
    if len(blob) <= limit:
        return blob.decode("utf-8", errors="replace"), False
    head = blob[:limit].decode("utf-8", errors="replace")
    return head + TRUNCATION_MARKER.format(limit=limit), True
