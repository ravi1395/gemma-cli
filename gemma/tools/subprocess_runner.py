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
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Dict, Iterable, Optional, Tuple


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

#: Bound on how long we wait for a reader thread to finish draining a
#: pipe after the child has exited. The pipes hit EOF promptly under
#: normal conditions; this just protects against a wedged reader.
_READER_JOIN_TIMEOUT = 5.0

#: Chunk size used by the capped stream readers. 8 KB is a sweet spot:
#: large enough to amortise the read syscall, small enough that the
#: in-thread truncation check fires promptly past ``max_output_bytes``.
_READ_CHUNK_BYTES = 8192


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

    # Stream stdout/stderr through capped reader threads instead of
    # ``proc.communicate`` so a runaway tool can't pull megabytes of
    # output into RAM only for us to truncate it after the fact. Each
    # reader holds at most ``max_output_bytes`` of payload; anything
    # past the cap is read and immediately discarded so the OS pipe
    # buffer doesn't fill and stall the child.
    out_buf = bytearray()
    err_buf = bytearray()
    out_flags: Dict[str, bool] = {"truncated": False}
    err_flags: Dict[str, bool] = {"truncated": False}

    t_out = threading.Thread(
        target=_drain_capped,
        args=(proc.stdout, max_output_bytes, out_buf, out_flags),
        daemon=True,
        name="gemma-tool-stdout",
    )
    t_err = threading.Thread(
        target=_drain_capped,
        args=(proc.stderr, max_output_bytes, err_buf, err_flags),
        daemon=True,
        name="gemma-tool-stderr",
    )
    t_out.start()
    t_err.start()

    timed_out = False
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        # Polite first (SIGTERM gives Python children a chance to tear
        # down), then mandatory (SIGKILL).
        proc.terminate()
        try:
            proc.wait(timeout=_KILL_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    # Once the child is gone the pipes hit EOF and the reader threads
    # exit on their own. Join with a generous bound so a stuck reader
    # never wedges the CLI; if join times out we drop the partial
    # buffers and move on.
    t_out.join(timeout=_READER_JOIN_TIMEOUT)
    t_err.join(timeout=_READER_JOIN_TIMEOUT)
    if proc.stdout is not None:
        proc.stdout.close()
    if proc.stderr is not None:
        proc.stderr.close()

    dur_ms = int((time.monotonic() - started) * 1000)
    stdout, stdout_trunc = _format_capped(
        bytes(out_buf), out_flags["truncated"], max_output_bytes,
    )
    stderr, stderr_trunc = _format_capped(
        bytes(err_buf), err_flags["truncated"], max_output_bytes,
    )

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

def _drain_capped(
    stream: Optional[IO[bytes]],
    limit: int,
    out_buf: bytearray,
    flags: Dict[str, bool],
) -> None:
    """Reader-thread target: pull ``stream`` into ``out_buf`` up to ``limit``.

    The crucial property: once we've collected ``limit`` bytes we keep
    reading and discarding so the OS pipe buffer never fills and the
    child never blocks on a write — but we don't grow ``out_buf`` past
    ``limit``. This is what gives us a hard memory ceiling regardless
    of how chatty the tool is.

    ``flags["truncated"]`` is set when at least one byte was discarded.
    Errors during read (broken pipe after kill, etc.) are swallowed —
    the caller already tracks ``timed_out`` separately.
    """
    if stream is None:
        return
    try:
        while True:
            chunk = stream.read(_READ_CHUNK_BYTES)
            if not chunk:
                return
            free = limit - len(out_buf)
            if free <= 0:
                # Past the cap — keep draining so the child isn't
                # blocked on a full pipe, but don't grow the buffer.
                flags["truncated"] = True
                continue
            if len(chunk) <= free:
                out_buf.extend(chunk)
            else:
                out_buf.extend(chunk[:free])
                flags["truncated"] = True
    except (OSError, ValueError):
        # ``ValueError`` fires when ``read`` is called on a closed
        # pipe (race with a SIGKILL); ``OSError`` covers EPIPE etc.
        # In both cases the partial buffer we already gathered is the
        # correct payload to surface.
        return


def _format_capped(blob: bytes, truncated: bool, limit: int) -> Tuple[str, bool]:
    """Decode ``blob`` to UTF-8 and append the truncation marker if needed.

    Non-UTF-8 bytes are replaced — we would rather surface a best-effort
    transcript than crash a tool over mojibake in its output.
    """
    if not blob:
        return ("" if not truncated else TRUNCATION_MARKER.format(limit=limit), truncated)
    text = blob.decode("utf-8", errors="replace")
    if truncated:
        return text + TRUNCATION_MARKER.format(limit=limit), True
    return text, False
