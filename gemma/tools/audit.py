"""Append-only JSONL audit log for tool invocations.

Every tool call — successful, refused, or failed — produces exactly one
JSON record written to ``~/.gemma/tool-audit.jsonl``. The log is
append-only: existing lines are never rewritten or truncated, which
makes tampering detection straightforward (a shortened file is visible
evidence).

Threat model
------------
Two classes of failure to guard against:

1. **Data leak.** The raw user prompt can contain secrets. We redact
   every argument value through :func:`gemma.redaction.redact` before
   writing. The post-redaction form is what gets persisted.
2. **Silent no-op.** If the log file is missing, unwritable, or the
   disk is full, the audit record is dropped — but we *never* let a
   logging failure mask a real tool call. The caller sees a warning
   on stderr; the tool itself still ran.

File format
-----------
One JSON object per line. Fields defined in :func:`make_record`.
Permissions are set to 0o600 on creation; the containing directory
to 0o700.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from gemma import redaction as _redaction


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Default location. Overridable at runtime via ``set_log_path``.
DEFAULT_LOG_PATH = Path.home() / ".gemma" / "tool-audit.jsonl"

# Guards concurrent writes from the same process. Cross-process safety
# is provided by the append mode (O_APPEND) on POSIX; on Windows we
# accept best-effort semantics — tool-audit is not a transaction log.
_WRITE_LOCK = threading.Lock()

# Injectable log path (tests point this at tmp_path).
_current_log_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PathDigest:
    """A file that was read, written, or archived, captured by sha256.

    The hash lets an auditor verify *what content* was touched even
    after the file is later modified. We keep it to 16 hex chars in
    the log to stay compact — collision risk is negligible for a
    per-session audit.
    """

    path: str
    sha256: str


@dataclass(frozen=True)
class AuditRecord:
    """One line in the audit log.

    All fields are JSON-serialisable primitives or lists/dicts
    thereof. The record's shape must remain backwards-compatible:
    new fields may be added, existing fields must not be renamed or
    removed (auditors may parse log files older than the current
    release).
    """

    ts: str                   # ISO-8601 UTC
    session_id: str
    tool: str
    capability: str           # Capability value; string for forward-compat.
    args_redacted: Dict[str, Any]
    paths_touched: List[PathDigest] = field(default_factory=list)
    exit_code: int = 0
    duration_ms: int = 0
    approved_by: str = "auto"  # "auto" | "user" | "flag" | "refused"
    refusal_reason: Optional[str] = None
    cached: bool = False      # True when the result was served from AgentSessionCache


# ---------------------------------------------------------------------------
# Path management
# ---------------------------------------------------------------------------

def set_log_path(path: Optional[Path]) -> None:
    """Redirect the audit log.

    Passing ``None`` resets to :data:`DEFAULT_LOG_PATH`. Tests use this
    to write into a ``tmp_path``; runtime callers should almost always
    leave the default alone.
    """
    global _current_log_path
    _current_log_path = Path(path) if path is not None else None


def current_log_path() -> Path:
    """Return the active log path (default or override)."""
    return _current_log_path if _current_log_path is not None else DEFAULT_LOG_PATH


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def append(record: AuditRecord) -> None:
    """Write ``record`` as one JSONL line, creating the file if needed.

    Never raises on I/O failure — the caller must not be blocked by a
    logging hiccup. A warning is emitted to stderr instead so the user
    sees *something* go wrong.
    """
    path = current_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        line = json.dumps(asdict(record), separators=(",", ":"), sort_keys=True)

        # O_APPEND makes writes atomic within POSIX's documented guarantees
        # for writes smaller than PIPE_BUF (typically 4 KB). Our lines are
        # well under that.
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        fd = os.open(path, flags, 0o600)
        try:
            with _WRITE_LOCK:
                os.write(fd, (line + "\n").encode("utf-8"))
        finally:
            os.close(fd)

    except OSError as exc:
        # Do not propagate — audit failures must not prevent work.
        print(f"gemma: warning: tool audit write failed: {exc}", file=sys.stderr)


def make_record(
    *,
    tool: str,
    capability: str,
    args: Dict[str, Any],
    session_id: str,
    paths_touched: Optional[List[PathDigest]] = None,
    exit_code: int = 0,
    duration_ms: int = 0,
    approved_by: str = "auto",
    refusal_reason: Optional[str] = None,
    cached: bool = False,
) -> AuditRecord:
    """Build an :class:`AuditRecord`, redacting string-valued args.

    Values that are not plain strings are passed through untouched.
    That matches the threat model: secrets travel as strings from user
    input or file content, not as ints or bools.
    """
    redacted: Dict[str, Any] = {}
    for k, v in args.items():
        if isinstance(v, str):
            clean, _findings = _redaction.redact(v)
            redacted[k] = clean
        else:
            redacted[k] = v

    return AuditRecord(
        ts=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        session_id=session_id,
        tool=tool,
        capability=capability,
        args_redacted=redacted,
        paths_touched=list(paths_touched or []),
        exit_code=exit_code,
        duration_ms=duration_ms,
        approved_by=approved_by,
        refusal_reason=refusal_reason,
        cached=cached,
    )


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def tail(n: int = 50, *, since_iso: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return the last ``n`` records, optionally filtered by timestamp.

    Reads the whole file. For today's expected log sizes (a few MB)
    that is fine; if logs grow, switch to a reverse-seek reader.

    Args:
        n: Maximum number of records to return.
        since_iso: If set, keep only records with ``ts >= since_iso``.
    """
    path = current_log_path()
    if not path.exists():
        return []

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # Corrupt line — skip, don't crash a user running
                # `gemma tools audit`.
                continue
            if since_iso and rec.get("ts", "") < since_iso:
                continue
            records.append(rec)

    return records[-n:]


# ---------------------------------------------------------------------------
# Hashing helper
# ---------------------------------------------------------------------------

def sha256_of(path: Path, *, max_bytes: int = 16 * 1024 * 1024) -> str:
    """Return a sha256 digest (16 hex chars) of ``path``'s first bytes.

    Capped at ``max_bytes`` — the audit trail doesn't need to hash a
    10-GB file to be useful. Missing files and dirs return a sentinel
    so callers don't have to special-case them.
    """
    try:
        if path.is_dir():
            return "dir"
        h = hashlib.sha256()
        with path.open("rb") as fh:
            h.update(fh.read(max_bytes))
        return h.hexdigest()[:16]
    except OSError:
        return "missing"
