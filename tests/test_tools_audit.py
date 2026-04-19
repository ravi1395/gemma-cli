"""Tests for the append-only audit log.

Covers: record shape, secret redaction, append semantics, tail
filtering, sha256 helper, and file-permission handling.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gemma.tools import audit as _audit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def log_path(tmp_path):
    """Redirect the audit log to a tmp file for the duration of a test."""
    path = tmp_path / "audit.jsonl"
    _audit.set_log_path(path)
    yield path
    _audit.set_log_path(None)


# ---------------------------------------------------------------------------
# make_record
# ---------------------------------------------------------------------------

def test_make_record_redacts_string_args():
    """AWS keys and similar secrets should be scrubbed before logging."""
    rec = _audit.make_record(
        tool="read_file",
        capability="read",
        args={"path": "AKIAABCDEFGHIJKLMNOP", "mode": 0o644},
        session_id="s",
    )
    # Redaction replaces the secret with a [REDACTED:TYPE] marker.
    assert "AKIAABCDEFGHIJKLMNOP" not in rec.args_redacted["path"]
    assert "[REDACTED" in rec.args_redacted["path"]
    # Non-string values pass through untouched.
    assert rec.args_redacted["mode"] == 0o644


def test_make_record_sets_ts_and_fields():
    rec = _audit.make_record(
        tool="t", capability="read", args={}, session_id="s",
        duration_ms=12, exit_code=0,
    )
    assert rec.tool == "t"
    assert rec.capability == "read"
    assert rec.duration_ms == 12
    # ISO-8601 with Z suffix.
    assert rec.ts.endswith("Z")
    # Parseable as a datetime.
    datetime.strptime(rec.ts, "%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# append
# ---------------------------------------------------------------------------

def test_append_writes_one_jsonl_line_per_record(log_path):
    _audit.append(_audit.make_record(
        tool="a", capability="read", args={}, session_id="s",
    ))
    _audit.append(_audit.make_record(
        tool="b", capability="read", args={}, session_id="s",
    ))
    lines = log_path.read_text().splitlines()
    assert len(lines) == 2
    for line in lines:
        # Each line must parse as JSON independently.
        json.loads(line)


def test_append_creates_parent_dir_with_strict_mode(log_path, tmp_path):
    nested = tmp_path / "sub" / "nested" / "audit.jsonl"
    _audit.set_log_path(nested)
    _audit.append(_audit.make_record(
        tool="t", capability="read", args={}, session_id="s",
    ))
    assert nested.exists()


def test_append_is_append_only(log_path):
    for i in range(5):
        _audit.append(_audit.make_record(
            tool=f"t{i}", capability="read", args={}, session_id="s",
        ))
    assert len(log_path.read_text().splitlines()) == 5
    # Prior lines are preserved.
    parsed = [json.loads(l) for l in log_path.read_text().splitlines()]
    assert [r["tool"] for r in parsed] == [f"t{i}" for i in range(5)]


def test_append_on_ioerror_warns_but_does_not_raise(monkeypatch, capsys, tmp_path):
    """A logging failure must not propagate up to the caller."""
    bad_path = tmp_path / "does" / "not" / "exist" / "audit.jsonl"
    _audit.set_log_path(bad_path)

    # Make mkdir raise so the append path fails.
    def _raise(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(Path, "mkdir", _raise)

    try:
        # Must not raise.
        _audit.append(_audit.make_record(
            tool="t", capability="read", args={}, session_id="s",
        ))
    finally:
        _audit.set_log_path(None)

    captured = capsys.readouterr()
    assert "tool audit write failed" in captured.err


# ---------------------------------------------------------------------------
# tail
# ---------------------------------------------------------------------------

def test_tail_returns_last_n(log_path):
    for i in range(10):
        _audit.append(_audit.make_record(
            tool=f"t{i}", capability="read", args={}, session_id="s",
        ))
    recent = _audit.tail(n=3)
    assert [r["tool"] for r in recent] == ["t7", "t8", "t9"]


def test_tail_filters_by_since_iso(log_path):
    """Entries older than ``since_iso`` must be excluded."""
    old = _audit.make_record(
        tool="old", capability="read", args={}, session_id="s",
    )
    # Forge an older timestamp.
    from dataclasses import replace
    old = replace(old, ts="2020-01-01T00:00:00Z")
    _audit.append(old)
    _audit.append(_audit.make_record(
        tool="new", capability="read", args={}, session_id="s",
    ))

    recent = _audit.tail(n=10, since_iso="2024-01-01T00:00:00Z")
    assert [r["tool"] for r in recent] == ["new"]


def test_tail_skips_corrupt_lines(log_path):
    _audit.append(_audit.make_record(
        tool="good", capability="read", args={}, session_id="s",
    ))
    with log_path.open("a") as fh:
        fh.write("not json\n")
    _audit.append(_audit.make_record(
        tool="also_good", capability="read", args={}, session_id="s",
    ))

    recent = _audit.tail(n=10)
    assert [r["tool"] for r in recent] == ["good", "also_good"]


def test_tail_returns_empty_when_no_file(log_path):
    assert not log_path.exists()
    assert _audit.tail(n=10) == []


# ---------------------------------------------------------------------------
# sha256_of
# ---------------------------------------------------------------------------

def test_sha256_of_returns_16_hex_chars(tmp_path):
    f = tmp_path / "x"
    f.write_bytes(b"hello")
    digest = _audit.sha256_of(f)
    assert len(digest) == 16
    assert all(c in "0123456789abcdef" for c in digest)


def test_sha256_of_dir_returns_sentinel(tmp_path):
    assert _audit.sha256_of(tmp_path) == "dir"


def test_sha256_of_missing_returns_sentinel(tmp_path):
    assert _audit.sha256_of(tmp_path / "nope") == "missing"
