"""Tests for the RAG manifest — FileEntry, Manifest, and diff semantics.

We deliberately avoid Redis in this file; the manifest is a pure in-
memory data structure and its tests should run in microseconds.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from gemma.rag.manifest import FileEntry, Manifest, ManifestDiff, _sha1_of_file


# ---------------------------------------------------------------------------
# FileEntry
# ---------------------------------------------------------------------------

def test_file_entry_from_disk_relative_and_hashed(tmp_path):
    (tmp_path / "a.py").write_text("print('hi')\n")
    entry = FileEntry.from_disk(tmp_path, tmp_path / "a.py")
    assert entry.path == "a.py"
    assert entry.size == len("print('hi')\n")
    # sha1 hex is 40 chars.
    assert len(entry.sha1) == 40


def test_file_entry_from_disk_nested(tmp_path):
    sub = tmp_path / "src" / "pkg"
    sub.mkdir(parents=True)
    (sub / "mod.py").write_text("x = 1")
    entry = FileEntry.from_disk(tmp_path, sub / "mod.py")
    # POSIX form regardless of OS.
    assert entry.path == "src/pkg/mod.py"


def test_file_entry_roundtrip_json(tmp_path):
    e = FileEntry(
        path="a.py", mtime_ns=123, size=10,
        sha1="deadbeef" * 5, chunk_ids=["c1", "c2"],
    )
    blob = e.to_json()
    back = FileEntry.from_json(blob)
    assert back == e


def test_file_entry_from_json_tolerates_missing_chunk_ids():
    # A manifest written by an older version may lack chunk_ids.
    blob = json.dumps({
        "path": "a.py", "mtime_ns": 1, "size": 2, "sha1": "ab" * 20,
    })
    e = FileEntry.from_json(blob)
    assert e.chunk_ids == []


def test_file_entry_same_content_ignores_chunk_ids():
    e1 = FileEntry(path="a", mtime_ns=1, size=2, sha1="x", chunk_ids=["c1"])
    e2 = FileEntry(path="a", mtime_ns=1, size=2, sha1="x", chunk_ids=["c9"])
    assert e1.same_content_as(e2)


def test_file_entry_same_content_detects_content_drift():
    e1 = FileEntry(path="a", mtime_ns=1, size=2, sha1="x")
    e2 = FileEntry(path="a", mtime_ns=1, size=2, sha1="y")
    assert not e1.same_content_as(e2)


# ---------------------------------------------------------------------------
# Manifest (de)serialisation
# ---------------------------------------------------------------------------

def test_manifest_from_redis_hash_skips_corrupt_blobs():
    good = FileEntry(path="a.py", mtime_ns=1, size=2, sha1="x").to_json()
    bad = "{ not json"
    m = Manifest.from_redis_hash({"a.py": good, "broken.py": bad})
    assert "a.py" in m
    assert "broken.py" not in m  # corrupt row is dropped


def test_manifest_to_redis_hash_round_trip():
    m1 = Manifest({"a.py": FileEntry(path="a.py", mtime_ns=1, size=2, sha1="x")})
    m2 = Manifest.from_redis_hash(m1.to_redis_hash())
    assert m1.paths() == m2.paths()


def test_manifest_all_chunk_ids_flattens():
    m = Manifest({
        "a.py": FileEntry(path="a.py", mtime_ns=0, size=0, sha1="", chunk_ids=["c1", "c2"]),
        "b.py": FileEntry(path="b.py", mtime_ns=0, size=0, sha1="", chunk_ids=["c3"]),
    })
    assert sorted(m.all_chunk_ids()) == ["c1", "c2", "c3"]


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------

def _entry(path, sha1="x", mtime=1, size=10, cids=()):
    return FileEntry(path=path, mtime_ns=mtime, size=size, sha1=sha1, chunk_ids=list(cids))


def test_diff_empty_prior_everything_added():
    prior = Manifest()
    new = Manifest({"a.py": _entry("a.py"), "b.py": _entry("b.py")})
    d = prior.diff(new)
    assert len(d.added) == 2
    assert d.changed == []
    assert d.removed == []


def test_diff_unchanged_entries_preserve_chunk_ids():
    """An unchanged file must carry its *old* chunk_ids through the diff."""
    prior = Manifest({"a.py": _entry("a.py", cids=["c1", "c2"])})
    new = Manifest({"a.py": _entry("a.py", cids=[])})  # fresh walk hasn't chunked yet
    d = prior.diff(new)
    assert len(d.unchanged) == 1
    assert d.unchanged[0].chunk_ids == ["c1", "c2"]
    assert d.added == [] and d.changed == [] and d.removed == []


def test_diff_changed_emits_old_and_new_together():
    prior = Manifest({"a.py": _entry("a.py", sha1="v1", cids=["old"])})
    new = Manifest({"a.py": _entry("a.py", sha1="v2")})
    d = prior.diff(new)
    assert len(d.changed) == 1
    old, new_e = d.changed[0]
    assert old.sha1 == "v1"
    assert new_e.sha1 == "v2"
    # changed is not double-counted as added.
    assert d.added == []


def test_diff_removed_carries_full_old_entry():
    """Removed files must carry chunk_ids through so the indexer can GC them."""
    prior = Manifest({"a.py": _entry("a.py", cids=["c1", "c2"])})
    new = Manifest()  # file was deleted from disk
    d = prior.diff(new)
    assert len(d.removed) == 1
    assert d.removed[0].chunk_ids == ["c1", "c2"]


def test_diff_has_work_flag():
    empty_diff = ManifestDiff(added=[], changed=[], removed=[], unchanged=[])
    assert empty_diff.has_work is False
    assert ManifestDiff(added=[_entry("a")], changed=[], removed=[], unchanged=[]).has_work


# ---------------------------------------------------------------------------
# sha1 helper
# ---------------------------------------------------------------------------

def test_sha1_of_file_matches_known_value(tmp_path):
    # sha1("hello world") = 2aae6c35c94fcfb415dbe95f408b9ce91ee846ed
    p = tmp_path / "x"
    p.write_bytes(b"hello world")
    assert _sha1_of_file(p) == "2aae6c35c94fcfb415dbe95f408b9ce91ee846ed"


def test_sha1_of_file_streams_large_files(tmp_path):
    """Hashing works for files larger than the 64 KiB stream buffer."""
    p = tmp_path / "big"
    p.write_bytes(b"A" * (200 * 1024))
    digest = _sha1_of_file(p, chunk_size=16 * 1024)
    # Sanity: deterministic and 40 hex chars.
    assert len(digest) == 40
    assert digest == _sha1_of_file(p, chunk_size=64 * 1024)
