"""Manifest — the record of what has already been indexed.

The manifest is a Redis hash mapping **relative path → FileEntry**. On
every ``gemma rag index`` invocation we:

1. Load the previous manifest from Redis.
2. Walk the workspace, build a fresh ``FileEntry`` per file.
3. Diff the two to decide what to re-embed.

A file is considered **unchanged** when ``(mtime_ns, size, sha1)``
matches. ``(mtime_ns, size)`` alone would be fast but risks false
negatives when tools rewrite bytes without changing mtime; ``sha1``
alone would be correct but forces us to read every byte of every file.
Using all three means we only pay the sha1 cost when mtime/size have
actually shifted — i.e. only for files we *suspect* have changed.

Serialisation format
--------------------
Each manifest entry is stored as a single JSON blob per path so we can
roundtrip through ``hgetall`` without a second round-trip to fetch
chunk lists. The blob is small — a few hundred bytes — so total memory
use scales linearly with the number of files, not chunks.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class FileEntry:
    """One row of the manifest — metadata for a single indexed file.

    Attributes:
        path:       Path relative to the workspace root, POSIX form.
                    Stored verbatim in Redis; used as the manifest-hash
                    field name.
        mtime_ns:   Modification time in nanoseconds since epoch. Cheap
                    to compare and usually sufficient.
        size:       File size in bytes. A backup signal alongside mtime
                    so an ``mtime`` reset doesn't hide a real edit.
        sha1:       40-char hex digest of the file contents. The
                    authoritative change-detector.
        chunk_ids:  Chunk IDs emitted by the chunker for this file.
                    Kept on the entry so we can delete exactly this
                    file's chunks when it's removed or replaced.
    """

    path: str
    mtime_ns: int
    size: int
    sha1: str
    chunk_ids: List[str] = field(default_factory=list)

    @classmethod
    def from_disk(
        cls,
        root: Path,
        path: Path,
        *,
        chunk_ids: Optional[List[str]] = None,
    ) -> "FileEntry":
        """Build a FileEntry by stat'ing and hashing a file.

        ``path`` is resolved relative to ``root`` so the stored ``path``
        is stable across machines (``gemma/rag/store.py`` vs.
        ``/home/alice/proj/gemma/rag/store.py``).
        """
        abs_path = (root / path).resolve() if not path.is_absolute() else path
        rel = abs_path.relative_to(root.resolve()).as_posix()
        stat = abs_path.stat()
        return cls(
            path=rel,
            mtime_ns=stat.st_mtime_ns,
            size=stat.st_size,
            sha1=_sha1_of_file(abs_path),
            chunk_ids=list(chunk_ids or []),
        )

    def to_json(self) -> str:
        """Serialise the entry as a compact JSON string for Redis storage."""
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, blob: str) -> "FileEntry":
        """Inverse of :meth:`to_json`; tolerates unknown future keys."""
        data = json.loads(blob)
        return cls(
            path=data["path"],
            mtime_ns=int(data["mtime_ns"]),
            size=int(data["size"]),
            sha1=data["sha1"],
            chunk_ids=list(data.get("chunk_ids", [])),
        )

    def same_content_as(self, other: "FileEntry") -> bool:
        """Return True iff (mtime_ns, size, sha1) all match ``other``.

        Content-equality ignores ``chunk_ids`` because those are derived
        state — the indexer might re-chunk the same bytes differently
        if the chunker changes, but the manifest shouldn't mark it
        changed for that reason alone.
        """
        return (
            self.mtime_ns == other.mtime_ns
            and self.size == other.size
            and self.sha1 == other.sha1
        )


@dataclass(frozen=True)
class ManifestDiff:
    """Result of diffing two manifests.

    Attributes:
        added:     Paths present in the new snapshot but not the old.
        changed:   Paths whose content differs. Carries both the old
                   entry (so we know which chunk_ids to delete) and the
                   new one (so we can replace it).
        removed:   Paths present in the old manifest but not the new —
                   their chunks must be deleted.
        unchanged: Paths that passed the content-equality check. Used
                   only for telemetry; no store writes happen for them.
    """

    added: List[FileEntry]
    changed: List[Tuple[FileEntry, FileEntry]]  # (old, new)
    removed: List[FileEntry]
    unchanged: List[FileEntry]

    @property
    def has_work(self) -> bool:
        """True if the indexer has any writes to perform."""
        return bool(self.added or self.changed or self.removed)


class Manifest:
    """In-memory manifest with (de)serialisation helpers.

    Keep this class dumb: no Redis awareness. The indexer owns the
    Redis roundtrip and passes a ``Dict[path, blob]`` in, or asks us
    to produce one out. That makes it easy to unit-test the diff
    without spinning up a Redis (even a fake one).
    """

    def __init__(self, entries: Optional[Dict[str, FileEntry]] = None):
        self._entries: Dict[str, FileEntry] = dict(entries or {})

    # ------------------------------------------------------------------
    # Factory / serialisation
    # ------------------------------------------------------------------

    @classmethod
    def from_redis_hash(cls, blobs: Dict[str, str]) -> "Manifest":
        """Build a manifest from the ``hgetall`` output of a Redis hash.

        Corrupt rows (JSON decode errors) are dropped rather than
        aborting the whole index — a single bad blob shouldn't force a
        full rebuild.
        """
        entries: Dict[str, FileEntry] = {}
        for path, blob in blobs.items():
            try:
                entries[path] = FileEntry.from_json(blob)
            except (ValueError, KeyError):
                # Skip — the indexer will treat this path as "added"
                # and rewrite a valid entry for it.
                continue
        return cls(entries)

    def to_redis_hash(self) -> Dict[str, str]:
        """Inverse of :meth:`from_redis_hash` — produces an ``hmset`` payload."""
        return {p: e.to_json() for p, e in self._entries.items()}

    # ------------------------------------------------------------------
    # Entry access
    # ------------------------------------------------------------------

    def __contains__(self, path: str) -> bool:
        return path in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, path: str) -> Optional[FileEntry]:
        return self._entries.get(path)

    def all_chunk_ids(self) -> List[str]:
        """Flatten every entry's chunk_ids into one list (useful for GC)."""
        return [cid for e in self._entries.values() for cid in e.chunk_ids]

    def paths(self) -> List[str]:
        return list(self._entries.keys())

    def entries(self) -> List[FileEntry]:
        return list(self._entries.values())

    def set(self, entry: FileEntry) -> None:
        self._entries[entry.path] = entry

    def remove(self, path: str) -> None:
        self._entries.pop(path, None)

    # ------------------------------------------------------------------
    # Diff
    # ------------------------------------------------------------------

    def diff(self, new: "Manifest") -> ManifestDiff:
        """Compare this (prior) manifest against ``new`` (current-tree).

        ``self`` is treated as the "before" and ``new`` as the "after".
        """
        added: List[FileEntry] = []
        changed: List[Tuple[FileEntry, FileEntry]] = []
        unchanged: List[FileEntry] = []

        old_paths = set(self._entries.keys())
        new_paths = set(new._entries.keys())

        for path in new_paths:
            new_entry = new._entries[path]
            old_entry = self._entries.get(path)
            if old_entry is None:
                added.append(new_entry)
            elif old_entry.same_content_as(new_entry):
                # Preserve the *old* chunk_ids on the unchanged entry
                # so the indexer doesn't need to re-chunk to know them.
                unchanged.append(old_entry)
            else:
                changed.append((old_entry, new_entry))

        removed = [self._entries[p] for p in old_paths - new_paths]

        return ManifestDiff(
            added=added, changed=changed, removed=removed, unchanged=unchanged,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha1_of_file(path: Path, chunk_size: int = 64 * 1024) -> str:
    """Stream the file through SHA-1 in ``chunk_size`` blocks.

    Streaming is important for multi-MB source files — loading the
    whole file into memory just to hash it would be wasteful when the
    indexer will discard everything except the digest.
    """
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()
