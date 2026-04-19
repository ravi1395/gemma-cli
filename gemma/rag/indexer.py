"""Incremental RAG indexer.

Pipeline
--------
::

    walk -> build fresh manifest -> diff vs. stored manifest
         -> delete (removed + changed-old) chunks
         -> chunk + embed (added + changed-new) files
         -> upsert new chunks
         -> save refreshed manifest + namespace meta

Every indexer run is idempotent: running it twice in a row does all
the work the first time and no embedding calls the second.

Safety
------
The indexer only *reads* source files and *writes* to Redis. It never
writes back into the workspace. We still route each candidate file
through :func:`gemma.safety.is_denylisted` so a RAG pass over a repo
root transparently skips ``.git``, ``.env``, ``~/.ssh``, etc.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence

import numpy as np

from gemma import safety as _safety
from gemma.chunking import Chunk, chunk_for_path
from gemma.rag._math import normalise
from gemma.rag.manifest import FileEntry, Manifest, ManifestDiff
from gemma.rag.store import RedisVectorStore


#: File extensions the indexer will consider by default.
#:
#: Deliberately short — we want to index *source code and docs*, not
#: binary blobs, images, or node_modules. The extension whitelist is
#: the simplest way to enforce that; users who want more granularity
#: can pass their own list.
_DEFAULT_EXTENSIONS: tuple[str, ...] = (
    ".py", ".pyi",
    ".md", ".markdown", ".rst",
    ".ts", ".tsx", ".js", ".jsx",
    ".java", ".kt",
    ".go", ".rs",
    ".yaml", ".yml", ".toml", ".json",
    ".txt",
    ".sh", ".bash",
)

#: Directory names to skip outright. Matched against the directory's
#: own name, not the full path, so ``src/node_modules`` and
#: ``vendor/node_modules`` are both pruned.
_SKIP_DIRS: frozenset[str] = frozenset({
    ".git", ".hg", ".svn",
    "node_modules", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "dist", "build", ".venv", "venv",
    ".gradle", "target",
    "archive",  # gemma's own archive folder — never re-index archived content.
})

#: Largest file (in bytes) we'll attempt to index. Source files above
#: this are almost always generated (bundled JS, huge test fixtures)
#: and would dominate the embedding budget for little retrieval value.
_MAX_FILE_SIZE = 512 * 1024

#: Ollama's batched embed endpoint handles this many inputs per call
#: without truncation. Higher batches speed up the initial full-repo
#: index but risk a single slow request stalling the whole run.
_EMBED_BATCH = 32


logger = logging.getLogger(__name__)


@dataclass
class IndexStats:
    """Per-run counters, surfaced by the CLI after indexing."""

    files_scanned: int = 0
    files_skipped: int = 0        # denylisted, too big, binary, etc.
    files_added: int = 0
    files_changed: int = 0
    files_removed: int = 0
    files_unchanged: int = 0
    chunks_written: int = 0
    chunks_deleted: int = 0
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.files_scanned} scanned | "
            f"+{self.files_added} added · ~{self.files_changed} changed · "
            f"-{self.files_removed} removed · ={self.files_unchanged} unchanged | "
            f"{self.chunks_written} chunks written, "
            f"{self.chunks_deleted} chunks deleted"
            + (f" | errors: {len(self.errors)}" if self.errors else "")
        )


class RAGIndexer:
    """Drives an incremental index of a workspace into a vector store.

    The indexer is stateless — construct, call :meth:`index`, discard.
    Long-lived clients (Redis, Embedder) are passed in so the caller
    controls their lifecycle.
    """

    def __init__(
        self,
        root: Path,
        store: RedisVectorStore,
        embedder: Any,  # gemma.embeddings.Embedder (duck-typed for tests)
        *,
        extensions: Sequence[str] = _DEFAULT_EXTENSIONS,
        max_file_size: int = _MAX_FILE_SIZE,
    ):
        self._root = Path(root).resolve()
        self._store = store
        self._embedder = embedder
        self._extensions = tuple(ext.lower() for ext in extensions)
        self._max_file_size = max_file_size
        self._policy = _safety.default_policy(self._root)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index(
        self,
        *,
        progress: Optional[Callable[[str], None]] = None,
        force_hash: bool = False,
    ) -> IndexStats:
        """Walk the workspace, diff against the stored manifest, apply changes.

        ``progress`` is an optional callback invoked for high-level
        milestones ("walking", "embedding batch 2/5", "saving
        manifest") so the CLI can render a status line without this
        module depending on Rich.

        ``force_hash`` bypasses the mtime+size fast path and always
        recomputes sha1 — useful for paranoid callers or after a
        filesystem clock reset.
        """
        stats = IndexStats()
        _p = progress or (lambda _msg: None)

        # Load the prior manifest before walking so probe_from_disk can
        # skip sha1 computation when mtime+size are unchanged (#2).
        _p("loading prior manifest")
        prior_manifest = Manifest.from_redis_hash(self._store.load_manifest_hash())

        # --- 1. Walk ---
        _p("walking workspace")
        current_entries: List[FileEntry] = []
        for abs_path in self._walk():
            stats.files_scanned += 1
            try:
                rel = abs_path.relative_to(self._root).as_posix()
                prior_entry = None if force_hash else prior_manifest.get(rel)
                entry = FileEntry.probe_from_disk(self._root, abs_path, prior=prior_entry)
            except OSError as exc:
                stats.errors.append(f"stat failed: {abs_path}: {exc}")
                stats.files_skipped += 1
                continue
            current_entries.append(entry)

        new_manifest = Manifest({e.path: e for e in current_entries})

        # --- 2. Diff ---
        _p("diffing against stored manifest")
        diff = prior_manifest.diff(new_manifest)
        stats.files_added = len(diff.added)
        stats.files_changed = len(diff.changed)
        stats.files_removed = len(diff.removed)
        stats.files_unchanged = len(diff.unchanged)

        # Short-circuit when there's nothing to do: preserves the prior
        # manifest as-is so the timestamps in it remain authoritative.
        if not diff.has_work:
            _p("nothing to do")
            return stats

        # --- 3. Delete ---
        _p("deleting stale chunks")
        stats.chunks_deleted += self._apply_deletes(diff)

        # --- 4. Re-chunk + embed ---
        _p("chunking + embedding")
        updated = self._apply_upserts(diff, new_manifest, progress=_p, stats=stats)

        # --- 5. Persist manifest + meta ---
        _p("saving manifest")
        self._store.save_manifest_hash(updated.to_redis_hash())
        # Record namespace meta so future runs can detect model drift.
        try:
            dim = _probe_embedding_dim(self._embedder)
            self._store.set_meta(dim=dim, model=getattr(self._embedder, "model", "unknown"))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("could not record namespace meta: %s", exc)

        return stats

    # ------------------------------------------------------------------
    # Walk
    # ------------------------------------------------------------------

    def _walk(self) -> Iterable[Path]:
        """Yield candidate file paths under ``self._root``.

        Filters: denylisted dirs, denylisted paths, non-allowlisted
        extensions, oversized files, unreadable files.
        """
        # os.walk-style via Path.rglob — pathlib handles symlinks the
        # same way as the underlying stat, which is what we want.
        stack: List[Path] = [self._root]
        while stack:
            current = stack.pop()
            try:
                children = sorted(current.iterdir())
            except OSError as exc:
                logger.debug("skipping unreadable dir %s: %s", current, exc)
                continue

            for child in children:
                if child.is_symlink():
                    # Symlinks can loop; skip them rather than recurse.
                    continue
                if child.is_dir():
                    if child.name in _SKIP_DIRS:
                        continue
                    if _safety.is_denylisted(child, self._policy):
                        continue
                    stack.append(child)
                    continue
                # Regular files.
                if child.suffix.lower() not in self._extensions:
                    continue
                if _safety.is_denylisted(child, self._policy):
                    continue
                try:
                    if child.stat().st_size > self._max_file_size:
                        continue
                except OSError:
                    continue
                yield child

    # ------------------------------------------------------------------
    # Deletes
    # ------------------------------------------------------------------

    def _apply_deletes(self, diff: ManifestDiff) -> int:
        """Drop chunks for every removed path and every changed path's old chunks."""
        deleted = 0
        for old_entry in diff.removed:
            deleted += self._store.delete_file(old_entry.path)
        for old_entry, _new_entry in diff.changed:
            # Use the per-file reverse index rather than the chunk_ids
            # embedded in the manifest — both should agree, but the
            # reverse index is authoritative because upsert populates
            # it atomically with the chunk write.
            deleted += self._store.delete_file(old_entry.path)
        return deleted

    # ------------------------------------------------------------------
    # Upserts
    # ------------------------------------------------------------------

    def _apply_upserts(
        self,
        diff: ManifestDiff,
        new_manifest: Manifest,
        *,
        progress: Callable[[str], None],
        stats: IndexStats,
    ) -> Manifest:
        """Chunk + embed every added/changed file; return the final manifest.

        The returned manifest has ``chunk_ids`` populated for newly
        indexed files and inherits chunk_ids from ``diff.unchanged``
        for the rest.
        """
        # Collect work items and their chunks up-front so we can batch
        # the embedding call efficiently across files.
        pending: list[tuple[FileEntry, list[Chunk]]] = []

        for entry in diff.added:
            chunks = self._chunk_file(entry)
            if chunks is not None:
                pending.append((entry, chunks))

        for _old, new_entry in diff.changed:
            chunks = self._chunk_file(new_entry)
            if chunks is not None:
                pending.append((new_entry, chunks))

        # Flatten into one big list of (entry, chunk) so we can batch
        # embed in fixed-size slices regardless of file boundaries.
        flat: list[tuple[FileEntry, Chunk]] = [
            (entry, c) for entry, chunks in pending for c in chunks
        ]
        total_batches = (len(flat) + _EMBED_BATCH - 1) // _EMBED_BATCH

        # Per-entry accumulator for chunk_ids written.
        new_ids_by_path: dict[str, list[str]] = {}

        for batch_no, start in enumerate(range(0, len(flat), _EMBED_BATCH), start=1):
            batch = flat[start : start + _EMBED_BATCH]
            if not batch:
                break
            progress(f"embedding batch {batch_no}/{total_batches}")
            texts = [_embed_input(entry, chunk) for entry, chunk in batch]
            try:
                vectors = self._embedder.embed_batch(texts)
            except Exception as exc:
                stats.errors.append(f"embed_batch failed: {exc}")
                logger.exception("embed_batch failed")
                continue

            for (entry, chunk), vec in zip(batch, vectors):
                if vec is None or vec.size == 0:
                    stats.errors.append(f"empty embedding for {entry.path}:{chunk.id}")
                    continue
                normalised = normalise(vec)
                try:
                    self._store.upsert_chunk(
                        chunk_id=chunk.id,
                        path=entry.path,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                        text=chunk.text,
                        header=chunk.header,
                        embedding=normalised,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    stats.errors.append(f"upsert failed for {chunk.id}: {exc}")
                    continue
                new_ids_by_path.setdefault(entry.path, []).append(chunk.id)
                stats.chunks_written += 1

        # Rebuild the final manifest: unchanged entries keep their
        # prior chunk_ids (they were already carried through in diff),
        # added/changed entries get the newly written ids.
        final = Manifest()
        for entry in diff.unchanged:
            final.set(entry)
        for entry in diff.added:
            final.set(FileEntry(
                path=entry.path, mtime_ns=entry.mtime_ns, size=entry.size,
                sha1=entry.sha1, chunk_ids=new_ids_by_path.get(entry.path, []),
            ))
        for _old, new_entry in diff.changed:
            final.set(FileEntry(
                path=new_entry.path, mtime_ns=new_entry.mtime_ns,
                size=new_entry.size, sha1=new_entry.sha1,
                chunk_ids=new_ids_by_path.get(new_entry.path, []),
            ))
        return final

    def _chunk_file(self, entry: FileEntry) -> Optional[List[Chunk]]:
        """Read + chunk one file. Returns None if the read/chunk fails."""
        abs_path = self._root / entry.path
        try:
            source = abs_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.debug("skipping unreadable file %s: %s", abs_path, exc)
            return None

        try:
            return chunk_for_path(source, entry.path)
        except Exception as exc:
            logger.warning("chunking failed for %s: %s", entry.path, exc)
            return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _embed_input(entry: FileEntry, chunk: Chunk) -> str:
    """Augment chunk text with a short header so retrieval is scope-aware.

    A chunk from ``handlers/auth.py``, function ``login``, benefits
    from embedding the name "``auth.py::login``" alongside the code —
    queries like "auth login flow" then have extra signal beyond what
    the raw tokens carry.
    """
    header = chunk.header or ""
    if header:
        return f"[{entry.path} :: {header}]\n{chunk.text}"
    return f"[{entry.path}]\n{chunk.text}"


def _probe_embedding_dim(embedder: Any) -> int:
    """Best-effort probe of the embedding dimension for namespace meta.

    We ask the embedder for a single short input. If anything goes
    wrong we return 0 rather than raising — the dim meta is advisory.
    """
    try:
        out = embedder.embed("dim-probe")
        return int(out.shape[0]) if hasattr(out, "shape") else int(len(out))
    except Exception:
        return 0
