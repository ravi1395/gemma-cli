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

import hashlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

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
    # --- Item #10: cache telemetry ---
    # Embed-cache hits counted at the chunk level (not batch level), so
    # ``chunks_cache_hit`` + actual embed calls = total chunks needing
    # vectors in this run. Surfaced by the CLI summary so users can see
    # that a "re-index after branch switch" run did mostly cache reads.
    chunks_cache_hit: int = 0
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        cache_note = (
            f" | cache_hits={self.chunks_cache_hit}"
            if self.chunks_cache_hit
            else ""
        )
        return (
            f"{self.files_scanned} scanned | "
            f"+{self.files_added} added · ~{self.files_changed} changed · "
            f"-{self.files_removed} removed · ={self.files_unchanged} unchanged | "
            f"{self.chunks_written} chunks written, "
            f"{self.chunks_deleted} chunks deleted"
            + cache_note
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
        concurrency: int = 1,
        embedder_factory: Optional[Callable[[], Any]] = None,
        cache_enabled: bool = False,
        cache_ttl_seconds: Optional[int] = None,
    ):
        """Wire up an indexer.

        Args:
            root:              Workspace root; walked recursively.
            store:             Vector store to upsert into.
            embedder:          Primary embedder — used for the serial
                               fast path, the dim probe, and as a
                               fallback when ``embedder_factory`` is
                               not supplied.
            extensions:        File-extension allowlist.
            max_file_size:     Skip files larger than this (bytes).
            concurrency:       Item #9. When ``> 1`` and there is more
                               than one embed batch, fan the embed
                               calls out across a ``ThreadPoolExecutor``
                               of this size. Each worker uses its own
                               embedder (built via
                               ``embedder_factory``) so HTTP sessions
                               don't serialise. Clamped to [1, 16].
            embedder_factory:  Item #9. Zero-arg callable returning a
                               fresh embedder for one worker thread.
                               Required when ``concurrency > 1``;
                               absent factories fall back to the shared
                               primary embedder (acceptable for
                               in-memory stubs; not ideal for real
                               Ollama).
            cache_enabled:     Item #10. When True, consult the
                               store's embed cache before each embed
                               call and write misses back.
            cache_ttl_seconds: Item #10. TTL written on cache misses;
                               ``None`` or ``<= 0`` means "no expiry".
        """
        self._root = Path(root).resolve()
        self._store = store
        self._embedder = embedder
        self._extensions = tuple(ext.lower() for ext in extensions)
        self._max_file_size = max_file_size
        self._policy = _safety.default_policy(self._root)
        # #9 knobs. Clamp concurrency to a sane range so a misconfigured
        # profile cannot create a thread storm.
        self._concurrency = max(1, min(16, int(concurrency)))
        self._embedder_factory = embedder_factory
        # #10 knobs.
        self._cache_enabled = bool(cache_enabled)
        self._cache_ttl_seconds = (
            int(cache_ttl_seconds)
            if cache_ttl_seconds and cache_ttl_seconds > 0
            else None
        )

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

        Embed calls are fanned out across a bounded thread pool when
        ``concurrency > 1`` and there's more than one batch worth of
        work; otherwise the serial path is preserved untouched so the
        micro-benchmarked hot path is unchanged (#9).

        When ``cache_enabled`` is True, every batch first consults the
        store's content-hash embed cache and only calls the embedder
        for the misses. Hits are copied into the result vector list
        (#10).
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
        if not flat:
            # Nothing to embed — skip straight to manifest rebuild.
            return self._rebuild_manifest(diff, {})

        # Materialise the batch list up-front so both the serial and
        # parallel paths iterate the same structure.
        batches: list[list[tuple[FileEntry, Chunk]]] = [
            flat[i : i + _EMBED_BATCH]
            for i in range(0, len(flat), _EMBED_BATCH)
        ]
        total_batches = len(batches)

        # Embed all batches; vector_batches[i] is either a list of
        # numpy arrays (one per chunk in batches[i]) or None if the
        # embed call for that batch failed outright.
        vector_batches = self._embed_all_batches(
            batches, progress=progress, stats=stats, total=total_batches,
        )

        # Per-entry accumulator for chunk_ids written.
        new_ids_by_path: dict[str, list[str]] = {}

        # Upsert is sequential — Redis writes serialise on the wire
        # anyway, and this keeps ``stats`` and ``new_ids_by_path``
        # free of locking concerns.
        for batch, vectors in zip(batches, vector_batches):
            if vectors is None:
                # embed_batch failed for this slice; skip its chunks.
                continue
            for (entry, chunk), vec in zip(batch, vectors):
                if vec is None or vec.size == 0:
                    stats.errors.append(
                        f"empty embedding for {entry.path}:{chunk.id}"
                    )
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
                    stats.errors.append(
                        f"upsert failed for {chunk.id}: {exc}"
                    )
                    continue
                new_ids_by_path.setdefault(entry.path, []).append(chunk.id)
                stats.chunks_written += 1

        return self._rebuild_manifest(diff, new_ids_by_path)

    # ------------------------------------------------------------------
    # Embed orchestration (items #9, #10)
    # ------------------------------------------------------------------

    def _embed_all_batches(
        self,
        batches: List[List[Tuple[FileEntry, Chunk]]],
        *,
        progress: Callable[[str], None],
        stats: IndexStats,
        total: int,
    ) -> List[Optional[List[np.ndarray]]]:
        """Embed every batch, preserving order. Items #9 + #10.

        The serial path is selected when concurrency is 1 or only one
        batch exists — the executor setup cost dwarfs the saving for
        a single-batch run.

        Returns a list of length ``total`` where each entry is either
        the list of vectors for that batch or ``None`` if the embed
        call failed.
        """
        if self._concurrency <= 1 or total <= 1:
            # ---- Serial fast path -----------------------------------
            # Matches the pre-#9 behaviour byte-for-byte so the
            # microbench doesn't regress on single-batch or
            # concurrency-disabled configurations.
            out: List[Optional[List[np.ndarray]]] = []
            for batch_no, batch in enumerate(batches, start=1):
                progress(f"embedding batch {batch_no}/{total}")
                out.append(self._embed_one_batch(
                    batch, embedder=self._embedder, stats=stats,
                ))
            return out

        # ---- Concurrent path ----------------------------------------
        # One embedder per worker thread, memoised via threading.local
        # so the factory is invoked at most ``concurrency`` times even
        # if the work queue is much longer.
        tls = threading.local()

        def _embedder_for_thread() -> Any:
            emb = getattr(tls, "embedder", None)
            if emb is not None:
                return emb
            factory = self._embedder_factory
            emb = factory() if factory is not None else self._embedder
            tls.embedder = emb
            return emb

        def _run(batch: List[Tuple[FileEntry, Chunk]]) -> Optional[List[np.ndarray]]:
            return self._embed_one_batch(
                batch, embedder=_embedder_for_thread(), stats=stats,
            )

        # ``executor.map`` preserves submission order when consumed in
        # order — same guarantee we rely on for tool-call fan-out (#20).
        workers = min(self._concurrency, total)
        results: List[Optional[List[np.ndarray]]] = []
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="gemma-embed",
        ) as executor:
            for batch_no, vectors in enumerate(
                executor.map(_run, batches), start=1,
            ):
                progress(f"embedding batch {batch_no}/{total}")
                results.append(vectors)
        return results

    def _embed_one_batch(
        self,
        batch: List[Tuple[FileEntry, Chunk]],
        *,
        embedder: Any,
        stats: IndexStats,
    ) -> Optional[List[np.ndarray]]:
        """Embed one batch with cache consultation (item #10).

        Returns a list of vectors in the same order as ``batch``, or
        ``None`` if the underlying embed call failed catastrophically
        and no vectors could be produced. Per-chunk failures are
        represented as zero-size arrays so the caller can skip them
        individually.
        """
        texts = [_embed_input(entry, chunk) for entry, chunk in batch]

        # --- Fast path: no cache ---------------------------------
        if not self._cache_enabled:
            try:
                return list(embedder.embed_batch(texts))
            except Exception as exc:
                stats.errors.append(f"embed_batch failed: {exc}")
                logger.exception("embed_batch failed")
                return None

        # --- Cached path ----------------------------------------
        model = str(getattr(embedder, "model", "unknown"))
        hashes = [_content_hash(t) for t in texts]

        try:
            cached = self._store.mget_embed_cache(model, hashes)
        except Exception as exc:
            # Cache read failure should degrade to a plain embed call,
            # not abort the whole run. Log once per failing batch.
            logger.warning("embed cache mget failed: %s", exc)
            cached = [None] * len(hashes)

        miss_indices = [i for i, vec in enumerate(cached) if vec is None]

        # --- Everything hit — skip embedder call entirely. -----
        if not miss_indices:
            stats.chunks_cache_hit += len(batch)
            return list(cached)  # type: ignore[arg-type]

        # --- Embed misses in a single batched call. ------------
        miss_texts = [texts[i] for i in miss_indices]
        try:
            fresh = embedder.embed_batch(miss_texts)
        except Exception as exc:
            stats.errors.append(f"embed_batch failed: {exc}")
            logger.exception("embed_batch failed")
            return None

        # --- Merge + write-through -----------------------------
        out: List[np.ndarray] = list(cached)  # type: ignore[assignment]
        to_cache: dict[str, np.ndarray] = {}
        for local_idx, global_idx in enumerate(miss_indices):
            vec = fresh[local_idx] if local_idx < len(fresh) else None
            out[global_idx] = vec  # type: ignore[assignment]
            if vec is not None and getattr(vec, "size", 0) > 0:
                to_cache[hashes[global_idx]] = vec

        hits = len(batch) - len(miss_indices)
        if hits > 0:
            stats.chunks_cache_hit += hits

        if to_cache:
            try:
                self._store.mset_embed_cache(
                    model, to_cache, ttl_seconds=self._cache_ttl_seconds,
                )
            except Exception as exc:  # pragma: no cover - defensive
                # Don't abort the run if cache write fails; the vectors
                # will simply be recomputed next time.
                logger.warning("embed cache mset failed: %s", exc)

        return out

    def _rebuild_manifest(
        self,
        diff: ManifestDiff,
        new_ids_by_path: dict[str, list[str]],
    ) -> Manifest:
        """Assemble the post-run manifest from diff + fresh chunk ids.

        Factored out of :meth:`_apply_upserts` so the no-work short
        circuit and the happy path share one code path for the
        manifest side-effect.
        """
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

def _content_hash(embed_input: str) -> str:
    """Content-addressable key for the embed cache (#10).

    The hash is over the exact string that will be sent to the
    embedder — i.e. including the ``[path :: header]`` prefix — so a
    chunk moved to a different file gets a different key. That's the
    correct behaviour: the retrieval signal for ``handlers/auth.py``
    differs from the same code at ``legacy/auth.py``.
    """
    return hashlib.sha256(embed_input.encode("utf-8")).hexdigest()


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
