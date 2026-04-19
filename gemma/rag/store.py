"""Redis-backed vector store for RAG chunks.

Design choice: **plain Redis + client-side cosine**
---------------------------------------------------
We deliberately do NOT use RediSearch's ``FT.SEARCH``/HNSW. Reasons:

1. The memory subsystem already keeps embeddings as raw numpy bytes
   and computes cosine in Python (see :mod:`gemma.memory.store`). Using
   the same approach for RAG means users only need ``redis:7``, not
   ``redis-stack``.
2. ``fakeredis`` supports every command we use here (HSET/SADD/SMEMBERS/
   DEL/GET/HMGET) but does not support RediSearch — reusing the same
   pattern keeps the test suite hermetic.
3. For the sizes we expect in a single repo (roughly 10k–100k chunks,
   768-dim float32 → ~300 MB worst case), a client-side cosine against
   pre-normalised vectors is dominated by NumPy BLAS and runs in tens
   of milliseconds. The operational complexity of RediSearch isn't
   worth it at this scale.

Key layout
----------
All keys are scoped to a namespace returned by
:func:`gemma.rag.namespace.resolve_namespace` so two repos (or two
branches) never share state::

    gemma:rag:{ns}:index                    SET    all active chunk_ids
    gemma:rag:{ns}:chunk:{chunk_id}         HASH   chunk metadata+text
    gemma:rag:{ns}:embed:{chunk_id}         STR    float32 bytes (no nulls)
    gemma:rag:{ns}:manifest                 HASH   path -> JSON FileEntry
    gemma:rag:{ns}:meta                     HASH   {dim, model, last_indexed_at}
    gemma:rag:{ns}:file_chunks:{path_key}   SET    chunk_ids for that path

``file_chunks`` is a reverse index: when a file is removed or replaced,
we need to find and delete every chunk that came from it. Storing a
per-file set avoids scanning the whole index on every delete.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import redis  # type: ignore
except ImportError:  # pragma: no cover -- redis is a declared dep
    redis = None  # type: ignore


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def _k_index(ns: str) -> str:
    return f"gemma:rag:{ns}:index"


def _k_chunk(ns: str, cid: str) -> str:
    return f"gemma:rag:{ns}:chunk:{cid}"


def _k_embed(ns: str, cid: str) -> str:
    return f"gemma:rag:{ns}:embed:{cid}"


def _k_manifest(ns: str) -> str:
    return f"gemma:rag:{ns}:manifest"


def _k_meta(ns: str) -> str:
    return f"gemma:rag:{ns}:meta"


def _k_file_chunks(ns: str, path: str) -> str:
    # Path is already the POSIX-relative form used in the manifest; we
    # embed it verbatim. Redis keys are binary-safe so slashes are fine.
    return f"gemma:rag:{ns}:file_chunks:{path}"


# ---------------------------------------------------------------------------
# Content-hash embedding cache (item #10)
# ---------------------------------------------------------------------------
#
# The cache is deliberately **namespace-agnostic**: if two branches of
# the same repo contain an identical chunk, they should both benefit
# from one embedding. The version prefix (``v1``) lets us evolve the
# ``embed_input`` format later without orphaning existing keys — bump
# to ``v2`` and the old keys just time out via their TTL.
#
# Keys:
#   gemma:rag:embed_cache:v1:{model}:{sha256(embed_input)} -> float32 bytes

#: Pinned key prefix used by mget/mset helpers and by the CLI admin
#: surface. Exported so scan-based cleanup tools can match it.
EMBED_CACHE_PREFIX = "gemma:rag:embed_cache:v1"


def _k_embed_cache(model: str, content_hash: str) -> str:
    """Return the Redis key for one cached embedding.

    ``model`` is embedded verbatim so a mixed-model install (e.g. a
    profile that flipped to ``mxbai-embed-large``) doesn't silently
    read vectors from the wrong encoder. ``content_hash`` is a 64-char
    lowercase hex digest from sha256.
    """
    return f"{EMBED_CACHE_PREFIX}:{model}:{content_hash}"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class StoreSnapshot:
    """Lightweight read-only view of a namespace captured in one pipeline.

    Replaces three sequential Redis round-trips in ``status_command`` with
    a single pipelined call (#11).
    """

    #: Embedding-model fingerprint set by the indexer (model, dim, last_indexed_at).
    meta: Dict[str, str] = field(default_factory=dict)
    #: Number of files currently recorded in the manifest.
    manifest_size: int = 0
    #: Total number of active chunks in the index SET.
    chunk_count: int = 0


@dataclass
class StoredChunk:
    """A chunk as it comes back from the store.

    Mirrors :class:`gemma.chunking.Chunk` but with an additional
    ``score`` field populated by :meth:`RedisVectorStore.search`.
    """

    id: str
    path: str
    start_line: int
    end_line: int
    text: str
    header: Optional[str]
    score: float = 0.0

    @property
    def line_range(self) -> str:
        if self.start_line == self.end_line:
            return str(self.start_line)
        return f"{self.start_line}-{self.end_line}"


# ---------------------------------------------------------------------------
# RedisVectorStore
# ---------------------------------------------------------------------------

class RedisVectorStore:
    """Stores chunks + embeddings in Redis, scored client-side.

    The store is namespace-scoped: one instance corresponds to one
    ``{root_hash}:{branch}`` workspace. Callers construct a new store
    when switching workspaces.
    """

    def __init__(
        self,
        namespace: str,
        redis_url: str = "redis://localhost:6379/0",
        *,
        client: Optional[Any] = None,
        pool: Optional[Any] = None,
    ):
        """Wire up the store.

        Args:
            namespace: Namespace from :func:`resolve_namespace`.
            redis_url: Used to build a real client when ``client`` is None.
            client:    Pre-built Redis client. Tests inject ``fakeredis``
                       here; production code lets us build our own.
            pool:      Optional shared :class:`redis.ConnectionPool` (#3).
                       When supplied, text + binary clients are built
                       from it so the store doesn't open its own TCP
                       connections.

        Note: Redis clients must be constructed with
        ``decode_responses=True`` so ``hgetall`` returns ``str``
        keys/values. Embeddings are the one binary payload we store —
        we fetch those via a separately-built binary client.
        """
        self._namespace = namespace
        self._redis_url = redis_url
        self._pool = pool
        self._client = client
        self._binary_client: Optional[Any] = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    @property
    def namespace(self) -> str:
        return self._namespace

    def _conn(self) -> Any:
        """Return the decode_responses=True client, lazily built."""
        if self._client is None:
            if redis is None:
                raise RuntimeError(
                    "redis-py not installed. RAG requires the 'memory' extra."
                )
            if self._pool is not None:
                self._client = redis.Redis(connection_pool=self._pool, decode_responses=True)
            else:
                self._client = redis.Redis.from_url(self._redis_url, decode_responses=True)
        return self._client

    def _binary_conn(self) -> Any:
        """Return a client with ``decode_responses=False`` for embedding bytes.

        Two clients are needed because a single client can't decode
        some responses and pass others through. We cache the binary
        client for the life of the store.
        """
        if self._binary_client is None:
            if self._client is not None and hasattr(self._client, "connection_pool"):
                # Reuse the same pool when possible (real redis-py). For
                # fakeredis, fall through to the else branch which
                # assumes the fakeredis client can be used as-is with
                # bytes by calling the ``*_byte`` flavoured methods we
                # use explicitly below.
                try:
                    self._binary_client = type(self._client)(
                        connection_pool=self._client.connection_pool,
                        decode_responses=False,
                    )
                except TypeError:
                    # fakeredis doesn't accept connection_pool kwarg.
                    self._binary_client = self._client
            elif redis is None:
                raise RuntimeError("redis-py not installed. RAG requires the 'memory' extra.")
            else:
                self._binary_client = redis.Redis.from_url(
                    self._redis_url, decode_responses=False,
                )
        return self._binary_client

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def upsert_chunk(
        self,
        chunk_id: str,
        *,
        path: str,
        start_line: int,
        end_line: int,
        text: str,
        header: Optional[str],
        embedding: np.ndarray,
    ) -> None:
        """Insert or replace a single chunk.

        The embedding is stored as raw ``float32`` bytes. Callers should
        L2-normalise vectors **before** upserting so :meth:`search` can
        skip normalisation in its hot loop; the indexer does this.
        """
        c = self._conn()
        bc = self._binary_conn()

        c.hset(
            _k_chunk(self._namespace, chunk_id),
            mapping={
                "path": path,
                "start_line": str(start_line),
                "end_line": str(end_line),
                "text": text,
                "header": header or "",
            },
        )
        # Raw bytes — use the binary client so decode_responses doesn't
        # try to utf-8 decode non-text floats.
        bc.set(_k_embed(self._namespace, chunk_id), embedding.astype(np.float32).tobytes())

        c.sadd(_k_index(self._namespace), chunk_id)
        c.sadd(_k_file_chunks(self._namespace, path), chunk_id)

    def delete_chunk(self, chunk_id: str) -> None:
        """Delete a single chunk and its embedding.

        We fetch the chunk's ``path`` first so we can also evict it
        from the per-file reverse index. If the chunk is already gone,
        this is a no-op.
        """
        c = self._conn()
        bc = self._binary_conn()
        path = c.hget(_k_chunk(self._namespace, chunk_id), "path")
        c.delete(_k_chunk(self._namespace, chunk_id))
        bc.delete(_k_embed(self._namespace, chunk_id))
        c.srem(_k_index(self._namespace), chunk_id)
        if path:
            c.srem(_k_file_chunks(self._namespace, path), chunk_id)

    def delete_file(self, path: str) -> int:
        """Remove every chunk whose source file is ``path``.

        Returns the number of chunks deleted — useful for telemetry
        on the removed/changed branches of the indexer.
        """
        c = self._conn()
        chunks = list(c.smembers(_k_file_chunks(self._namespace, path)))
        for cid in chunks:
            self.delete_chunk(cid)
        c.delete(_k_file_chunks(self._namespace, path))
        return len(chunks)

    # ------------------------------------------------------------------
    # Manifest / meta
    # ------------------------------------------------------------------

    def load_manifest_hash(self) -> Dict[str, str]:
        c = self._conn()
        return dict(c.hgetall(_k_manifest(self._namespace)))

    def save_manifest_hash(self, blobs: Dict[str, str]) -> None:
        """Replace the manifest hash atomically.

        We delete-then-set rather than upserting field-by-field because
        a file that was removed from the workspace must also leave the
        manifest — and ``hset`` has no "replace the whole hash"
        semantic.
        """
        c = self._conn()
        key = _k_manifest(self._namespace)
        pipe = c.pipeline()
        pipe.delete(key)
        if blobs:
            pipe.hset(key, mapping=blobs)
        pipe.execute()

    def set_meta(self, dim: int, model: str) -> None:
        """Record the embedding-model fingerprint for the namespace.

        If a future index run uses a different model or dimension we
        can spot the drift and (caller's decision) clear the namespace
        rather than mix incompatible vectors.
        """
        c = self._conn()
        c.hset(
            _k_meta(self._namespace),
            mapping={
                "dim": str(dim),
                "model": model,
                "last_indexed_at": str(int(time.time())),
            },
        )

    def get_meta(self) -> Dict[str, str]:
        c = self._conn()
        return dict(c.hgetall(_k_meta(self._namespace)))

    def snapshot(self) -> "StoreSnapshot":
        """Fetch meta, manifest size, and chunk count in one Redis pipeline.

        Replaces three sequential round-trips (``get_meta``,
        ``load_manifest_hash`` + ``len``, ``chunk_count``) with a single
        pipelined batch so ``status_command`` pays one network RTT instead
        of three (#11).

        Returns:
            A :class:`StoreSnapshot` populated from the current namespace.
        """
        c = self._conn()
        pipe = c.pipeline()
        pipe.hgetall(_k_meta(self._namespace))
        # HLEN is cheaper than HGETALL when we only need the count.
        pipe.hlen(_k_manifest(self._namespace))
        pipe.scard(_k_index(self._namespace))
        meta_raw, manifest_size, chunk_count = pipe.execute()
        return StoreSnapshot(
            meta=dict(meta_raw) if meta_raw else {},
            manifest_size=int(manifest_size),
            chunk_count=int(chunk_count),
        )

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def chunk_count(self) -> int:
        c = self._conn()
        return int(c.scard(_k_index(self._namespace)))

    def all_chunk_ids(self) -> List[str]:
        c = self._conn()
        return list(c.smembers(_k_index(self._namespace)))

    def get_chunk(self, chunk_id: str) -> Optional[StoredChunk]:
        """Fetch a chunk's metadata (sans embedding)."""
        c = self._conn()
        row = c.hgetall(_k_chunk(self._namespace, chunk_id))
        if not row:
            return None
        return StoredChunk(
            id=chunk_id,
            path=row.get("path", ""),
            start_line=int(row.get("start_line", 0) or 0),
            end_line=int(row.get("end_line", 0) or 0),
            text=row.get("text", ""),
            header=row.get("header") or None,
        )

    def get_chunks(self, chunk_ids: List[str]) -> List[Optional[StoredChunk]]:
        """Pipelined fetch of many chunks in a single round-trip.

        Replaces the per-winner ``get_chunk`` loop in :meth:`search` with
        one batched HGETALL pipeline (#1). Returns a list the same length
        as ``chunk_ids``; entries where the chunk was absent are ``None``.
        """
        if not chunk_ids:
            return []
        c = self._conn()
        pipe = c.pipeline()
        for cid in chunk_ids:
            pipe.hgetall(_k_chunk(self._namespace, cid))
        rows = pipe.execute()
        out: List[Optional[StoredChunk]] = []
        for cid, row in zip(chunk_ids, rows):
            if not row:
                out.append(None)
                continue
            out.append(StoredChunk(
                id=cid,
                path=row.get("path", ""),
                start_line=int(row.get("start_line", 0) or 0),
                end_line=int(row.get("end_line", 0) or 0),
                text=row.get("text", ""),
                header=row.get("header") or None,
            ))
        return out

    def get_embedding(self, chunk_id: str) -> Optional[np.ndarray]:
        """Fetch one embedding. Returns None if the chunk is absent."""
        bc = self._binary_conn()
        raw = bc.get(_k_embed(self._namespace, chunk_id))
        if not raw:
            return None
        return np.frombuffer(raw, dtype=np.float32)

    def load_all_embeddings(self) -> Dict[str, np.ndarray]:
        """Bulk-fetch every embedding for in-process cosine scoring.

        For 10k chunks × 768-dim × 4-byte floats this is ~30 MB, which
        we're comfortable holding in RAM. Callers wanting to bound
        memory should page via a future ``search_paginated`` variant.
        """
        bc = self._binary_conn()
        cids = self.all_chunk_ids()
        if not cids:
            return {}

        # MGET in a single round-trip. Redis keys, not chunk IDs, go on
        # the wire; we reassemble the dict on our side.
        keys = [_k_embed(self._namespace, cid) for cid in cids]
        raws = bc.mget(keys)
        out: Dict[str, np.ndarray] = {}
        for cid, raw in zip(cids, raws):
            if raw:
                out[cid] = np.frombuffer(raw, dtype=np.float32)
        return out

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self, query: np.ndarray, k: int = 5,
    ) -> List[StoredChunk]:
        """Return the top ``k`` chunks by cosine similarity.

        The query is normalised here so callers don't have to remember.
        Stored embeddings are normalised at upsert time, so the cosine
        of two L2-normalised vectors is just their dot product — fast
        and branch-free.
        """
        hits, _embed_map = self.search_with_embeddings(query, k=k)
        return hits

    def search_with_embeddings(
        self, query: np.ndarray, k: int = 5,
    ) -> Tuple[List[StoredChunk], Dict[str, np.ndarray]]:
        """Top-``k`` search that also returns the full pool embedding map (#1).

        Folds what used to be two MGETs per query (one here, one in
        :meth:`RAGRetriever.query` for the MMR re-rank) into a single
        bulk fetch. Callers that don't need the matrix can keep using
        :meth:`search`.

        The winner metadata is fetched via a single pipelined HGETALL
        batch (:meth:`get_chunks`) rather than per-winner round-trips.

        Returns:
            ``(hits, embed_map)``. ``embed_map`` maps every chunk id in
            the namespace to its L2-normalised embedding so the caller
            can compute pairwise similarities without a second MGET.
        """
        if query.size == 0:
            return [], {}

        q = query.astype(np.float32)
        qnorm = float(np.linalg.norm(q))
        if qnorm == 0.0:
            return [], {}
        q = q / qnorm

        embeds = self.load_all_embeddings()
        if not embeds:
            return [], {}

        cids = list(embeds.keys())
        matrix = np.stack([embeds[c] for c in cids], axis=0)  # (N, D)
        # Dot product against pre-normalised vectors. Cheap and exact.
        scores = matrix @ q

        # Pick top k without sorting the full array.
        if k >= len(cids):
            top_idx = np.argsort(-scores)
        else:
            top_idx = np.argpartition(-scores, k)[:k]
            top_idx = top_idx[np.argsort(-scores[top_idx])]

        ordered_cids = [cids[int(i)] for i in top_idx]
        scored = {cids[int(i)]: float(scores[int(i)]) for i in top_idx}
        # One pipelined HGETALL instead of k sequential round-trips.
        fetched = self.get_chunks(ordered_cids)

        results: List[StoredChunk] = []
        for cid, sc in zip(ordered_cids, fetched):
            if sc is None:
                continue
            sc.score = scored[cid]
            results.append(sc)
        return results, embeds

    # ------------------------------------------------------------------
    # Embedding cache (item #10)
    # ------------------------------------------------------------------

    def mget_embed_cache(
        self, model: str, hashes: List[str],
    ) -> List[Optional[np.ndarray]]:
        """Bulk-fetch cached embeddings for a list of content hashes.

        Always returns a list the same length as ``hashes``; entries
        where the cache missed (or held a zero-length value) are
        ``None``. Uses the binary client so Redis doesn't try to utf-8
        decode float bytes.

        Args:
            model:  Embedding-model tag recorded in the key — keeps the
                    cache honest across profile switches.
            hashes: Lowercase hex digests (sha256) of ``embed_input``.
        """
        if not hashes:
            return []
        bc = self._binary_conn()
        keys = [_k_embed_cache(model, h) for h in hashes]
        raws = bc.mget(keys)
        return [
            np.frombuffer(r, dtype=np.float32) if r else None
            for r in raws
        ]

    def mset_embed_cache(
        self,
        model: str,
        vectors: Dict[str, np.ndarray],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Bulk-store embeddings keyed by content hash, optionally with TTL.

        Writes happen in a single pipelined round-trip. Each vector is
        serialised to its ``float32`` byte representation — callers can
        reverse this with ``np.frombuffer(raw, dtype=np.float32)``.

        Args:
            model:       Embedding-model tag. Must match the one used
                         by :meth:`mget_embed_cache` on the read side.
            vectors:     ``{content_hash: vector}`` mapping. Empty
                         input is a silent no-op.
            ttl_seconds: Expiry for each key. ``None`` or ``<= 0``
                         writes with no TTL (persist until evicted).
        """
        if not vectors:
            return
        bc = self._binary_conn()
        pipe = bc.pipeline()
        for content_hash, vec in vectors.items():
            key = _k_embed_cache(model, content_hash)
            raw = vec.astype(np.float32).tobytes()
            if ttl_seconds and ttl_seconds > 0:
                pipe.set(key, raw, ex=ttl_seconds)
            else:
                pipe.set(key, raw)
        pipe.execute()

    def embed_cache_stats(
        self, model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return a small summary of the embed cache.

        Used by ``gemma rag cache stats`` — O(keys) under the hood
        because we have to ``SCAN`` to count. For the cache sizes we
        expect (tens of thousands of keys) this is still <100 ms.

        Args:
            model: If given, restricts the scan to keys for that
                   specific model. Otherwise summarises across models.

        Returns:
            ``{"total_keys": int, "approx_bytes": int, "per_model":
            {model: count}}``. ``approx_bytes`` is a fast
            ``STRLEN`` sum — it omits Redis's own per-key overhead.
        """
        c = self._conn()
        pattern = (
            f"{EMBED_CACHE_PREFIX}:{model}:*"
            if model
            else f"{EMBED_CACHE_PREFIX}:*"
        )
        per_model: Dict[str, int] = {}
        total_keys = 0
        approx_bytes = 0
        # Stream scan + STRLEN in a pipeline to bound RTTs.
        for key in c.scan_iter(match=pattern, count=500):
            total_keys += 1
            # key looks like gemma:rag:embed_cache:v1:<model>:<hash>
            parts = key.split(":", 5)
            if len(parts) >= 6:
                key_model = parts[4]
                per_model[key_model] = per_model.get(key_model, 0) + 1
            try:
                approx_bytes += int(c.strlen(key))
            except Exception:  # pragma: no cover - defensive
                pass
        return {
            "total_keys": total_keys,
            "approx_bytes": approx_bytes,
            "per_model": per_model,
        }

    def clear_embed_cache(self, model: Optional[str] = None) -> int:
        """Drop every embed-cache key, optionally filtered by model.

        Uses ``scan_iter`` rather than ``KEYS`` so a huge cache doesn't
        block the Redis server. Deletes are pipelined.

        Args:
            model: If given, restrict deletion to that model's keys.

        Returns:
            The number of keys actually deleted.
        """
        c = self._conn()
        pattern = (
            f"{EMBED_CACHE_PREFIX}:{model}:*"
            if model
            else f"{EMBED_CACHE_PREFIX}:*"
        )
        count = 0
        pipe = c.pipeline()
        for key in c.scan_iter(match=pattern, count=500):
            pipe.delete(key)
            count += 1
        pipe.execute()
        return count

    # ------------------------------------------------------------------
    # Admin
    # ------------------------------------------------------------------

    def clear_namespace(self) -> int:
        """Nuke every RAG key for this namespace.

        Returns the number of keys deleted. Used by ``gemma rag reset``
        (future) and by tests that want a clean slate.
        """
        c = self._conn()
        pattern = f"gemma:rag:{self._namespace}:*"
        # scan_iter so we don't block the server on huge namespaces.
        count = 0
        pipe = c.pipeline()
        for key in c.scan_iter(match=pattern, count=500):
            pipe.delete(key)
            count += 1
        pipe.execute()
        return count
