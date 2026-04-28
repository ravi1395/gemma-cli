"""SQLite-backed RAG vector store — drop-in replacement for the Redis one.

Mirrors :class:`gemma.rag.store.RedisVectorStore`'s public surface so
``gemma rag {index,query,status,reset}`` and the indexer pipeline work
without changes regardless of backend.

Implementation notes
--------------------
* **Single SQLite file**, multiple namespaces. The ``namespace`` column
  on every RAG table partitions one workspace from the next, so a
  developer with five repos has one ``store.sqlite`` and five
  namespaces inside it.
* **Vector storage** — raw ``float32`` BLOBs in ``rag_embeddings``.
  Cascade delete from ``rag_chunks`` keeps them in sync.
* **Vector search** — load all embeddings for the namespace, stack as a
  numpy matrix, dot-product against the (already L2-normalised) query.
  Same algorithm the Redis path used; just sourced from a SQL query
  instead of a Redis MGET. At ~10k chunks this is single-digit ms.
* **Embed-cache TTL** — handled by ``expires_at`` + ``sweep_expired``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from gemma.rag.store import StoredChunk, StoreSnapshot
from gemma.storage.sqlite_db import open_db, sweep_expired

if TYPE_CHECKING:
    from gemma.config import Config


class SQLiteRAGStore:
    """SQLite implementation of the RAG vector-store contract.

    Constructor diverges from the Redis variant in that it reads the
    SQLite path off ``config`` instead of taking a ``redis_url``. The
    factory in :mod:`gemma.storage` papers over that difference.
    """

    def __init__(
        self,
        config: "Config",
        namespace: str,
        *,
        pool=None,   # accepted, ignored — for factory parity
    ) -> None:
        _ = pool
        self._namespace = namespace
        self._conn = open_db(config)
        # Snapshot the cache cap once — config is otherwise stateless
        # for this store, so we don't need to keep a reference to it.
        # ``0`` (or negative) disables the cap.
        self._embed_cache_max = max(0, int(getattr(
            config, "embed_cache_max_entries", 0,
        )))

    # ------------------------------------------------------------------
    # Connection / metadata
    # ------------------------------------------------------------------

    @property
    def namespace(self) -> str:
        return self._namespace

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
        """Insert or replace one chunk + its (already-normalised) embedding."""
        self._conn.execute(
            """
            INSERT INTO rag_chunks(namespace, chunk_id, path, start_line, end_line, text, header)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(namespace, chunk_id) DO UPDATE SET
              path = excluded.path,
              start_line = excluded.start_line,
              end_line = excluded.end_line,
              text = excluded.text,
              header = excluded.header
            """,
            (
                self._namespace, chunk_id, path,
                int(start_line), int(end_line),
                text, header or "",
            ),
        )
        arr = np.asarray(embedding, dtype=np.float32)
        self._conn.execute(
            """
            INSERT INTO rag_embeddings(namespace, chunk_id, vector, dim)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(namespace, chunk_id) DO UPDATE SET
              vector = excluded.vector,
              dim = excluded.dim
            """,
            (self._namespace, chunk_id, arr.tobytes(), int(arr.size)),
        )
        self._conn.commit()

    def delete_chunk(self, chunk_id: str) -> None:
        # ON DELETE CASCADE on rag_embeddings kills the vector too.
        self._conn.execute(
            "DELETE FROM rag_chunks WHERE namespace = ? AND chunk_id = ?",
            (self._namespace, chunk_id),
        )
        self._conn.commit()

    def delete_file(self, path: str) -> int:
        """Remove every chunk for a file. Returns the count deleted."""
        cur = self._conn.execute(
            "DELETE FROM rag_chunks WHERE namespace = ? AND path = ?",
            (self._namespace, path),
        )
        n = cur.rowcount or 0
        self._conn.commit()
        return n

    # ------------------------------------------------------------------
    # Manifest / meta
    # ------------------------------------------------------------------

    def load_manifest_hash(self) -> Dict[str, str]:
        # Stream the cursor — for a 10k-file manifest this avoids
        # holding the ``rows`` list and the resulting dict in memory at
        # the same time.
        cursor = self._conn.execute(
            "SELECT path, blob_sha FROM rag_manifest WHERE namespace = ?",
            (self._namespace,),
        )
        return {r["path"]: r["blob_sha"] for r in cursor}

    def save_manifest_hash(self, blobs: Dict[str, str]) -> None:
        """Replace the manifest atomically: delete-all then insert-batch."""
        self._conn.execute(
            "DELETE FROM rag_manifest WHERE namespace = ?", (self._namespace,)
        )
        if blobs:
            self._conn.executemany(
                """
                INSERT INTO rag_manifest(namespace, path, blob_sha)
                VALUES (?, ?, ?)
                """,
                [
                    (self._namespace, p, sha)
                    for p, sha in blobs.items()
                ],
            )
        self._conn.commit()

    def set_meta(self, dim: int, model: str) -> None:
        rows = [
            (self._namespace, "dim", str(int(dim))),
            (self._namespace, "model", model),
            (self._namespace, "last_indexed_at", str(int(time.time()))),
        ]
        self._conn.executemany(
            """
            INSERT INTO rag_meta(namespace, key, value)
            VALUES (?, ?, ?)
            ON CONFLICT(namespace, key) DO UPDATE SET value = excluded.value
            """,
            rows,
        )
        self._conn.commit()

    def get_meta(self) -> Dict[str, str]:
        rows = self._conn.execute(
            "SELECT key, value FROM rag_meta WHERE namespace = ?",
            (self._namespace,),
        ).fetchall()
        return {r["key"]: r["value"] for r in rows}

    def snapshot(self) -> "StoreSnapshot":
        """Single-query summary used by ``gemma rag status``."""
        meta = self.get_meta()
        manifest_size = int(self._conn.execute(
            "SELECT COUNT(*) AS c FROM rag_manifest WHERE namespace = ?",
            (self._namespace,),
        ).fetchone()["c"])
        chunk_count = self.chunk_count()
        return StoreSnapshot(
            meta=meta,
            manifest_size=manifest_size,
            chunk_count=chunk_count,
        )

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def chunk_count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS c FROM rag_chunks WHERE namespace = ?",
            (self._namespace,),
        ).fetchone()
        return int(row["c"]) if row else 0

    def all_chunk_ids(self) -> List[str]:
        cursor = self._conn.execute(
            "SELECT chunk_id FROM rag_chunks WHERE namespace = ?",
            (self._namespace,),
        )
        return [r["chunk_id"] for r in cursor]

    def get_chunk(self, chunk_id: str) -> Optional[StoredChunk]:
        row = self._conn.execute(
            """
            SELECT chunk_id, path, start_line, end_line, text, header
              FROM rag_chunks
             WHERE namespace = ? AND chunk_id = ?
            """,
            (self._namespace, chunk_id),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_chunk(row)

    def get_chunks(self, chunk_ids: List[str]) -> List[Optional[StoredChunk]]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" * len(chunk_ids))
        cursor = self._conn.execute(
            f"""
            SELECT chunk_id, path, start_line, end_line, text, header
              FROM rag_chunks
             WHERE namespace = ? AND chunk_id IN ({placeholders})
            """,
            (self._namespace, *chunk_ids),
        )
        by_id = {r["chunk_id"]: self._row_to_chunk(r) for r in cursor}
        # Preserve caller order; ``None`` for missing IDs (matches Redis path).
        return [by_id.get(cid) for cid in chunk_ids]

    def get_embedding(self, chunk_id: str) -> Optional[np.ndarray]:
        row = self._conn.execute(
            """
            SELECT vector FROM rag_embeddings
             WHERE namespace = ? AND chunk_id = ?
            """,
            (self._namespace, chunk_id),
        ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row["vector"], dtype=np.float32)

    def load_all_embeddings(self) -> Dict[str, np.ndarray]:
        """Single SELECT vs Redis's MGET; same shape returned.

        For an index of 10k+ chunks this avoids holding the raw row
        list and the parsed-vector dict in memory simultaneously.
        """
        cursor = self._conn.execute(
            """
            SELECT chunk_id, vector FROM rag_embeddings WHERE namespace = ?
            """,
            (self._namespace,),
        )
        return {
            r["chunk_id"]: np.frombuffer(r["vector"], dtype=np.float32)
            for r in cursor
        }

    # ------------------------------------------------------------------
    # Search (cosine via numpy, identical to the Redis path)
    # ------------------------------------------------------------------

    def search(
        self, query: np.ndarray, k: int = 5,
    ) -> List[StoredChunk]:
        hits, _ = self.search_with_embeddings(query, k=k)
        return hits

    def search_with_embeddings(
        self, query: np.ndarray, k: int = 5,
    ) -> Tuple[List[StoredChunk], Dict[str, np.ndarray]]:
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

        # Stack values directly; dict iteration order is preserved so
        # ``cids[i]`` aligns with ``matrix[i]`` without per-key hashing.
        cids = list(embeds)
        matrix = np.stack(list(embeds.values()), axis=0)
        scores = matrix @ q

        if k >= len(cids):
            top_idx = np.argsort(-scores)
        else:
            top_idx = np.argpartition(-scores, k)[:k]
            top_idx = top_idx[np.argsort(-scores[top_idx])]

        ordered_cids = [cids[int(i)] for i in top_idx]
        scored = {cids[int(i)]: float(scores[int(i)]) for i in top_idx}
        fetched = self.get_chunks(ordered_cids)

        results: List[StoredChunk] = []
        for cid, sc in zip(ordered_cids, fetched):
            if sc is None:
                continue
            sc.score = scored[cid]
            results.append(sc)
        return results, embeds

    # ------------------------------------------------------------------
    # Embed cache (content-hash → vector) — same surface as Redis path
    # ------------------------------------------------------------------

    def mget_embed_cache(
        self, model: str, hashes: List[str],
    ) -> List[Optional[np.ndarray]]:
        if not hashes:
            return []
        sweep_expired(self._conn)
        placeholders = ",".join("?" * len(hashes))
        cursor = self._conn.execute(
            f"""
            SELECT content_hash, vector
              FROM embed_cache
             WHERE model = ? AND content_hash IN ({placeholders})
               AND expires_at > ?
            """,
            (model, *hashes, time.time()),
        )
        by_hash = {
            r["content_hash"]: np.frombuffer(r["vector"], dtype=np.float32)
            for r in cursor
        }
        return [by_hash.get(h) for h in hashes]

    def mset_embed_cache(
        self,
        model: str,
        vectors: Dict[str, np.ndarray],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        if not vectors:
            return
        # Default TTL when caller doesn't pin one — 30 days, matching the
        # Redis path's recommended behaviour. Negative or zero means
        # "no TTL" → use an absurdly far future expires_at.
        far_future = 32503680000.0  # year 3000-ish
        ttl = int(ttl_seconds or 0)
        now = time.time()
        expires_at = far_future if ttl <= 0 else now + ttl

        rows = []
        for content_hash, vec in vectors.items():
            arr = np.asarray(vec, dtype=np.float32)
            rows.append((
                model, content_hash, arr.tobytes(),
                int(arr.size), expires_at,
            ))
        self._conn.executemany(
            """
            INSERT INTO embed_cache(model, content_hash, vector, dim, expires_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(model, content_hash) DO UPDATE SET
              vector = excluded.vector,
              dim = excluded.dim,
              expires_at = excluded.expires_at
            """,
            rows,
        )
        # FIFO eviction: if this batch pushed total cached entries past
        # the configured cap, drop the oldest (smallest ``expires_at``,
        # which monotonically tracks insertion time at a fixed TTL).
        # Done *after* the insert so a single huge batch is allowed to
        # land before being trimmed back down to the cap.
        self._enforce_embed_cache_cap()
        self._conn.commit()

    def _enforce_embed_cache_cap(self) -> None:
        """Trim the embed cache back to ``embed_cache_max_entries``.

        No-op when the cap is disabled (``0``) or the table is already
        within budget. Eviction is FIFO via ``expires_at`` — that field
        is monotonic across writes at a fixed TTL, so it doubles as an
        insertion-order proxy without needing a separate timestamp
        column.
        """
        cap = self._embed_cache_max
        if cap <= 0:
            return
        row = self._conn.execute(
            "SELECT COUNT(*) AS c FROM embed_cache"
        ).fetchone()
        total = int(row["c"]) if row else 0
        excess = total - cap
        if excess <= 0:
            return
        # SQLite supports DELETE with ORDER BY + LIMIT only when built
        # with ``SQLITE_ENABLE_UPDATE_DELETE_LIMIT``; the rowid-IN form
        # is portable and uses the existing ``idx_embed_cache_expires``
        # index for the inner SELECT.
        self._conn.execute(
            """
            DELETE FROM embed_cache
             WHERE rowid IN (
               SELECT rowid FROM embed_cache
                ORDER BY expires_at ASC
                LIMIT ?
             )
            """,
            (excess,),
        )

    def embed_cache_stats(
        self, model: Optional[str] = None,
    ) -> Dict[str, Any]:
        sweep_expired(self._conn)
        if model:
            row = self._conn.execute(
                """
                SELECT COUNT(*) AS c, COALESCE(SUM(LENGTH(vector)), 0) AS b
                  FROM embed_cache WHERE model = ?
                """,
                (model,),
            ).fetchone()
            return {
                "total_keys": int(row["c"]),
                "approx_bytes": int(row["b"]),
                "per_model": {model: int(row["c"])} if row["c"] else {},
            }
        rows = self._conn.execute(
            """
            SELECT model, COUNT(*) AS c, COALESCE(SUM(LENGTH(vector)), 0) AS b
              FROM embed_cache GROUP BY model
            """
        ).fetchall()
        per_model = {r["model"]: int(r["c"]) for r in rows}
        return {
            "total_keys": sum(per_model.values()),
            "approx_bytes": int(sum(r["b"] for r in rows)),
            "per_model": per_model,
        }

    def clear_embed_cache(self, model: Optional[str] = None) -> int:
        if model:
            cur = self._conn.execute(
                "DELETE FROM embed_cache WHERE model = ?", (model,)
            )
        else:
            cur = self._conn.execute("DELETE FROM embed_cache")
        n = cur.rowcount or 0
        self._conn.commit()
        return n

    # ------------------------------------------------------------------
    # Admin
    # ------------------------------------------------------------------

    def clear_namespace(self) -> int:
        """Delete every RAG row scoped to this namespace.

        Returns a rough count of deleted rows (chunks + manifest +
        meta). Embeddings cascade with chunks.
        """
        n = 0
        for table in ("rag_chunks", "rag_manifest", "rag_meta"):
            cur = self._conn.execute(
                f"DELETE FROM {table} WHERE namespace = ?", (self._namespace,)
            )
            n += cur.rowcount or 0
        self._conn.commit()
        return n

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_chunk(row) -> StoredChunk:
        return StoredChunk(
            id=row["chunk_id"],
            path=row["path"],
            start_line=int(row["start_line"]),
            end_line=int(row["end_line"]),
            text=row["text"],
            header=row["header"] or None,
        )

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
