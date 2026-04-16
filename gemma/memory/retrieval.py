"""Embedding-based semantic retrieval of condensed memories.

Given a user query, we embed it, then compute cosine similarity against all
stored memory embeddings. Top-K above a similarity threshold are returned.

At the expected scale (<= low thousands of memories) a single vectorized
numpy matmul takes well under a millisecond on Apple Silicon, so there's
no need for HNSW / FAISS / RedisSearch yet.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from gemma.config import Config
from gemma.embeddings import Embedder
from gemma.memory.models import MemoryRecord
from gemma.memory.store import MemoryStore


class MemoryRetriever:
    """Find the memories most relevant to a query via cosine similarity."""

    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder,
        config: Config,
    ):
        self._store = store
        self._embedder = embedder
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_relevant(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
    ) -> list[tuple[MemoryRecord, float]]:
        """Return top-K (record, similarity) pairs above the threshold."""
        k = top_k if top_k is not None else self._config.memory_top_k
        threshold = (
            min_similarity
            if min_similarity is not None
            else self._config.memory_min_similarity
        )
        if k <= 0 or not query:
            return []

        try:
            query_vec = self._embedder.embed(query)
        except Exception:
            return []
        if query_vec.size == 0:
            return []

        embeddings = self._store.get_all_embeddings()
        if not embeddings:
            return []

        ids = list(embeddings.keys())
        matrix = np.stack([embeddings[mid] for mid in ids])
        sims = self._cosine_similarity_batch(query_vec, matrix)

        # Rank and filter
        order = np.argsort(-sims)  # descending
        results: list[tuple[MemoryRecord, float]] = []
        for idx in order:
            score = float(sims[idx])
            if score < threshold:
                break
            mid = ids[idx]
            record = self._store.get_memory(mid, bump_access=True)
            if record is not None and record.is_active():
                results.append((record, score))
            if len(results) >= k:
                break
        return results

    def find_conflicting(
        self,
        query_text: str,
        *,
        threshold: Optional[float] = None,
    ) -> list[tuple[MemoryRecord, float]]:
        """Find existing memories near-duplicate to a new candidate statement.

        Uses a higher similarity threshold than general retrieval because we
        want only plausible contradictions (not weakly related memories).
        """
        effective = (
            threshold
            if threshold is not None
            else self._config.memory_conflict_threshold
        )
        # Pull an unbounded top-K with the raised threshold.
        return self.find_relevant(
            query_text, top_k=50, min_similarity=effective
        )

    # ------------------------------------------------------------------
    # Math helpers
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Scalar cosine similarity. Zero vectors -> 0.0."""
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    @staticmethod
    def _cosine_similarity_batch(q: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Vectorized cosine sim: 1-D query against 2-D (N, D) matrix."""
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0.0:
            return np.zeros(matrix.shape[0], dtype=np.float32)
        row_norms = np.linalg.norm(matrix, axis=1)
        # Avoid division by zero for any empty row
        safe_norms = np.where(row_norms == 0.0, 1.0, row_norms)
        dots = matrix @ q
        sims = dots / (safe_norms * q_norm)
        # Zero-rows -> 0 similarity (not NaN)
        sims = np.where(row_norms == 0.0, 0.0, sims)
        return sims.astype(np.float32)
