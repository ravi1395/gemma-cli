"""Query the RAG store — cosine top-k + MMR re-ranking.

Why MMR
-------
Plain cosine top-k tends to produce near-duplicate hits: when a model
asks "how does auth work?", the five chunks most similar to the query
are often five overlapping pieces of the *same* function. Maximal
Marginal Relevance (Carbonell & Goldstein, 1998) trades a small amount
of raw relevance for diversity by scoring each candidate against both

* its similarity to the query, and
* its similarity to the hits we've already selected.

Control parameter ``lambda ∈ [0, 1]``:

* ``lambda = 1``  → behaviour identical to plain cosine top-k.
* ``lambda = 0``  → pick the most-diverse chunks regardless of
                    relevance (rarely useful).
* ``lambda ≈ 0.5`` → good default; surface relevant *and* varied
                     chunks.

We pull a larger candidate pool (``fetch_k = k * 4``) first and then
MMR-shrink it to ``k``. This keeps the quadratic similarity work
inside the smaller pool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from gemma.rag.store import RedisVectorStore, StoredChunk


#: Default MMR diversity weight. 0.5 balances relevance and novelty.
_DEFAULT_LAMBDA = 0.5

#: Default candidate-pool multiplier: ``fetch_k = k * this``. Larger
#: values give MMR more room to diversify at the cost of extra dot
#: products in-process.
_FETCH_MULTIPLIER = 4


@dataclass
class RetrievalHit:
    """One retrieval result, ready for rendering or for prompt injection."""

    chunk_id: str
    path: str
    start_line: int
    end_line: int
    text: str
    header: Optional[str]
    score: float

    @property
    def line_range(self) -> str:
        if self.start_line == self.end_line:
            return str(self.start_line)
        return f"{self.start_line}-{self.end_line}"

    @property
    def citation(self) -> str:
        """Short citation footer — ``gemma/rag/store.py:42-97``."""
        return f"{self.path}:{self.line_range}"


class RAGRetriever:
    """Top-k retriever backed by :class:`RedisVectorStore`.

    Stateless beyond the injected dependencies. Construct once per
    workspace (so we pass the right namespace-scoped store) and call
    :meth:`query` as often as needed.
    """

    def __init__(self, store: RedisVectorStore, embedder: Any):
        self._store = store
        self._embedder = embedder

    def query(
        self,
        q: str,
        *,
        k: int = 5,
        mmr_lambda: float = _DEFAULT_LAMBDA,
        fetch_k: Optional[int] = None,
    ) -> List[RetrievalHit]:
        """Run a RAG query. Returns up to ``k`` MMR-diversified hits.

        Args:
            q: Natural-language query.
            k: Maximum number of hits to return.
            mmr_lambda: MMR trade-off weight (see module docstring).
            fetch_k: Candidate-pool size. Defaults to ``k *
                :data:`_FETCH_MULTIPLIER```. Callers who want raw
                cosine top-k can pass ``fetch_k=k, mmr_lambda=1.0``.
        """
        if not q.strip():
            return []

        try:
            query_vec = self._embedder.embed(q)
        except Exception:
            # Embedder failure (model not pulled, Ollama down) is
            # recoverable — the caller should degrade gracefully.
            return []

        if query_vec.size == 0:
            return []

        pool = fetch_k if fetch_k is not None else max(k, k * _FETCH_MULTIPLIER)

        # Phase 1: plain top-pool by cosine. The store normalises the
        # query vector itself, so we can hand it through raw.
        candidates = self._store.search(query_vec, k=pool)
        if not candidates:
            return []

        # Phase 2: MMR on the candidates. We need each candidate's
        # *embedding* for the pairwise similarities; fetch them in bulk.
        embed_map = self._store.load_all_embeddings()
        if not embed_map:
            return []

        qvec = _normalise(np.asarray(query_vec, dtype=np.float32))
        selected = _mmr(
            candidates=candidates,
            embed_map=embed_map,
            query_vec=qvec,
            k=min(k, len(candidates)),
            lam=mmr_lambda,
        )

        return [
            RetrievalHit(
                chunk_id=c.id,
                path=c.path,
                start_line=c.start_line,
                end_line=c.end_line,
                text=c.text,
                header=c.header,
                score=c.score,
            )
            for c in selected
        ]


# ---------------------------------------------------------------------------
# MMR
# ---------------------------------------------------------------------------

def _mmr(
    *,
    candidates: List[StoredChunk],
    embed_map: dict[str, np.ndarray],
    query_vec: np.ndarray,
    k: int,
    lam: float,
) -> List[StoredChunk]:
    """Select ``k`` candidates via Maximal Marginal Relevance.

    Assumes stored vectors are L2-normalised (the indexer guarantees
    this), so dot-product = cosine. Also assumes ``query_vec`` is
    already normalised.
    """
    if k <= 0 or not candidates:
        return []

    # Filter candidates that don't have an embedding available.
    usable = [(c, embed_map[c.id]) for c in candidates if c.id in embed_map]
    if not usable:
        # Fall through to plain order; better than returning empty.
        return candidates[:k]

    remaining = list(usable)
    selected: List[StoredChunk] = []

    # Pre-compute relevance (cosine to query) once.
    relevance = {c.id: float(v @ query_vec) for c, v in usable}

    while remaining and len(selected) < k:
        best_score = -float("inf")
        best_idx = 0
        for i, (cand, cand_vec) in enumerate(remaining):
            rel = relevance[cand.id]
            # Similarity to the most-similar already-selected item.
            if selected:
                sims = [float(cand_vec @ embed_map[s.id]) for s in selected if s.id in embed_map]
                max_sim = max(sims) if sims else 0.0
            else:
                max_sim = 0.0
            mmr_score = lam * rel - (1.0 - lam) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i
        chosen, _cv = remaining.pop(best_idx)
        # Attach the MMR-adjusted score so callers see the trade-off.
        chosen.score = best_score
        selected.append(chosen)

    return selected


def _normalise(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm
