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

from gemma.rag._math import normalise
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

    def as_dict(self) -> dict:
        """Return a JSON-serialisable dict for tool-result payloads."""
        return {
            "chunk_id": self.chunk_id,
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "text": self.text,
            "header": self.header,
            "score": float(self.score),
            "citation": self.citation,
        }


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

        # Phase 1+2: one MGET for the top-pool cosine AND the MMR pairwise
        # map. Previously this path fired two full ``load_all_embeddings``
        # calls (~60 MB on the wire at 10k chunks × 768-dim); now it fires
        # one (#1). The store also pipelines the per-winner HGETALLs.
        candidates, embed_map = self._store.search_with_embeddings(query_vec, k=pool)
        if not candidates or not embed_map:
            return []

        qvec = normalise(np.asarray(query_vec, dtype=np.float32))
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

    Vectorised implementation (#5): the candidate pool is pre-stacked
    into a single ``(pool, D)`` matrix and ``max_sim_so_far`` is updated
    each iteration with ``np.maximum(max_sim_so_far, pool @ pool[pick])``
    — one BLAS dot instead of ``pool`` Python-level dot products per
    pick. Tie-breaking is preserved: ``np.argmax`` returns the first
    maximum, matching the old ``>`` loop semantics.
    """
    if k <= 0 or not candidates:
        return []

    # Filter candidates that don't have an embedding available.
    usable: List[StoredChunk] = [c for c in candidates if c.id in embed_map]
    if not usable:
        # Fall through to plain order; better than returning empty.
        return candidates[:k]

    pool = np.stack([embed_map[c.id] for c in usable], axis=0).astype(np.float32)
    relevance = pool @ query_vec.astype(np.float32)
    n = pool.shape[0]
    # Sentinel distinguishes "no selection yet" (max-sim is 0 in the
    # legacy loop) from "negative similarity to a selected item" (the
    # legacy loop keeps negatives, does not clip at zero).
    max_sim: Optional[np.ndarray] = None
    taken = np.zeros(n, dtype=bool)

    selected: List[StoredChunk] = []
    target = min(k, n)
    while len(selected) < target:
        if max_sim is None:
            mmr = lam * relevance
        else:
            mmr = lam * relevance - (1.0 - lam) * max_sim
        mmr[taken] = -np.inf
        pick = int(np.argmax(mmr))
        chosen = usable[pick]
        chosen.score = float(mmr[pick])
        selected.append(chosen)
        taken[pick] = True
        # Update max-similarity-to-selected in one BLAS call. First
        # assignment seeds the vector with the first pick's sims so
        # negative similarities are preserved (matches legacy ``max()``).
        sims = pool @ pool[pick]
        if max_sim is None:
            max_sim = sims
        else:
            np.maximum(max_sim, sims, out=max_sim)

    return selected


