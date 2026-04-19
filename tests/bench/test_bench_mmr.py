"""Benchmark ``gemma.rag.retrieval._mmr``.

Why it matters
--------------
MMR is the per-query diversification step. The current implementation
is a Python ``while``-loop computing pairwise dot products one at a
time. At small candidate pools this is invisible; at larger pools it
scales as O(k × fetch_k × D) with a Python overhead constant that
swamps the NumPy work.

Benchmarks here target the proposed vectorized-MMR rewrite. Three
parameter axes are covered so we can spot the crossover point at
which the Python loop becomes the bottleneck:

* ``fetch_k``: candidate pool (16, 64, 256)
* ``k``:       final selection (5, 20)
* ``D``:       embedding dimension (768 — nomic's native)
"""

from __future__ import annotations

import numpy as np
import pytest

from gemma.rag._math import normalise as _normalise
from gemma.rag.retrieval import _mmr
from gemma.rag.store import StoredChunk


def _make_candidates(fetch_k: int, dim: int, rng: np.random.Generator):
    """Build a (fetch_k, dim) candidate pool of L2-normalised vectors."""
    vecs = rng.standard_normal((fetch_k, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    chunks = [
        StoredChunk(
            id=f"c{i}", path="x", start_line=1, end_line=2,
            text="", header=None, score=float(vecs[i, 0]),
        )
        for i in range(fetch_k)
    ]
    embed_map = {f"c{i}": vecs[i] for i in range(fetch_k)}
    return chunks, embed_map


@pytest.mark.parametrize("fetch_k", [16, 64, 256])
@pytest.mark.parametrize("k", [5, 20])
def test_mmr_python_loop(benchmark, fetch_k, k):
    """Current Python-loop MMR — baseline we will improve against."""
    rng = np.random.default_rng(42)
    dim = 768
    chunks, embed_map = _make_candidates(fetch_k, dim, rng)
    q = _normalise(rng.standard_normal(dim).astype(np.float32))

    # ``k`` must not exceed the pool — skip impossible combos.
    if k > fetch_k:
        pytest.skip("k > fetch_k")

    benchmark(
        _mmr,
        candidates=chunks, embed_map=embed_map,
        query_vec=q, k=k, lam=0.5,
    )
