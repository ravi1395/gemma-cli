"""Benchmark ``RAGRetriever.query`` end-to-end.

This is the user-visible latency of ``gemma rag query``. It bundles:

* ``Embedder.embed(q)``        — stubbed out here, near-zero.
* ``store.search(q, fetch_k)`` — internally loads all embeddings.
* ``store.load_all_embeddings()`` again, for MMR.
* ``_mmr()`` — Python loop today.

The whole call is the target of improvement #1 in
``docs/perf/improvements.md`` ("one ``load_all_embeddings`` per
query, not two") and #5 ("vectorized MMR"). Run these before and
after each change; a win here is a win for every user.
"""

from __future__ import annotations

import pytest

from gemma.rag.retrieval import RAGRetriever
from tests.bench.conftest import CORPUS_SIZES


@pytest.mark.parametrize("n_chunks", CORPUS_SIZES)
@pytest.mark.parametrize("k", [5, 20])
def test_query_end_to_end(benchmark, store_factory, embedder, n_chunks, k):
    """Whole pipeline, MMR-on."""
    store = store_factory(n_chunks)
    retriever = RAGRetriever(store, embedder)
    benchmark(retriever.query, "bench query", k=k, mmr_lambda=0.5)


@pytest.mark.parametrize("n_chunks", CORPUS_SIZES)
def test_query_mmr_off(benchmark, store_factory, embedder, n_chunks):
    """Lambda=1 degenerates MMR to pure cosine — isolates the loop cost."""
    store = store_factory(n_chunks)
    retriever = RAGRetriever(store, embedder)
    # fetch_k=k avoids the 4× pool multiplier so this truly measures
    # the cosine path only.
    benchmark(retriever.query, "bench query", k=5, mmr_lambda=1.0, fetch_k=5)
