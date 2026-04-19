"""Benchmark ``RedisVectorStore`` hot paths.

These benchmarks are the backstop for improvements #1 and #3 in
``docs/perf/improvements.md`` — they prove (or disprove) that a
given refactor speeds up the code *in isolation* before we touch
the caller.

Covered hot paths:

* ``load_all_embeddings`` — MGET + np.frombuffer across N keys.
  Called twice per query today (#1).
* ``search`` — load_all_embeddings + matrix-vector dot + argpartition.
* ``get_chunk`` — per-winner HGETALL, run k times inside ``search``.
* ``chunk_count`` — one SCARD, cheap but frequently called.

Each is parametrized over the three canonical corpus sizes. Expect
the 10k numbers to dominate the picture: the MGET wire payload is
~30 MB there, whereas at 100 chunks we're essentially benchmarking
Python call overhead.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.bench.conftest import CORPUS_SIZES


# ---------------------------------------------------------------------------
# load_all_embeddings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_chunks", CORPUS_SIZES)
def test_load_all_embeddings(benchmark, store_factory, n_chunks):
    """MGET → numpy dict. The single most expensive RAG read."""
    store = store_factory(n_chunks)
    benchmark(store.load_all_embeddings)


# ---------------------------------------------------------------------------
# search (cosine top-k)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_chunks", CORPUS_SIZES)
def test_search_topk(benchmark, store_factory, embedder, n_chunks):
    """End-to-end cosine top-k over the whole corpus."""
    store = store_factory(n_chunks)
    q = embedder.embed("bench query")
    benchmark(store.search, q, 5)


@pytest.mark.parametrize("n_chunks", CORPUS_SIZES)
def test_search_large_k(benchmark, store_factory, embedder, n_chunks):
    """k=50 exercises argsort rather than argpartition on small corpora."""
    store = store_factory(n_chunks)
    q = embedder.embed("bench query")
    benchmark(store.search, q, 50)


# ---------------------------------------------------------------------------
# get_chunk / chunk_count
# ---------------------------------------------------------------------------

def test_chunk_count(benchmark, store_factory):
    """Single SCARD — baseline for a Redis round-trip."""
    store = store_factory(1_000)
    benchmark(store.chunk_count)


def test_get_chunk_single(benchmark, store_factory):
    """HGETALL + re-hydration into ``StoredChunk``."""
    store = store_factory(1_000)
    benchmark(store.get_chunk, "c500")


# ---------------------------------------------------------------------------
# snapshot vs. three-reads
# ---------------------------------------------------------------------------

def test_snapshot_pipelined(benchmark, store_factory):
    """Single pipelined batch: meta + manifest size + chunk count (#11)."""
    store = store_factory(1_000)
    benchmark(store.snapshot)


def test_three_reads_sequential(benchmark, store_factory):
    """Baseline: three sequential round-trips that snapshot() replaces."""
    store = store_factory(1_000)

    def _three_reads():
        _meta = store.get_meta()
        _manifest_size = len(store.load_manifest_hash())
        _chunk_count = store.chunk_count()

    benchmark(_three_reads)
