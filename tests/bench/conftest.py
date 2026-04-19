"""Shared fixtures for the performance benchmark suite.

The benchmarks need three things to be reproducible:

* a **populated vector store** at a deterministic size — we build it
  from scratch per-fixture so cold/warm behaviour is identical across
  runs;
* a **stub embedder** so no Ollama call ever happens (embedding latency
  is its own benchmark, not part of the RAG hot-path);
* an **isolated fakeredis server** so no real Redis is required and the
  measurements are free of network jitter.

By holding everything in-memory we are measuring Python + NumPy work,
not I/O. That is exactly what we want: the improvements we plan to
make are algorithmic or architectural, and any win has to survive
the disappearance of network noise.
"""

from __future__ import annotations

import hashlib
from typing import List

import fakeredis
import numpy as np
import pytest

from gemma.rag.store import RedisVectorStore


# ---------------------------------------------------------------------------
# Stub embedder
# ---------------------------------------------------------------------------

class BenchEmbedder:
    """Deterministic, no-network embedder returning 768-dim float32 vectors.

    768 is nomic-embed-text's native dim, so the memory footprint and
    NumPy op sizes match what users see in production.
    """

    model = "bench-embed"
    dim = 768

    def __init__(self) -> None:
        self._rng_cache: dict[str, np.ndarray] = {}

    def _one(self, text: str) -> np.ndarray:
        # Cached so repeated embed(query) calls are free — we are
        # measuring the retrieval pipeline, not the embed hash.
        if text in self._rng_cache:
            return self._rng_cache[text]
        seed = int.from_bytes(hashlib.sha1(text.encode()).digest()[:4], "little")
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.dim).astype(np.float32)
        v /= np.linalg.norm(v) or 1.0
        self._rng_cache[text] = v
        return v

    def embed(self, text: str) -> np.ndarray:
        return self._one(text)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        return [self._one(t) for t in texts]


# ---------------------------------------------------------------------------
# Store builders
# ---------------------------------------------------------------------------

def _make_store(namespace: str = "bench") -> RedisVectorStore:
    """Spin up a fresh fakeredis-backed store."""
    server = fakeredis.FakeServer()
    text_client = fakeredis.FakeRedis(server=server, decode_responses=True)
    bin_client = fakeredis.FakeRedis(server=server, decode_responses=False)
    store = RedisVectorStore(namespace=namespace, client=text_client)
    store._binary_client = bin_client
    return store


def _populate(store: RedisVectorStore, n_chunks: int, embedder: BenchEmbedder) -> None:
    """Populate a store with ``n_chunks`` synthetic normalised vectors.

    We distribute chunks across 100 synthetic "files" so that the
    store's internal ``file_chunks`` reverse index exercises the same
    code path as a real workspace.
    """
    for i in range(n_chunks):
        path = f"f{(i % 100):03d}.py"
        store.upsert_chunk(
            chunk_id=f"c{i}",
            path=path,
            start_line=i, end_line=i + 5,
            text=f"chunk body {i}", header=None,
            embedding=embedder.embed(f"chunk_{i}"),
        )


@pytest.fixture
def embedder() -> BenchEmbedder:
    return BenchEmbedder()


@pytest.fixture
def store_factory(embedder):
    """Return a builder so benchmarks can ask for any size they need.

    We return a factory (not a populated store) because
    ``pytest-benchmark`` will call the setup fn multiple times during
    warm-up; we want a *fresh* store each setup run to eliminate
    carryover state between iterations.
    """
    def _build(n_chunks: int, namespace: str = "bench") -> RedisVectorStore:
        store = _make_store(namespace)
        _populate(store, n_chunks, embedder)
        return store
    return _build


# ---------------------------------------------------------------------------
# Standard corpus sizes
# ---------------------------------------------------------------------------

#: Sizes we benchmark across. Small = small repo / single package;
#: medium = typical microservice; large = mono-repo scale. Keep this
#: list small so the full suite stays under a minute on a laptop.
CORPUS_SIZES = (100, 1_000, 10_000)
