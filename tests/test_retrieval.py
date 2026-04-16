"""Unit tests for the embedding-based MemoryRetriever.

We inject a stub Embedder that returns deterministic vectors so similarity
math is easy to reason about.
"""

from __future__ import annotations

import numpy as np
import pytest

from gemma.config import Config
from gemma.memory.models import MemoryCategory, MemoryRecord
from gemma.memory.retrieval import MemoryRetriever


class StubEmbedder:
    """Maps known strings to fixed vectors; fresh queries -> onehot-ish vector."""

    def __init__(self, mapping: dict[str, np.ndarray]):
        self._mapping = mapping

    @property
    def model(self) -> str:
        return "stub"

    def embed(self, text: str) -> np.ndarray:
        if text in self._mapping:
            return self._mapping[text]
        # Default: a normalized vector that is dissimilar to any stored memory
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def test_cosine_similarity_basic():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert MemoryRetriever.cosine_similarity(a, b) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert MemoryRetriever.cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_returns_zero():
    a = np.zeros(4, dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    assert MemoryRetriever.cosine_similarity(a, b) == 0.0


# ---------------------------------------------------------------------------
# find_relevant
# ---------------------------------------------------------------------------

def _seed_memories(store, records: list[MemoryRecord], vectors: list[np.ndarray]) -> None:
    for rec, vec in zip(records, vectors):
        store.save_memory(rec)
        store.save_embedding(rec.memory_id, vec)


def test_find_relevant_returns_top_k_sorted(store, cfg):
    cfg.memory_top_k = 3
    cfg.memory_min_similarity = 0.0
    records = [
        MemoryRecord(content="python fact", category=MemoryCategory.USER_PREFERENCE, importance=4, session_id="s"),
        MemoryRecord(content="java fact", category=MemoryCategory.FACTUAL_CONTEXT, importance=3, session_id="s"),
        MemoryRecord(content="weather fact", category=MemoryCategory.FACTUAL_CONTEXT, importance=1, session_id="s"),
    ]
    vectors = [
        np.array([1.0, 0.0, 0.0], dtype=np.float32),     # aligned with query
        np.array([0.7, 0.7, 0.0], dtype=np.float32),     # partial overlap
        np.array([0.0, 0.0, 1.0], dtype=np.float32),     # orthogonal
    ]
    _seed_memories(store, records, vectors)

    query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    embedder = StubEmbedder({"python": query_vec})

    retriever = MemoryRetriever(store, embedder, cfg)
    results = retriever.find_relevant("python", top_k=3, min_similarity=0.0)

    assert len(results) == 3
    # Descending by similarity
    sims = [score for _, score in results]
    assert sims == sorted(sims, reverse=True)
    # Most similar first
    assert results[0][0].content == "python fact"


def test_find_relevant_respects_min_similarity(store, cfg):
    records = [
        MemoryRecord(content="close", category=MemoryCategory.FACTUAL_CONTEXT, importance=3, session_id="s"),
        MemoryRecord(content="far", category=MemoryCategory.FACTUAL_CONTEXT, importance=3, session_id="s"),
    ]
    vectors = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),  # orthogonal -> sim 0
    ]
    _seed_memories(store, records, vectors)

    embedder = StubEmbedder({"q": np.array([1.0, 0.0], dtype=np.float32)})
    retriever = MemoryRetriever(store, embedder, cfg)
    results = retriever.find_relevant("q", top_k=5, min_similarity=0.5)

    assert len(results) == 1
    assert results[0][0].content == "close"


def test_find_relevant_no_memories_returns_empty(store, cfg):
    embedder = StubEmbedder({})
    retriever = MemoryRetriever(store, embedder, cfg)
    assert retriever.find_relevant("anything") == []


def test_find_relevant_empty_query_returns_empty(store, cfg, sample_memories):
    for rec in sample_memories:
        store.save_memory(rec)
        store.save_embedding(
            rec.memory_id, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        )
    embedder = StubEmbedder({})
    retriever = MemoryRetriever(store, embedder, cfg)
    assert retriever.find_relevant("") == []


def test_find_relevant_handles_embedding_failure(store, cfg, sample_memories):
    class BrokenEmbedder:
        @property
        def model(self) -> str: return "broken"
        def embed(self, text: str) -> np.ndarray:
            raise RuntimeError("ollama down")

    for rec in sample_memories:
        store.save_memory(rec)
        store.save_embedding(
            rec.memory_id, np.array([1.0, 0.0], dtype=np.float32)
        )
    retriever = MemoryRetriever(store, BrokenEmbedder(), cfg)  # type: ignore[arg-type]
    assert retriever.find_relevant("anything") == []


def test_find_relevant_skips_superseded(store, cfg):
    alive = MemoryRecord(
        content="alive",
        category=MemoryCategory.FACTUAL_CONTEXT,
        importance=3,
        session_id="s",
    )
    store.save_memory(alive)
    store.save_embedding(alive.memory_id, np.array([1.0, 0.0], dtype=np.float32))

    dead = MemoryRecord(
        content="dead",
        category=MemoryCategory.FACTUAL_CONTEXT,
        importance=3,
        session_id="s",
    )
    store.save_memory(dead)
    store.save_embedding(dead.memory_id, np.array([1.0, 0.0], dtype=np.float32))
    store.supersede_memory(dead.memory_id, "replacement")

    embedder = StubEmbedder({"q": np.array([1.0, 0.0], dtype=np.float32)})
    retriever = MemoryRetriever(store, embedder, cfg)
    results = retriever.find_relevant("q", top_k=5, min_similarity=0.0)

    # Superseded memory is removed from the index (and thus from embeddings map)
    contents = [r.content for r, _ in results]
    assert "alive" in contents
    assert "dead" not in contents


# ---------------------------------------------------------------------------
# find_conflicting
# ---------------------------------------------------------------------------

def test_find_conflicting_uses_higher_threshold(store, cfg):
    cfg.memory_conflict_threshold = 0.95
    rec_close = MemoryRecord(
        content="a",
        category=MemoryCategory.FACTUAL_CONTEXT,
        importance=3,
        session_id="s",
    )
    rec_farther = MemoryRecord(
        content="b",
        category=MemoryCategory.FACTUAL_CONTEXT,
        importance=3,
        session_id="s",
    )
    _seed_memories(
        store,
        [rec_close, rec_farther],
        [
            np.array([1.0, 0.0], dtype=np.float32),     # sim 1.0 with query
            np.array([0.8, 0.6], dtype=np.float32),     # sim 0.8 with query
        ],
    )

    embedder = StubEmbedder({"probe": np.array([1.0, 0.0], dtype=np.float32)})
    retriever = MemoryRetriever(store, embedder, cfg)
    hits = retriever.find_conflicting("probe")

    # With threshold 0.95, only the exactly-aligned vector qualifies
    assert [r.content for r, _ in hits] == ["a"]
