"""Tests for :class:`RAGRetriever` — embedding, top-k, MMR diversity.

MMR is the interesting knob here: we stage a store where three
candidates are nearly identical (east-facing) and one is orthogonal,
then demonstrate that:

  * ``lambda=1`` picks the three near-duplicates (pure relevance).
  * ``lambda=0.3`` pulls in the orthogonal one early to diversify.

No real embeddings are used; the stub embedder returns vectors we
know exactly.
"""

from __future__ import annotations

import fakeredis
import numpy as np
import pytest

from gemma.rag.retrieval import RAGRetriever, _mmr, _normalise
from gemma.rag.store import RedisVectorStore, StoredChunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class StubEmbedder:
    """Returns a fixed vector per input. No hashing — purely table-driven."""

    model = "stub"

    def __init__(self, mapping: dict[str, np.ndarray]):
        self._mapping = mapping

    def embed(self, text: str) -> np.ndarray:
        # Unknown queries map to a deterministic "unknown" vector so
        # tests that don't care about this path still get a sensible
        # return.
        return self._mapping.get(text, np.array([0.0, 0.0, 1.0], dtype=np.float32))

    def embed_batch(self, texts):  # pragma: no cover - not used here
        return [self.embed(t) for t in texts]


@pytest.fixture
def store_with_chunks():
    """Store pre-populated with four chunks in a known geometry."""
    server = fakeredis.FakeServer()
    text_client = fakeredis.FakeRedis(server=server, decode_responses=True)
    bin_client = fakeredis.FakeRedis(server=server, decode_responses=False)
    s = RedisVectorStore(namespace="ns_retr", client=text_client)
    s._binary_client = bin_client

    # Three near-duplicates pointing east and one pointing north.
    east1 = np.array([1.0, 0.02], dtype=np.float32)
    east2 = np.array([1.0, 0.01], dtype=np.float32)
    east3 = np.array([1.0, 0.03], dtype=np.float32)
    north = np.array([0.0, 1.0], dtype=np.float32)

    for cid, vec, line in [
        ("e1", east1, 10), ("e2", east2, 20), ("e3", east3, 30), ("n1", north, 40),
    ]:
        s.upsert_chunk(
            chunk_id=cid, path="a.py",
            start_line=line, end_line=line + 5,
            text=f"text for {cid}", header=cid,
            embedding=vec / np.linalg.norm(vec),  # pre-normalise
        )
    return s


# ---------------------------------------------------------------------------
# _mmr unit tests
# ---------------------------------------------------------------------------

def test_mmr_lambda_one_equals_raw_cosine(store_with_chunks):
    """With lambda=1 MMR ranks by pure relevance (ties broken by order)."""
    embed_map = store_with_chunks.load_all_embeddings()
    # Candidates in score-descending order — as the store would return.
    q = _normalise(np.array([1.0, 0.0], dtype=np.float32))
    raw_hits = store_with_chunks.search(q, k=4)
    selected = _mmr(
        candidates=raw_hits, embed_map=embed_map,
        query_vec=q, k=3, lam=1.0,
    )
    # The three east-facing chunks should all come before "n1".
    selected_ids = [c.id for c in selected]
    assert "n1" not in selected_ids
    assert set(selected_ids) == {"e1", "e2", "e3"}


def test_mmr_low_lambda_diversifies(store_with_chunks):
    """With lambda=0.3 MMR should reach for the orthogonal chunk."""
    embed_map = store_with_chunks.load_all_embeddings()
    q = _normalise(np.array([1.0, 0.0], dtype=np.float32))
    raw_hits = store_with_chunks.search(q, k=4)
    selected = _mmr(
        candidates=raw_hits, embed_map=embed_map,
        query_vec=q, k=3, lam=0.3,
    )
    selected_ids = [c.id for c in selected]
    # First pick is still the top east-facing chunk (relevance beats
    # diversity on a fresh set). But by the third pick, the orthogonal
    # "n1" should appear.
    assert "n1" in selected_ids


def test_mmr_respects_k(store_with_chunks):
    embed_map = store_with_chunks.load_all_embeddings()
    q = _normalise(np.array([1.0, 0.0], dtype=np.float32))
    raw = store_with_chunks.search(q, k=4)
    selected = _mmr(
        candidates=raw, embed_map=embed_map,
        query_vec=q, k=2, lam=0.5,
    )
    assert len(selected) == 2


def test_mmr_empty_candidates_returns_empty():
    assert _mmr(
        candidates=[], embed_map={}, query_vec=np.zeros(2, dtype=np.float32),
        k=5, lam=0.5,
    ) == []


# ---------------------------------------------------------------------------
# RAGRetriever end-to-end
# ---------------------------------------------------------------------------

def test_query_returns_retrieval_hits(store_with_chunks):
    q_vec = np.array([1.0, 0.0], dtype=np.float32)
    embedder = StubEmbedder({"east": q_vec})
    retriever = RAGRetriever(store_with_chunks, embedder)

    hits = retriever.query("east", k=3)
    assert len(hits) == 3
    # All returned as RetrievalHit with correct geometry.
    for h in hits:
        assert h.path == "a.py"
        assert h.end_line > h.start_line
        # line_range / citation helpers work.
        assert "-" in h.line_range
        assert h.citation.startswith("a.py:")


def test_query_empty_string_returns_nothing(store_with_chunks):
    retriever = RAGRetriever(store_with_chunks, StubEmbedder({}))
    assert retriever.query("   ", k=5) == []


def test_query_returns_empty_when_embedder_fails(store_with_chunks):
    class BrokenEmbedder:
        model = "broken"

        def embed(self, _t):
            raise RuntimeError("ollama down")

    retriever = RAGRetriever(store_with_chunks, BrokenEmbedder())
    # Graceful degradation — caller sees an empty list, not an exception.
    assert retriever.query("anything", k=3) == []


def test_query_against_empty_store():
    """Retriever on an empty store never calls MMR."""
    server = fakeredis.FakeServer()
    text_client = fakeredis.FakeRedis(server=server, decode_responses=True)
    bin_client = fakeredis.FakeRedis(server=server, decode_responses=False)
    empty_store = RedisVectorStore(namespace="ns_empty", client=text_client)
    empty_store._binary_client = bin_client

    retriever = RAGRetriever(
        empty_store,
        StubEmbedder({"q": np.array([1.0, 0.0], dtype=np.float32)}),
    )
    assert retriever.query("q") == []
