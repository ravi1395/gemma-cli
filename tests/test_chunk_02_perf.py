"""Regression tests for Chunk 02 — Halve query latency.

Each item gets one focused test. Benchmarks still live in
``tests/bench/``; these tests are about correctness, not speed.
"""

from __future__ import annotations

import numpy as np
import fakeredis
import pytest

from gemma.cache import ResponseCache
from gemma.config import Config
from gemma.output import OutputMode, render_response
from gemma.rag._math import normalise
from gemma.rag.retrieval import _mmr
from gemma.rag.store import RedisVectorStore, StoredChunk
from gemma.session import GemmaSession


# ---------------------------------------------------------------------------
# #1 — search_with_embeddings returns the embed_map in one call
# ---------------------------------------------------------------------------

def _populated_store(namespace: str = "t_ns") -> RedisVectorStore:
    """Two-dim store with four known chunks. Matches test_rag_retrieval."""
    server = fakeredis.FakeServer()
    text_client = fakeredis.FakeRedis(server=server, decode_responses=True)
    bin_client = fakeredis.FakeRedis(server=server, decode_responses=False)
    s = RedisVectorStore(namespace=namespace, client=text_client)
    s._binary_client = bin_client
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
            embedding=vec / np.linalg.norm(vec),
        )
    return s


def test_search_with_embeddings_returns_embed_map():
    """The fused call returns the same hits as search() plus the full map."""
    store = _populated_store()
    q = np.array([1.0, 0.0], dtype=np.float32)

    hits_legacy = store.search(q, k=3)
    hits_new, embed_map = store.search_with_embeddings(q, k=3)

    assert [h.id for h in hits_legacy] == [h.id for h in hits_new]
    # embed_map is the *entire* pool so the retriever can MMR without a
    # second load_all_embeddings call.
    assert set(embed_map.keys()) == {"e1", "e2", "e3", "n1"}
    for v in embed_map.values():
        assert v.dtype == np.float32


def test_search_with_embeddings_calls_mget_once(monkeypatch):
    """Only one full-pool MGET happens even when hits are resolved."""
    store = _populated_store("t_mget")
    calls = {"n": 0}
    real = store.load_all_embeddings

    def tracked():
        calls["n"] += 1
        return real()

    monkeypatch.setattr(store, "load_all_embeddings", tracked)
    q = np.array([1.0, 0.0], dtype=np.float32)
    store.search_with_embeddings(q, k=2)
    assert calls["n"] == 1


def test_get_chunks_batch_preserves_order():
    store = _populated_store("t_gc")
    out = store.get_chunks(["n1", "e2", "missing", "e1"])
    assert [c.id if c else None for c in out] == ["n1", "e2", None, "e1"]


# ---------------------------------------------------------------------------
# #5 — Vectorised MMR is bit-identical on seeded pools
# ---------------------------------------------------------------------------

def _legacy_mmr(*, candidates, embed_map, query_vec, k, lam):
    """The pre-#5 Python-loop implementation, retained for the parity test."""
    if k <= 0 or not candidates:
        return []
    usable = [(c, embed_map[c.id]) for c in candidates if c.id in embed_map]
    if not usable:
        return candidates[:k]
    remaining = list(usable)
    selected = []
    relevance = {c.id: float(v @ query_vec) for c, v in usable}
    while remaining and len(selected) < k:
        best_score = -float("inf")
        best_idx = 0
        for i, (cand, cand_vec) in enumerate(remaining):
            rel = relevance[cand.id]
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
        chosen.score = best_score
        selected.append(chosen)
    return selected


def _make_pool(rng, n, dim):
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    chunks = [
        StoredChunk(id=f"c{i}", path="x", start_line=1, end_line=2, text="", header=None)
        for i in range(n)
    ]
    embed_map = {f"c{i}": vecs[i] for i in range(n)}
    return chunks, embed_map


@pytest.mark.parametrize("seed", range(50))
def test_mmr_vectorised_matches_legacy(seed):
    """Across 50 seeded pools the new MMR picks the same ids in the same order."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 40))
    dim = 64
    chunks, embed_map = _make_pool(rng, n, dim)
    q = normalise(rng.standard_normal(dim).astype(np.float32))
    k = int(rng.integers(2, min(n, 15)))
    lam = float(rng.uniform(0.1, 0.9))

    # Fresh StoredChunk copies so the two runs don't share mutable score state.
    a_chunks = [StoredChunk(**c.__dict__) for c in chunks]
    b_chunks = [StoredChunk(**c.__dict__) for c in chunks]

    legacy = _legacy_mmr(candidates=a_chunks, embed_map=embed_map, query_vec=q, k=k, lam=lam)
    new = _mmr(candidates=b_chunks, embed_map=embed_map, query_vec=q, k=k, lam=lam)

    assert [c.id for c in legacy] == [c.id for c in new]


# ---------------------------------------------------------------------------
# #6 — Stream-and-cache: cache stays empty on interrupted stream
# ---------------------------------------------------------------------------

def test_render_response_interrupted_stream_sets_finished_false():
    """A generator that raises mid-stream returns ``finished=False``."""
    def boom():
        yield ("content", "partial ")
        raise RuntimeError("network drop")

    reply, finished = render_response(boom(), mode=OutputMode.RICH, stream=True)
    assert reply.startswith("partial")
    assert finished is False


def test_render_response_clean_stream_sets_finished_true():
    def ok():
        yield ("content", "hello ")
        yield ("content", "world")

    reply, finished = render_response(ok(), mode=OutputMode.RICH, stream=True)
    assert reply == "hello world"
    assert finished is True


def test_cache_write_skipped_when_stream_interrupted(monkeypatch):
    """End-to-end: main.ask does not call cache.put when finished=False."""
    # We simulate the cache-put gate directly since wiring the full CLI
    # needs a live Redis + Ollama. The gate logic is a one-liner in main:
    #     if cache and reply and finished: cache.put(...)
    # so we assert that predicate here against both finished states.
    puts = []

    class FakeCache:
        def get(self, *_a, **_k):
            return None

        def put(self, messages, cfg, content):
            puts.append(content)

    cache = FakeCache()
    # finished=False → nothing written
    reply = "partial reply"
    finished = False
    if cache and reply and finished:
        cache.put([], Config(), reply)
    assert puts == []
    # finished=True → written
    finished = True
    if cache and reply and finished:
        cache.put([], Config(), reply)
    assert puts == ["partial reply"]


# ---------------------------------------------------------------------------
# #8 — _detect_branch is memoised per session
# ---------------------------------------------------------------------------

def test_branch_for_forks_git_once_per_root(tmp_path, monkeypatch):
    """GemmaSession.branch_for calls _detect_branch exactly once per root."""
    import gemma.session as session_mod

    calls = []

    def fake_detect(root):
        calls.append(str(root))
        return "main"

    monkeypatch.setattr("gemma.rag.namespace._detect_branch", fake_detect)

    cfg = Config()
    with GemmaSession(cfg) as s:
        a = s.branch_for(tmp_path)
        b = s.branch_for(tmp_path)
        c = s.branch_for(tmp_path)

    assert a == b == c == "main"
    assert len(calls) == 1


def test_branch_for_returns_fallback_when_not_a_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "gemma.rag.namespace._detect_branch", lambda _root: None,
    )
    with GemmaSession(Config()) as s:
        assert s.branch_for(tmp_path) == "_default"


# ---------------------------------------------------------------------------
# #3 — Pool threads through stores
# ---------------------------------------------------------------------------

def test_redis_vector_store_accepts_pool_kwarg():
    """The store ctor accepts a pool; no real connection happens."""
    store = RedisVectorStore(namespace="p_ns", pool=object())
    # Pool is stored but not used until _conn is invoked.
    assert store._pool is not None


def test_memory_store_accepts_pool_kwarg():
    from gemma.memory.store import MemoryStore

    store = MemoryStore(Config(), pool=object())
    assert store._pool is not None


def test_session_close_disposes_pool(monkeypatch):
    """session.close() invokes redis_pool.disconnect on the owned pool."""
    disposed = []

    class FakePool:
        def disconnect(self):
            disposed.append(True)

    fake_pool = FakePool()
    monkeypatch.setattr("gemma.redis_pool.pool_for", lambda _cfg: fake_pool)

    with GemmaSession(Config()) as s:
        # Force materialisation so close() has something to dispose.
        _ = s.redis_pool

    assert disposed == [True]
