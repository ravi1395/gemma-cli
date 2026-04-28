"""Tests for :class:`gemma.storage.sqlite_rag.SQLiteRAGStore`.

Focus: namespace isolation, vector search correctness, embed-cache TTL.
The SQLite path uses the same numpy cosine algorithm as the Redis path
so we don't re-test the math — we test the storage round-trip and
the SQL pieces that differ.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from gemma.config import Config
from gemma.storage.sqlite_rag import SQLiteRAGStore


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(
        storage_backend="sqlite",
        sqlite_path=str(tmp_path / "store.sqlite"),
    )


@pytest.fixture
def store(cfg) -> SQLiteRAGStore:
    s = SQLiteRAGStore(cfg, namespace="ns1")
    yield s
    s.close()


def _vec(values, dim=4) -> np.ndarray:
    """Build a unit-length float32 vector for repeatable cosine asserts."""
    arr = np.zeros(dim, dtype=np.float32)
    for i, v in enumerate(values):
        arr[i] = float(v)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 0 else arr


# ---------------------------------------------------------------------------
# Upsert + read
# ---------------------------------------------------------------------------

def test_upsert_and_get_chunk(store):
    store.upsert_chunk(
        "c1",
        path="src/main.py",
        start_line=10, end_line=20,
        text="def main():\n    pass",
        header="main()",
        embedding=_vec([1, 0, 0, 0]),
    )
    chunk = store.get_chunk("c1")
    assert chunk is not None
    assert chunk.path == "src/main.py"
    assert chunk.start_line == 10
    assert chunk.text.startswith("def main")
    assert chunk.header == "main()"


def test_chunk_count_and_all_ids(store):
    for i in range(3):
        store.upsert_chunk(
            f"c{i}", path=f"f{i}.py", start_line=1, end_line=2,
            text=f"x{i}", header=None,
            embedding=_vec([float(i), 0, 0, 0]),
        )
    assert store.chunk_count() == 3
    assert set(store.all_chunk_ids()) == {"c0", "c1", "c2"}


def test_delete_chunk_cascades_embedding(store):
    store.upsert_chunk(
        "c1", path="x.py", start_line=1, end_line=2, text="x",
        header=None, embedding=_vec([1, 0, 0, 0]),
    )
    store.delete_chunk("c1")
    assert store.get_chunk("c1") is None
    assert store.get_embedding("c1") is None


def test_delete_file_removes_all_chunks_for_path(store):
    store.upsert_chunk(
        "c1", path="x.py", start_line=1, end_line=2, text="a", header=None,
        embedding=_vec([1, 0, 0, 0]),
    )
    store.upsert_chunk(
        "c2", path="x.py", start_line=3, end_line=4, text="b", header=None,
        embedding=_vec([0, 1, 0, 0]),
    )
    store.upsert_chunk(
        "c3", path="y.py", start_line=1, end_line=2, text="c", header=None,
        embedding=_vec([0, 0, 1, 0]),
    )
    n = store.delete_file("x.py")
    assert n == 2
    assert store.chunk_count() == 1
    assert store.get_chunk("c3") is not None


# ---------------------------------------------------------------------------
# Namespace isolation
# ---------------------------------------------------------------------------

def test_namespaces_are_isolated(cfg):
    """Two stores against the same SQLite file but different namespaces
    must not see each other's chunks."""
    s_a = SQLiteRAGStore(cfg, namespace="repo_a")
    s_b = SQLiteRAGStore(cfg, namespace="repo_b")
    s_a.upsert_chunk(
        "c1", path="a.py", start_line=1, end_line=2, text="alpha",
        header=None, embedding=_vec([1, 0, 0, 0]),
    )
    s_b.upsert_chunk(
        "c1", path="b.py", start_line=1, end_line=2, text="beta",
        header=None, embedding=_vec([0, 1, 0, 0]),
    )
    assert s_a.get_chunk("c1").text == "alpha"
    assert s_b.get_chunk("c1").text == "beta"


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------

def test_search_returns_top_k_by_cosine(store):
    # Three chunks orthogonal in the 4-d unit basis.
    store.upsert_chunk("a", path="a", start_line=1, end_line=1, text="a", header=None,
                       embedding=_vec([1, 0, 0, 0]))
    store.upsert_chunk("b", path="b", start_line=1, end_line=1, text="b", header=None,
                       embedding=_vec([0, 1, 0, 0]))
    store.upsert_chunk("c", path="c", start_line=1, end_line=1, text="c", header=None,
                       embedding=_vec([0, 0, 1, 0]))

    # Query close to 'b' — expect b first.
    hits = store.search(_vec([0.1, 1, 0.1, 0]), k=2)
    assert hits[0].id == "b"
    assert len(hits) == 2
    # Score is cosine of L2-normalised vectors → in [-1, 1].
    assert hits[0].score > hits[1].score


def test_search_returns_empty_for_empty_store(store):
    hits = store.search(_vec([1, 0, 0, 0]), k=5)
    assert hits == []


# ---------------------------------------------------------------------------
# Manifest + meta
# ---------------------------------------------------------------------------

def test_manifest_round_trip(store):
    store.save_manifest_hash({"a.py": "sha-a", "b.py": "sha-b"})
    assert store.load_manifest_hash() == {"a.py": "sha-a", "b.py": "sha-b"}


def test_manifest_replace_drops_stale_paths(store):
    store.save_manifest_hash({"a.py": "sha-a", "b.py": "sha-b"})
    # Replace with smaller set — old paths must be gone.
    store.save_manifest_hash({"a.py": "sha-a-v2"})
    assert store.load_manifest_hash() == {"a.py": "sha-a-v2"}


def test_meta_set_and_get(store):
    store.set_meta(dim=768, model="nomic-embed")
    meta = store.get_meta()
    assert meta["dim"] == "768"
    assert meta["model"] == "nomic-embed"
    assert "last_indexed_at" in meta


def test_snapshot_aggregates_state(store):
    store.set_meta(dim=4, model="test")
    store.save_manifest_hash({"a.py": "h1", "b.py": "h2"})
    store.upsert_chunk("c1", path="a.py", start_line=1, end_line=2, text="x",
                       header=None, embedding=_vec([1, 0, 0, 0]))

    snap = store.snapshot()
    assert snap.manifest_size == 2
    assert snap.chunk_count == 1
    assert snap.meta["model"] == "test"


# ---------------------------------------------------------------------------
# Embed cache
# ---------------------------------------------------------------------------

def test_embed_cache_round_trip(store):
    vectors = {
        "h1": np.array([0.1, 0.2], dtype=np.float32),
        "h2": np.array([0.3, 0.4], dtype=np.float32),
    }
    store.mset_embed_cache("modelA", vectors, ttl_seconds=60)
    out = store.mget_embed_cache("modelA", ["h1", "h2", "missing"])
    assert out[0] is not None and out[1] is not None and out[2] is None
    np.testing.assert_array_equal(out[0], vectors["h1"])


def test_embed_cache_ttl_evicts_on_read(store):
    store.mset_embed_cache(
        "modelA",
        {"h1": np.array([1.0, 2.0], dtype=np.float32)},
        ttl_seconds=1,
    )
    # Force expiry via direct UPDATE rather than sleeping.
    store._conn.execute(
        "UPDATE embed_cache SET expires_at = ? WHERE content_hash = ?",
        (time.time() - 60, "h1"),
    )
    store._conn.commit()
    out = store.mget_embed_cache("modelA", ["h1"])
    assert out[0] is None


def test_embed_cache_clear_filters_by_model(store):
    store.mset_embed_cache(
        "modelA", {"h1": np.array([1.0], dtype=np.float32)}, ttl_seconds=60,
    )
    store.mset_embed_cache(
        "modelB", {"h2": np.array([2.0], dtype=np.float32)}, ttl_seconds=60,
    )
    n = store.clear_embed_cache(model="modelA")
    assert n == 1
    assert store.mget_embed_cache("modelB", ["h2"])[0] is not None


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------

def test_clear_namespace_only_drops_own(cfg):
    """clear_namespace must not touch sibling namespaces."""
    s_a = SQLiteRAGStore(cfg, namespace="ns_a")
    s_b = SQLiteRAGStore(cfg, namespace="ns_b")
    s_a.upsert_chunk("c1", path="a.py", start_line=1, end_line=1, text="a",
                     header=None, embedding=np.zeros(4, dtype=np.float32))
    s_b.upsert_chunk("c1", path="b.py", start_line=1, end_line=1, text="b",
                     header=None, embedding=np.zeros(4, dtype=np.float32))
    s_a.clear_namespace()
    assert s_a.chunk_count() == 0
    assert s_b.chunk_count() == 1
