"""Tests for :class:`RedisVectorStore` against fakeredis.

We use fakeredis twice: once with ``decode_responses=True`` for the
main client and once with ``decode_responses=False`` for the binary
client (embeddings). The store under test knows how to build the
binary flavour from the same fakeredis instance (see
:meth:`RedisVectorStore._binary_conn`).
"""

from __future__ import annotations

import fakeredis
import numpy as np
import pytest

from gemma.rag.store import RedisVectorStore, StoredChunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store():
    """A fresh store with a unique namespace per test.

    Two fakeredis clients — one decoding text, one returning bytes —
    share a single :class:`fakeredis.FakeServer` so the embedding
    binary blobs written by the binary client are visible to the
    text client's chunk metadata reads.
    """
    server = fakeredis.FakeServer()
    text_client = fakeredis.FakeRedis(server=server, decode_responses=True)
    bin_client = fakeredis.FakeRedis(server=server, decode_responses=False)

    s = RedisVectorStore(namespace="ns_test", client=text_client)
    # Pre-populate the binary client so the store skips its
    # connection-pool auto-detection path.
    s._binary_client = bin_client
    yield s
    text_client.flushall()


def _vec(*vals) -> np.ndarray:
    return np.array(vals, dtype=np.float32)


# ---------------------------------------------------------------------------
# Upsert + retrieval
# ---------------------------------------------------------------------------

def test_upsert_then_get_chunk(store):
    store.upsert_chunk(
        chunk_id="c1", path="a.py",
        start_line=1, end_line=10,
        text="print('hi')", header="<module-head>",
        embedding=_vec(1, 0, 0),
    )
    chunk = store.get_chunk("c1")
    assert chunk is not None
    assert chunk.id == "c1"
    assert chunk.path == "a.py"
    assert chunk.start_line == 1
    assert chunk.end_line == 10
    assert chunk.text == "print('hi')"
    assert chunk.header == "<module-head>"


def test_upsert_then_get_embedding(store):
    store.upsert_chunk(
        chunk_id="c1", path="a.py",
        start_line=1, end_line=10, text="x", header=None,
        embedding=_vec(0.6, 0.8),  # already unit-norm
    )
    emb = store.get_embedding("c1")
    assert emb is not None
    np.testing.assert_allclose(emb, [0.6, 0.8], rtol=1e-6)


def test_chunk_count_and_index_membership(store):
    store.upsert_chunk(
        chunk_id="c1", path="a.py", start_line=1, end_line=1,
        text="", header=None, embedding=_vec(1, 0),
    )
    store.upsert_chunk(
        chunk_id="c2", path="a.py", start_line=2, end_line=2,
        text="", header=None, embedding=_vec(0, 1),
    )
    assert store.chunk_count() == 2
    assert set(store.all_chunk_ids()) == {"c1", "c2"}


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

def test_delete_chunk_removes_everywhere(store):
    store.upsert_chunk(
        chunk_id="c1", path="a.py", start_line=1, end_line=1,
        text="t", header=None, embedding=_vec(1, 0),
    )
    store.delete_chunk("c1")
    assert store.get_chunk("c1") is None
    assert store.get_embedding("c1") is None
    assert "c1" not in store.all_chunk_ids()


def test_delete_file_removes_all_chunks_for_path(store):
    for cid in ("a1", "a2", "a3"):
        store.upsert_chunk(
            chunk_id=cid, path="a.py", start_line=1, end_line=1,
            text="", header=None, embedding=_vec(1, 0),
        )
    store.upsert_chunk(
        chunk_id="b1", path="b.py", start_line=1, end_line=1,
        text="", header=None, embedding=_vec(0, 1),
    )
    removed = store.delete_file("a.py")
    assert removed == 3
    # Only b1 remains.
    assert store.all_chunk_ids() == ["b1"]


def test_delete_nonexistent_chunk_is_noop(store):
    # Must not raise.
    store.delete_chunk("ghost")


# ---------------------------------------------------------------------------
# Manifest roundtrip
# ---------------------------------------------------------------------------

def test_save_and_load_manifest_hash(store):
    blobs = {
        "a.py": '{"path":"a.py","mtime_ns":1,"size":10,"sha1":"x","chunk_ids":[]}',
        "b.md": '{"path":"b.md","mtime_ns":2,"size":20,"sha1":"y","chunk_ids":["c1"]}',
    }
    store.save_manifest_hash(blobs)
    assert store.load_manifest_hash() == blobs


def test_save_manifest_replaces_whole_hash(store):
    """A second save must not leave orphan entries from the first."""
    store.save_manifest_hash({"a.py": "{}"})
    store.save_manifest_hash({"b.py": "{}"})
    loaded = store.load_manifest_hash()
    assert "a.py" not in loaded
    assert "b.py" in loaded


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------

def test_set_and_get_meta(store):
    store.set_meta(dim=768, model="nomic-embed-text")
    meta = store.get_meta()
    assert meta["dim"] == "768"
    assert meta["model"] == "nomic-embed-text"
    # last_indexed_at is a stringified unix ts.
    assert meta["last_indexed_at"].isdigit()


# ---------------------------------------------------------------------------
# Search (cosine top-k)
# ---------------------------------------------------------------------------

def test_search_returns_closest_first(store):
    # Three unit vectors in 2D — each one "closest" to a specific query.
    store.upsert_chunk(
        chunk_id="east", path="a", start_line=1, end_line=1,
        text="east", header=None, embedding=_vec(1, 0),
    )
    store.upsert_chunk(
        chunk_id="north", path="a", start_line=2, end_line=2,
        text="north", header=None, embedding=_vec(0, 1),
    )
    store.upsert_chunk(
        chunk_id="ne", path="a", start_line=3, end_line=3,
        text="ne", header=None, embedding=_vec(0.707, 0.707),
    )

    # Query pointing east → east first, then NE, then north.
    hits = store.search(_vec(1, 0), k=3)
    assert [h.id for h in hits] == ["east", "ne", "north"]
    # Scores must be cosine (= dot after both are normalised).
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_search_k_clamps_to_available(store):
    store.upsert_chunk(
        chunk_id="only", path="a", start_line=1, end_line=1,
        text="", header=None, embedding=_vec(1, 0),
    )
    hits = store.search(_vec(1, 0), k=10)
    assert len(hits) == 1


def test_search_empty_store_returns_nothing(store):
    hits = store.search(_vec(1, 0), k=5)
    assert hits == []


def test_search_zero_query_returns_empty(store):
    store.upsert_chunk(
        chunk_id="c", path="a", start_line=1, end_line=1,
        text="", header=None, embedding=_vec(1, 0),
    )
    assert store.search(np.zeros(2, dtype=np.float32), k=5) == []


# ---------------------------------------------------------------------------
# Namespace isolation
# ---------------------------------------------------------------------------

def test_namespace_isolation():
    """Two stores on different namespaces must not see each other's chunks."""
    server = fakeredis.FakeServer()
    text_client = fakeredis.FakeRedis(server=server, decode_responses=True)
    bin_client = fakeredis.FakeRedis(server=server, decode_responses=False)

    s1 = RedisVectorStore(namespace="ns_a", client=text_client)
    s1._binary_client = bin_client
    s2 = RedisVectorStore(namespace="ns_b", client=text_client)
    s2._binary_client = bin_client

    s1.upsert_chunk(
        chunk_id="c1", path="a", start_line=1, end_line=1,
        text="hello from a", header=None, embedding=_vec(1, 0),
    )

    # ns_b sees nothing.
    assert s2.chunk_count() == 0
    assert s2.get_chunk("c1") is None
    # ns_a keeps its own.
    assert s1.chunk_count() == 1


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------

def test_clear_namespace_removes_everything(store):
    store.upsert_chunk(
        chunk_id="c1", path="a.py", start_line=1, end_line=1,
        text="", header=None, embedding=_vec(1, 0),
    )
    store.set_meta(dim=2, model="stub")
    store.save_manifest_hash({"a.py": "{}"})

    deleted = store.clear_namespace()
    assert deleted > 0
    assert store.chunk_count() == 0
    assert store.get_meta() == {}
    assert store.load_manifest_hash() == {}
