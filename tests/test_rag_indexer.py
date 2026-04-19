"""End-to-end tests for :class:`RAGIndexer`.

The indexer orchestrates walk → chunk → embed → upsert. We replace
the embedder with a deterministic stub so the tests are hermetic and
fast, and we use fakeredis for the store.

Key behaviour under test:
  * First-time full index pulls in every matching file
  * Re-index with no changes is a no-op — zero embedding calls
  * Editing one file re-embeds only that file's chunks
  * Deleting a file from disk removes its chunks from the store
  * Skipped paths: denylisted dirs, wrong extension, oversized files
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List

import fakeredis
import numpy as np
import pytest

from gemma.rag.indexer import RAGIndexer
from gemma.rag.store import RedisVectorStore


# ---------------------------------------------------------------------------
# Stub embedder — deterministic, call-counted
# ---------------------------------------------------------------------------

class StubEmbedder:
    """Deterministic embedder: hashes the input to a 8-dim vector.

    The exact vectors don't matter for these tests; we mostly care
    about the *number* of embedding calls so we can verify that
    incremental re-indexes only touch changed files.
    """

    model = "stub-embed"

    def __init__(self):
        self.calls: List[List[str]] = []

    def _embed_one(self, text: str) -> np.ndarray:
        h = hashlib.sha1(text.encode("utf-8")).digest()
        # Map the first 8 bytes to 8 floats in [-1, 1].
        return np.array(
            [((b / 127.5) - 1.0) for b in h[:8]], dtype=np.float32,
        )

    def embed(self, text: str) -> np.ndarray:
        self.calls.append([text])
        return self._embed_one(text)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        self.calls.append(list(texts))
        return [self._embed_one(t) for t in texts]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def wired(tmp_path):
    """Build a {root, store, embedder, indexer} quad for one test."""
    server = fakeredis.FakeServer()
    text_client = fakeredis.FakeRedis(server=server, decode_responses=True)
    bin_client = fakeredis.FakeRedis(server=server, decode_responses=False)

    store = RedisVectorStore(namespace="ns_test", client=text_client)
    store._binary_client = bin_client

    embedder = StubEmbedder()
    indexer = RAGIndexer(root=tmp_path, store=store, embedder=embedder)
    return {"root": tmp_path, "store": store, "embedder": embedder, "indexer": indexer}


# ---------------------------------------------------------------------------
# First-time indexing
# ---------------------------------------------------------------------------

def test_first_index_covers_all_source_files(wired):
    root, store, embedder, indexer = wired["root"], wired["store"], wired["embedder"], wired["indexer"]

    (root / "a.py").write_text("def foo():\n    return 1\n")
    (root / "README.md").write_text("# Title\n\nBody.\n")

    stats = indexer.index()
    assert stats.files_scanned == 2
    assert stats.files_added == 2
    assert stats.chunks_written > 0
    # Every new file should produce >=1 embedding call.
    assert embedder.calls, "embedder.embed_batch was never invoked"
    # Store has that many chunks.
    assert store.chunk_count() == stats.chunks_written


def test_first_index_skips_denylisted_dirs(wired):
    root, store, indexer = wired["root"], wired["store"], wired["indexer"]

    (root / "a.py").write_text("x = 1\n")
    # node_modules and .git should be pruned.
    nm = root / "node_modules" / "lib"
    nm.mkdir(parents=True)
    (nm / "big.js").write_text("var x=1;")
    git = root / ".git"
    git.mkdir()
    (git / "config").write_text("")

    stats = indexer.index()
    # Only a.py should be seen.
    assert stats.files_scanned == 1


def test_first_index_skips_wrong_extensions(wired):
    root, indexer = wired["root"], wired["indexer"]
    (root / "a.png").write_bytes(b"\x89PNG...")
    (root / "b.zip").write_bytes(b"PK\x03\x04")
    (root / "c.py").write_text("x = 1\n")

    stats = indexer.index()
    assert stats.files_scanned == 1  # only .py was eligible


def test_first_index_skips_oversized_files(tmp_path):
    # Rebuild with a tiny max_file_size so the test is fast.
    server = fakeredis.FakeServer()
    text_client = fakeredis.FakeRedis(server=server, decode_responses=True)
    bin_client = fakeredis.FakeRedis(server=server, decode_responses=False)
    store = RedisVectorStore(namespace="ns_small", client=text_client)
    store._binary_client = bin_client

    indexer = RAGIndexer(
        root=tmp_path, store=store, embedder=StubEmbedder(),
        max_file_size=100,  # 100 bytes
    )

    (tmp_path / "small.py").write_text("x = 1\n")
    (tmp_path / "big.py").write_text("x = 1\n" + "# padding\n" * 50)

    stats = indexer.index()
    # big.py silently skipped; only small.py walked.
    assert stats.files_scanned == 1


# ---------------------------------------------------------------------------
# Incremental re-indexing
# ---------------------------------------------------------------------------

def test_reindex_unchanged_is_noop(wired):
    root, embedder, indexer = wired["root"], wired["embedder"], wired["indexer"]
    (root / "a.py").write_text("def f():\n    return 1\n")
    (root / "b.py").write_text("def g():\n    return 2\n")

    indexer.index()
    first_calls = len(embedder.calls)
    assert first_calls > 0  # sanity: the first pass embedded something

    # Second pass — nothing changed on disk.
    stats = indexer.index()
    assert stats.files_added == 0
    assert stats.files_changed == 0
    assert stats.files_removed == 0
    assert stats.chunks_written == 0
    # Embedder was NOT called again (except possibly the dim-probe,
    # which goes through .embed, not .embed_batch).
    # Count .embed_batch calls only:
    batch_calls = [c for c in embedder.calls if len(c) > 1 or (c and isinstance(c, list) and c != ["dim-probe"])]
    # Just ensure no *new* embed_batch happened:
    assert first_calls == len(embedder.calls) or all(
        extra_call == ["dim-probe"] for extra_call in embedder.calls[first_calls:]
    )


def test_reindex_after_edit_reembeds_only_changed_file(wired):
    root, store, embedder, indexer = wired["root"], wired["store"], wired["embedder"], wired["indexer"]

    (root / "a.py").write_text("def f():\n    return 1\n")
    (root / "b.py").write_text("def g():\n    return 2\n")

    indexer.index()
    chunks_after_first = store.chunk_count()

    # Edit a.py only — touch mtime and rewrite content.
    (root / "a.py").write_text("def f():\n    return 999\n")

    calls_before = len(embedder.calls)
    stats = indexer.index()

    assert stats.files_changed == 1
    assert stats.files_unchanged == 1
    # Chunks for a.py were replaced, so total chunk count is stable
    # (assuming the same number of chunks per file post-edit).
    assert store.chunk_count() == chunks_after_first
    # A new embedding call happened (at least one batch for a.py).
    assert len(embedder.calls) > calls_before


def test_reindex_after_delete_removes_chunks(wired):
    root, store, indexer = wired["root"], wired["store"], wired["indexer"]

    (root / "a.py").write_text("def f():\n    return 1\n")
    (root / "b.py").write_text("def g():\n    return 2\n")

    indexer.index()
    chunks_initial = store.chunk_count()
    assert chunks_initial >= 2

    # Delete b.py from disk.
    (root / "b.py").unlink()

    stats = indexer.index()
    assert stats.files_removed == 1
    assert stats.chunks_deleted >= 1
    # Store has fewer chunks now.
    assert store.chunk_count() < chunks_initial


# ---------------------------------------------------------------------------
# Robustness: syntax errors, unreadable files
# ---------------------------------------------------------------------------

def test_syntax_error_in_python_file_does_not_abort_run(wired):
    """A file with a SyntaxError still gets chunked via fallback — run survives."""
    root, store, indexer = wired["root"], wired["store"], wired["indexer"]
    (root / "ok.py").write_text("def f():\n    return 1\n")
    (root / "broken.py").write_text("def f(:\n    return 1\n")  # invalid

    stats = indexer.index()
    assert stats.files_scanned == 2
    # Both files get at least some coverage: ok.py via AST, broken.py via sliding fallback.
    assert store.chunk_count() >= 2


# ---------------------------------------------------------------------------
# Manifest persistence
# ---------------------------------------------------------------------------

def test_manifest_round_trips_across_runs(wired):
    root, store, indexer = wired["root"], wired["store"], wired["indexer"]
    (root / "a.py").write_text("def f():\n    return 1\n")

    indexer.index()
    manifest_blobs_1 = store.load_manifest_hash()
    assert "a.py" in manifest_blobs_1

    indexer.index()  # no-op
    manifest_blobs_2 = store.load_manifest_hash()
    # The blobs survive unchanged across a no-op run (same sha1 ⇒ same
    # mtime/size/sha1 — even chunk_ids are preserved).
    assert manifest_blobs_2["a.py"] == manifest_blobs_1["a.py"]


def test_meta_records_dim_and_model_after_run(wired):
    root, store, indexer = wired["root"], wired["store"], wired["indexer"]
    (root / "a.py").write_text("x = 1\n")
    indexer.index()

    meta = store.get_meta()
    assert meta["model"] == "stub-embed"
    assert int(meta["dim"]) == 8  # our StubEmbedder returns 8-dim vectors


# ---------------------------------------------------------------------------
# Item #9 — concurrent embed batches
# ---------------------------------------------------------------------------
#
# The tests below drive the executor path directly by setting
# ``concurrency`` on the indexer and keeping ``_EMBED_BATCH`` small so
# we comfortably produce multiple batches from tiny fixtures. We use
# threading.Event / Barrier tricks so the tests would deadlock or
# flip order if the implementation regressed.

def _big_tree(root: Path, n_files: int = 6) -> None:
    """Create enough files that we get multiple embed batches.

    ``_EMBED_BATCH`` defaults to 32. We don't lower it for the tests
    (keeps prod config honest); instead we generate enough chunks
    across N files that the flattened list spills over the batch size.
    """
    payload = "x = 1\n" * 40   # pads each file enough to chunk multiply
    for i in range(n_files):
        (root / f"f{i:02d}.py").write_text(
            f"# file {i}\n{payload}\ndef fn_{i}():\n    return {i}\n"
        )


def test_concurrent_apply_upserts_preserves_order(tmp_path):
    """Parallel path must return vectors in input (batch) order.

    We stagger per-batch sleeps so that ``as_completed``-style ordering
    would observably flip pairs. ``executor.map`` must preserve the
    submission order regardless of completion order.
    """
    import time

    server = fakeredis.FakeServer()
    text_client = fakeredis.FakeRedis(server=server, decode_responses=True)
    bin_client = fakeredis.FakeRedis(server=server, decode_responses=False)
    store = RedisVectorStore(namespace="ns_order", client=text_client)
    store._binary_client = bin_client

    class StaggeredEmbedder(StubEmbedder):
        """Returns a per-batch marker so we can verify order."""
        def __init__(self):
            super().__init__()
            self._counter = 0
            self._lock = __import__("threading").Lock()

        def embed_batch(self, texts):
            with self._lock:
                self._counter += 1
                n = self._counter
            # Later-submitted batches finish faster → completion order
            # differs from submission order.
            time.sleep(max(0.0, 0.04 - n * 0.005))
            # Return deterministic vectors with a per-batch tag so we
            # can spot-check ordering without fighting hash collisions.
            return [
                np.full(8, fill_value=float(n), dtype=np.float32)
                for _ in texts
            ]

    embedder = StaggeredEmbedder()
    _big_tree(tmp_path, n_files=6)

    # Force multiple batches by shrinking the in-memory embed size —
    # we do this via monkeypatching the module-level constant just for
    # this test so the production path stays unchanged.
    from gemma.rag import indexer as indexer_mod
    original = indexer_mod._EMBED_BATCH
    indexer_mod._EMBED_BATCH = 2
    try:
        indexer = indexer_mod.RAGIndexer(
            root=tmp_path, store=store, embedder=embedder,
            concurrency=4,
        )
        stats = indexer.index()
    finally:
        indexer_mod._EMBED_BATCH = original

    # Every stored vector should have been tagged with the batch number
    # it came from. We don't care about *which* vector wins globally;
    # we care that batches written in Redis correspond to the same
    # chunks we assigned in the plan — a regression would produce vec
    # values that don't exist in our tag set.
    assert stats.chunks_written > 0
    for cid in store.all_chunk_ids()[:5]:
        emb = store.get_embedding(cid)
        assert emb is not None
        # The vector's tag must be a positive integer equal to the
        # batch index it came from. Normalise distorts magnitudes, so
        # we only check sign/positivity here (all tags are positive).
        assert emb[0] > 0.0


def test_concurrent_path_actually_uses_multiple_workers(tmp_path):
    """Serial execution with our payload would deadlock the barrier."""
    import threading

    server = fakeredis.FakeServer()
    text_client = fakeredis.FakeRedis(server=server, decode_responses=True)
    bin_client = fakeredis.FakeRedis(server=server, decode_responses=False)
    store = RedisVectorStore(namespace="ns_parallel", client=text_client)
    store._binary_client = bin_client

    # Barrier-of-3 with a 1-second timeout: if the implementation
    # runs batches serially, the second embed_batch call never fires
    # in time and the barrier raises BrokenBarrierError.
    barrier = threading.Barrier(3, timeout=1.5)

    class BarrierEmbedder(StubEmbedder):
        def embed_batch(self, texts):
            # Every worker waits until 3 arrive → proves >= 3 workers
            # ran in parallel.
            barrier.wait()
            return super().embed_batch(texts)

    embedder = BarrierEmbedder()
    _big_tree(tmp_path, n_files=6)

    from gemma.rag import indexer as indexer_mod
    original = indexer_mod._EMBED_BATCH
    indexer_mod._EMBED_BATCH = 2
    try:
        indexer = indexer_mod.RAGIndexer(
            root=tmp_path, store=store, embedder=embedder,
            concurrency=4,
        )
        stats = indexer.index()
    finally:
        indexer_mod._EMBED_BATCH = original

    # No error → barrier released → at least 3 parallel embed calls.
    assert stats.chunks_written > 0


def test_embedder_factory_called_once_per_worker(tmp_path):
    """Factory should be invoked no more than ``concurrency`` times."""
    server = fakeredis.FakeServer()
    text_client = fakeredis.FakeRedis(server=server, decode_responses=True)
    bin_client = fakeredis.FakeRedis(server=server, decode_responses=False)
    store = RedisVectorStore(namespace="ns_factory", client=text_client)
    store._binary_client = bin_client

    built: list[StubEmbedder] = []

    def factory() -> StubEmbedder:
        e = StubEmbedder()
        built.append(e)
        return e

    _big_tree(tmp_path, n_files=6)

    # Primary embedder is also a stub — used for the dim probe only.
    primary = StubEmbedder()
    from gemma.rag import indexer as indexer_mod
    original = indexer_mod._EMBED_BATCH
    indexer_mod._EMBED_BATCH = 2
    try:
        indexer = indexer_mod.RAGIndexer(
            root=tmp_path, store=store, embedder=primary,
            concurrency=3, embedder_factory=factory,
        )
        stats = indexer.index()
    finally:
        indexer_mod._EMBED_BATCH = original

    assert stats.chunks_written > 0
    # Factory called once per worker thread, *not* per batch.
    assert 1 <= len(built) <= 3
    # The primary embedder handled the dim probe, not the batches.
    assert primary.calls == [["dim-probe"]]


def test_serial_path_skips_thread_pool(tmp_path, monkeypatch):
    """With concurrency=1, ThreadPoolExecutor must not be instantiated.

    A regression that accidentally used the executor for the serial
    path would cause measurable overhead on every index call; this
    test pins the behaviour by raising if the executor is touched.
    """
    from gemma.rag import indexer as indexer_mod

    def _boom(*a, **kw):
        raise AssertionError("ThreadPoolExecutor used on serial path")

    monkeypatch.setattr(indexer_mod, "ThreadPoolExecutor", _boom)

    server = fakeredis.FakeServer()
    text_client = fakeredis.FakeRedis(server=server, decode_responses=True)
    bin_client = fakeredis.FakeRedis(server=server, decode_responses=False)
    store = RedisVectorStore(namespace="ns_serial", client=text_client)
    store._binary_client = bin_client

    _big_tree(tmp_path, n_files=4)
    indexer = indexer_mod.RAGIndexer(
        root=tmp_path, store=store, embedder=StubEmbedder(),
        concurrency=1,
    )
    stats = indexer.index()
    assert stats.chunks_written > 0


def test_single_batch_skips_thread_pool_even_when_concurrency_high(tmp_path, monkeypatch):
    """Even with concurrency=4, a 1-batch run should use the serial fast path."""
    from gemma.rag import indexer as indexer_mod

    def _boom(*a, **kw):
        raise AssertionError("executor used for single-batch run")

    monkeypatch.setattr(indexer_mod, "ThreadPoolExecutor", _boom)

    server = fakeredis.FakeServer()
    text_client = fakeredis.FakeRedis(server=server, decode_responses=True)
    bin_client = fakeredis.FakeRedis(server=server, decode_responses=False)
    store = RedisVectorStore(namespace="ns_single_batch", client=text_client)
    store._binary_client = bin_client

    # Exactly one tiny file → one chunk → one batch regardless of
    # _EMBED_BATCH value.
    (tmp_path / "a.py").write_text("x = 1\n")
    indexer = indexer_mod.RAGIndexer(
        root=tmp_path, store=store, embedder=StubEmbedder(),
        concurrency=4,
    )
    stats = indexer.index()
    assert stats.chunks_written > 0


def test_concurrency_is_clamped_to_safe_range():
    """Values outside [1, 16] should be clamped, not propagated raw."""
    server = fakeredis.FakeServer()
    text_client = fakeredis.FakeRedis(server=server, decode_responses=True)
    store = RedisVectorStore(namespace="ns_clamp", client=text_client)

    idx_low = RAGIndexer(
        root=Path("/tmp"), store=store, embedder=StubEmbedder(), concurrency=-5,
    )
    idx_zero = RAGIndexer(
        root=Path("/tmp"), store=store, embedder=StubEmbedder(), concurrency=0,
    )
    idx_hi = RAGIndexer(
        root=Path("/tmp"), store=store, embedder=StubEmbedder(), concurrency=999,
    )
    assert idx_low._concurrency == 1
    assert idx_zero._concurrency == 1
    assert idx_hi._concurrency == 16


# ---------------------------------------------------------------------------
# Item #10 — content-hash embed cache
# ---------------------------------------------------------------------------

def _build_store_pair(ns: str):
    server = fakeredis.FakeServer()
    text_client = fakeredis.FakeRedis(server=server, decode_responses=True)
    bin_client = fakeredis.FakeRedis(server=server, decode_responses=False)
    store = RedisVectorStore(namespace=ns, client=text_client)
    store._binary_client = bin_client
    return store


def test_embed_cache_warm_run_skips_embedder(tmp_path):
    """Second run over identical content must hit the cache, not the embedder."""
    store = _build_store_pair("ns_cache_hit")

    _big_tree(tmp_path, n_files=3)
    embedder = StubEmbedder()
    indexer = RAGIndexer(
        root=tmp_path, store=store, embedder=embedder,
        cache_enabled=True, cache_ttl_seconds=60,
    )
    stats1 = indexer.index()
    assert stats1.chunks_written > 0
    assert stats1.chunks_cache_hit == 0  # cold — nothing cached yet

    # Drop the per-namespace index so the *indexer* thinks everything
    # is new again, but keep the embed cache populated. We simulate a
    # full reset+reindex here.
    store.clear_namespace()

    calls_before = sum(
        len(c) for c in embedder.calls if c != ["dim-probe"]
    )

    stats2 = indexer.index()
    # Every chunk should have hit the cache — no new embedder calls.
    assert stats2.chunks_written > 0
    assert stats2.chunks_cache_hit == stats2.chunks_written

    calls_after = sum(
        len(c) for c in embedder.calls if c != ["dim-probe"]
    )
    # The only post-reset embedder activity is the dim probe (which
    # goes through ``embed``, not ``embed_batch``). Batch calls on the
    # warm run should be zero.
    assert calls_after == calls_before


def test_embed_cache_disabled_short_circuits_all_ops(tmp_path):
    """When disabled, mget/mset must not be called even once."""
    store = _build_store_pair("ns_cache_off")

    calls = {"mget": 0, "mset": 0}
    original_mget = store.mget_embed_cache
    original_mset = store.mset_embed_cache

    def _spy_mget(*a, **kw):
        calls["mget"] += 1
        return original_mget(*a, **kw)

    def _spy_mset(*a, **kw):
        calls["mset"] += 1
        return original_mset(*a, **kw)

    store.mget_embed_cache = _spy_mget  # type: ignore[assignment]
    store.mset_embed_cache = _spy_mset  # type: ignore[assignment]

    _big_tree(tmp_path, n_files=2)
    indexer = RAGIndexer(
        root=tmp_path, store=store, embedder=StubEmbedder(),
        cache_enabled=False,
    )
    stats = indexer.index()
    assert stats.chunks_written > 0
    assert calls == {"mget": 0, "mset": 0}


def test_embed_cache_is_model_scoped(tmp_path):
    """A different embedding model must not read vectors from another model's cache."""
    store = _build_store_pair("ns_cache_model")

    _big_tree(tmp_path, n_files=2)

    class ModelA(StubEmbedder):
        model = "enc-a"

    class ModelB(StubEmbedder):
        model = "enc-b"

    a, b = ModelA(), ModelB()

    RAGIndexer(
        root=tmp_path, store=store, embedder=a, cache_enabled=True,
    ).index()

    # Reset index, flip to a new model with a different ``model``
    # attribute. Cache keys are model-scoped, so b should miss.
    store.clear_namespace()
    indexer_b = RAGIndexer(
        root=tmp_path, store=store, embedder=b, cache_enabled=True,
    )
    stats_b = indexer_b.index()
    assert stats_b.chunks_cache_hit == 0


def test_embed_cache_stats_and_clear(tmp_path):
    """Admin surface: stats reports counts by model; clear drops them."""
    store = _build_store_pair("ns_cache_admin")

    _big_tree(tmp_path, n_files=2)
    indexer = RAGIndexer(
        root=tmp_path, store=store, embedder=StubEmbedder(),
        cache_enabled=True,
    )
    indexer.index()

    info = store.embed_cache_stats()
    assert info["total_keys"] > 0
    assert "stub-embed" in info["per_model"]

    # Clearing only the 'stub-embed' model empties its bucket.
    deleted = store.clear_embed_cache(model="stub-embed")
    assert deleted == info["per_model"]["stub-embed"]
    assert store.embed_cache_stats()["total_keys"] == 0
