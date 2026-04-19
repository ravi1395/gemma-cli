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
