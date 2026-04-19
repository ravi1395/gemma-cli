"""Benchmark ``RAGIndexer.index``.

Three scenarios that together map to the user's day-to-day experience:

1. ``cold`` — empty store, N files to index. Worst-case latency the
   user ever sees; improvements #9 (concurrent batches) target this.
2. ``noop`` — re-run after no disk changes. Today this pays a full
   SHA-1 pass over every file even though nothing will change;
   improvement #2 (lazy SHA-1) targets this and is expected to move
   the needle the most.
3. ``single-edit`` — re-run after editing one file. This is the
   common case during active development; we want the delta to be
   a single embedding batch.

We use ``tmp_path`` to write a small synthetic repo to disk so the
walk + sha1 + mtime pipeline is exercised end-to-end. Embedding is
still stubbed out — Ollama latency is not what we're improving here.
"""

from __future__ import annotations

from pathlib import Path

import fakeredis
import pytest

from gemma.rag.indexer import RAGIndexer
from gemma.rag.store import RedisVectorStore
from tests.bench.conftest import BenchEmbedder


# We keep the indexer corpus small (100 files) so each benchmark
# iteration completes in tens of milliseconds. Chunking cost scales
# linearly with content and is not what these benchmarks measure —
# we care about the surrounding machinery (walk, diff, sha1, upsert).
FILE_COUNT = 100
FILE_BODY = "def f():\n    return 1\n" * 4  # ~96 bytes, one Python chunk


def _make_repo(root: Path, n_files: int = FILE_COUNT) -> None:
    for i in range(n_files):
        (root / f"file_{i:04d}.py").write_text(FILE_BODY)


def _fresh_indexer(root: Path):
    server = fakeredis.FakeServer()
    text_client = fakeredis.FakeRedis(server=server, decode_responses=True)
    bin_client = fakeredis.FakeRedis(server=server, decode_responses=False)
    store = RedisVectorStore(namespace="bench_idx", client=text_client)
    store._binary_client = bin_client
    return RAGIndexer(root=root, store=store, embedder=BenchEmbedder())


def test_index_cold(benchmark, tmp_path):
    """Full first-time index over 100 files. Worst-case latency."""
    _make_repo(tmp_path)
    indexer = _fresh_indexer(tmp_path)
    benchmark(indexer.index)


def test_index_noop(benchmark, tmp_path):
    """Second run, no changes on disk.

    This is the ``lazy SHA-1`` target. Today the run hashes every
    file even though mtime and size haven't changed. After
    improvement #2 lands, expect this benchmark to collapse to a
    near-instant walk + diff.
    """
    _make_repo(tmp_path)
    indexer = _fresh_indexer(tmp_path)
    indexer.index()  # warm the manifest

    def _run():
        indexer.index()

    benchmark(_run)


def test_index_single_edit(benchmark, tmp_path):
    """Edit one file and re-index.

    The common dev-loop path: one embedding batch of 1 file, plus
    99 no-op entries in the manifest.
    """
    _make_repo(tmp_path)
    indexer = _fresh_indexer(tmp_path)
    indexer.index()

    target = tmp_path / "file_0000.py"

    def _run():
        # Mutating a byte changes size, so the file is detected as
        # changed regardless of the lazy-SHA-1 optimisation.
        target.write_text(FILE_BODY + "# edit\n")
        indexer.index()

    benchmark(_run)
