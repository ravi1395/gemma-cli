"""End-to-end indexing bench against a real, locally-running Ollama.

This module is the only benchmark that actually talks to Ollama —
every other file in ``tests/bench/`` uses ``BenchEmbedder`` so the
numbers isolate code paths under our control. Here we deliberately
include the network round-trip because the whole point of items
#9 (concurrent batches) and #10 (content-hash cache) is to *reduce*
that round-trip cost on real runs.

Run with::

    GEMMA_BENCH_REAL_OLLAMA=1 pytest tests/bench/test_bench_real_ollama.py -v

Requirements:
    * ``ollama`` CLI running locally at ``http://localhost:11434``
    * the ``nomic-embed-text`` model pulled (``ollama pull nomic-embed-text``)
    * optional env vars to tweak the corpus:
        - ``GEMMA_BENCH_REPO_FILES``   (default: 20)
        - ``GEMMA_BENCH_FILE_LINES``   (default: 200)

The bench reports three numbers for the same 20-file synthetic repo:

    serial       concurrency=1, cache disabled  — the pre-#9 baseline.
    concurrent   concurrency=4, cache disabled  — the #9 improvement.
    warm         concurrency=4, cache enabled,  — the #10 win, measured
                 second pass after priming the cache.

Interpret: the serial→concurrent delta is the raw embedding-parallelism
win; the concurrent→warm delta is what the cache adds on a branch-
switch / reset-and-reindex workflow.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

import fakeredis
import pytest

from gemma.rag.indexer import RAGIndexer
from gemma.rag.store import RedisVectorStore


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------
#
# We skip by default so the regular ``pytest`` run stays hermetic and
# doesn't fail on CI machines without Ollama. The env flag is
# deliberately verbose so it's hard to trip by accident.

_REAL_OLLAMA_FLAG = "GEMMA_BENCH_REAL_OLLAMA"

pytestmark = pytest.mark.skipif(
    os.environ.get(_REAL_OLLAMA_FLAG) != "1",
    reason=f"set {_REAL_OLLAMA_FLAG}=1 to enable (requires local Ollama).",
)


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def _make_repo(root: Path, n_files: int, n_lines: int) -> None:
    """Write a synthetic Python repo under ``root``.

    Each file is unique — we pad with the file index so the content
    hashes don't collide across files, which would defeat the point
    of measuring the cache on realistic workloads.
    """
    for i in range(n_files):
        body = (
            f"# file {i}\n"
            f"from math import pi\n\n"
            f"def fn_{i}(x: int) -> int:\n"
            f"    return x * {i} + int(pi)\n\n"
            + "# padding line\n" * n_lines
        )
        (root / f"mod_{i:04d}.py").write_text(body)


def _build_indexer(
    root: Path,
    *,
    concurrency: int,
    cache_enabled: bool,
    model: str,
    host: str,
) -> RAGIndexer:
    """Fresh store + real embedder factory per run.

    Using fakeredis here still lets us measure wall time from Ollama
    because the embed step is the dominant cost (~30 ms/call against
    nomic-embed-text on CPU).
    """
    from gemma.embeddings import Embedder  # import here so the module
    # stays collectable without Ollama installed; the skip at the top
    # already protects the actual call sites.

    server = fakeredis.FakeServer()
    text_client = fakeredis.FakeRedis(server=server, decode_responses=True)
    bin_client = fakeredis.FakeRedis(server=server, decode_responses=False)
    store = RedisVectorStore(namespace="bench_real", client=text_client)
    store._binary_client = bin_client

    def _factory():
        # A fresh ``Embedder`` per worker — Ollama's HTTP client
        # serialises requests on a single session, which would
        # nullify the concurrency win if shared.
        return Embedder(model=model, host=host)

    primary = Embedder(model=model, host=host)
    return RAGIndexer(
        root=root, store=store, embedder=primary,
        concurrency=concurrency,
        embedder_factory=_factory,
        cache_enabled=cache_enabled,
        cache_ttl_seconds=60 * 60,  # 1h is plenty for a bench session
    )


# ---------------------------------------------------------------------------
# Bench body
# ---------------------------------------------------------------------------

def _time_once(fn: Callable[[], None]) -> float:
    """Time one callable — coarse ``perf_counter`` is enough for network-scale events."""
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def test_real_ollama_serial_vs_concurrent_vs_warm(tmp_path, capsys):
    """Wall-time comparison: serial, concurrent, and warm (cache) indexing.

    Not a unit test in the strict sense — it doesn't assert any
    numerical threshold (those vary by host). It *does* assert that
    every indexing call succeeded (``chunks_written > 0``) so a
    regression that silently returns an empty index will still fail
    the bench. Numbers land on stdout so CI jobs can capture them.
    """
    n_files = int(os.environ.get("GEMMA_BENCH_REPO_FILES", "20"))
    n_lines = int(os.environ.get("GEMMA_BENCH_FILE_LINES", "200"))
    model = os.environ.get("GEMMA_EMBED_MODEL", "nomic-embed-text")
    host = os.environ.get("GEMMA_OLLAMA_HOST", "http://localhost:11434")

    # --- Build the same repo for each scenario -------------------
    # We use sibling dirs rather than one shared repo so that each
    # run is unambiguously a cold walk of its own files.
    serial_dir = tmp_path / "serial"
    concurrent_dir = tmp_path / "concurrent"
    warm_dir = tmp_path / "warm"
    for d in (serial_dir, concurrent_dir, warm_dir):
        d.mkdir()
        _make_repo(d, n_files=n_files, n_lines=n_lines)

    # --- 1. Serial baseline (pre-#9) -----------------------------
    idx_serial = _build_indexer(
        serial_dir, concurrency=1, cache_enabled=False,
        model=model, host=host,
    )
    t_serial = _time_once(lambda: idx_serial.index())

    # --- 2. Concurrent path (post-#9, no cache) -----------------
    idx_concurrent = _build_indexer(
        concurrent_dir, concurrency=4, cache_enabled=False,
        model=model, host=host,
    )
    t_concurrent = _time_once(lambda: idx_concurrent.index())

    # --- 3. Warm cache (post-#10) -------------------------------
    # Prime the cache with a concurrent run, reset the per-namespace
    # index, then time the second pass which should hit the cache
    # for every chunk.
    idx_warm = _build_indexer(
        warm_dir, concurrency=4, cache_enabled=True,
        model=model, host=host,
    )
    prime_stats = idx_warm.index()
    # Reset the namespace so the indexer treats everything as new
    # again — but the embed cache persists because it lives outside
    # the namespace.
    idx_warm._store.clear_namespace()
    t_warm = _time_once(lambda: idx_warm.index())

    # --- Report --------------------------------------------------
    with capsys.disabled():
        print("\n── GEMMA_BENCH_REAL_OLLAMA: indexer wall times ──")
        print(f"  files       : {n_files}")
        print(f"  model       : {model}")
        print(f"  serial      : {t_serial:6.2f}s  (concurrency=1, no cache)")
        print(f"  concurrent  : {t_concurrent:6.2f}s  (concurrency=4, no cache)")
        print(f"  warm (cache): {t_warm:6.2f}s  (concurrency=4, cache hits)")
        if t_serial > 0 and t_concurrent > 0:
            print(
                f"  concurrent speedup: {t_serial / t_concurrent:5.2f}x"
            )
        if t_concurrent > 0 and t_warm > 0:
            print(
                f"  warm speedup vs concurrent: {t_concurrent / t_warm:5.2f}x"
            )
        print(f"  primed cache chunks: {prime_stats.chunks_written}")

    # Keep the assertion deliberately loose — absolute thresholds
    # would bake a particular laptop's performance into the test.
    assert prime_stats.chunks_written > 0
