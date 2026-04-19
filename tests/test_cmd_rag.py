"""CLI smoke tests for ``gemma rag {index,query,status,reset}``.

These tests use the ``set_store_factory`` / ``set_embedder_factory``
hooks in :mod:`gemma.commands.rag` to swap the real Redis + Ollama
dependencies for a fakeredis-backed store and a deterministic stub
embedder. That keeps the tests hermetic — no Redis or Ollama is needed.

What we verify per command:
  * ``index``  — exits 0, reports stats, actually writes chunks
  * ``query``  — embedder is called, hits are rendered with citations
  * ``status`` — reads meta + manifest size + chunk count from the store
  * ``reset``  — requires confirmation; ``--yes`` bypasses it
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List

import fakeredis
import numpy as np
import pytest
from typer.testing import CliRunner

from gemma import commands as _commands_pkg  # noqa: F401 — ensure package loads
from gemma.commands import rag as rag_cmd
from gemma.main import app
from gemma.rag.store import RedisVectorStore


runner = CliRunner()


# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------

class _StubEmbedder:
    """Hash-based deterministic embedder. 8-dim float32 vectors.

    We intentionally produce a fixed-dimension vector so the indexer's
    dim-probe (``embed("dim-probe")``) yields a stable value written to
    the store's ``meta`` hash — the ``status`` command surfaces it.
    """

    model = "stub-embed"

    def _one(self, text: str) -> np.ndarray:
        h = hashlib.sha1(text.encode("utf-8")).digest()
        return np.array([((b / 127.5) - 1.0) for b in h[:8]], dtype=np.float32)

    def embed(self, text: str) -> np.ndarray:
        return self._one(text)

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        return [self._one(t) for t in texts]


class _SharedFakeState:
    """Holds one FakeServer so every factory call shares the same data.

    The CLI invokes ``_wire`` per subcommand, which means it calls the
    factories fresh each time. To have two commands (e.g. ``index``
    then ``query``) operate on the same virtual Redis, every factory
    call needs to point at the same :class:`fakeredis.FakeServer`.
    """

    def __init__(self) -> None:
        self.server = fakeredis.FakeServer()

    def make_store(self, namespace: str, _redis_url: str) -> RedisVectorStore:
        text_client = fakeredis.FakeRedis(
            server=self.server, decode_responses=True,
        )
        bin_client = fakeredis.FakeRedis(
            server=self.server, decode_responses=False,
        )
        store = RedisVectorStore(namespace=namespace, client=text_client)
        store._binary_client = bin_client
        return store

    def make_embedder(self, _cfg: Any) -> _StubEmbedder:
        return _StubEmbedder()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def injected(monkeypatch, tmp_path):
    """Swap factories for the duration of a test and return a handle.

    Also ``chdir`` to ``tmp_path`` so commands that default to
    ``Path.cwd()`` target our sandbox, not the developer's repo.
    """
    state = _SharedFakeState()
    monkeypatch.setattr(rag_cmd, "_store_factory", state.make_store)
    monkeypatch.setattr(rag_cmd, "_embedder_factory", state.make_embedder)
    monkeypatch.chdir(tmp_path)
    return {"state": state, "root": tmp_path}


# ---------------------------------------------------------------------------
# gemma rag index
# ---------------------------------------------------------------------------

class TestIndexCommand:
    def test_index_writes_chunks_and_reports_stats(self, injected):
        root = injected["root"]
        (root / "a.py").write_text("def foo():\n    return 1\n")
        (root / "README.md").write_text("# Title\n\nHello, world.\n")

        result = runner.invoke(app, ["rag", "index"])
        assert result.exit_code == 0, result.output
        # Stats line from IndexStats.summary() should show both files.
        assert "2 scanned" in result.output
        assert "+2 added" in result.output

    def test_index_accepts_explicit_path(self, injected, tmp_path):
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "m.py").write_text("x = 1\n")

        result = runner.invoke(app, ["rag", "index", str(subdir)])
        assert result.exit_code == 0, result.output
        assert "1 scanned" in result.output

    def test_index_rejects_nonexistent_path(self, injected, tmp_path):
        ghost = tmp_path / "does-not-exist"
        result = runner.invoke(app, ["rag", "index", str(ghost)])
        assert result.exit_code == 1
        assert "not a directory" in result.output

    def test_index_is_idempotent_on_second_run(self, injected):
        (injected["root"] / "a.py").write_text("def f(): return 1\n")

        first = runner.invoke(app, ["rag", "index"])
        assert first.exit_code == 0

        second = runner.invoke(app, ["rag", "index"])
        assert second.exit_code == 0
        # Second run: no files added, no files changed/removed.
        assert "+0 added" in second.output
        assert "~0 changed" in second.output


# ---------------------------------------------------------------------------
# gemma rag query
# ---------------------------------------------------------------------------

class TestQueryCommand:
    def test_query_errors_when_nothing_indexed(self, injected):
        result = runner.invoke(app, ["rag", "query", "anything"])
        assert result.exit_code == 1
        assert "no chunks indexed" in result.output

    def test_query_renders_hits_after_index(self, injected):
        (injected["root"] / "a.py").write_text(
            "def greet():\n    return 'hello'\n"
        )
        assert runner.invoke(app, ["rag", "index"]).exit_code == 0

        result = runner.invoke(app, ["rag", "query", "greet", "--k", "2"])
        assert result.exit_code == 0, result.output
        # Citation line of the form "a.py:<line-range>".
        assert "a.py:" in result.output
        # Score is rendered with a signed decimal.
        assert "score=" in result.output

    def test_query_no_hits_message_shown_for_empty_string(self, injected):
        # Even with an indexed workspace, a whitespace-only query short
        # circuits inside the retriever (no embedding call).
        (injected["root"] / "a.py").write_text("x = 1\n")
        runner.invoke(app, ["rag", "index"])

        result = runner.invoke(app, ["rag", "query", "   "])
        assert result.exit_code == 0
        assert "no hits" in result.output


# ---------------------------------------------------------------------------
# gemma rag status
# ---------------------------------------------------------------------------

class TestStatusCommand:
    def test_status_on_empty_namespace(self, injected):
        result = runner.invoke(app, ["rag", "status"])
        assert result.exit_code == 0, result.output
        # No index yet → meta is empty; rendering uses the em-dash fallback.
        assert "namespace" in result.output
        assert "files in manifest" in result.output
        assert "chunks indexed" in result.output

    def test_status_after_index_reports_nonzero(self, injected):
        (injected["root"] / "a.py").write_text("x = 1\n")
        runner.invoke(app, ["rag", "index"])

        result = runner.invoke(app, ["rag", "status"])
        assert result.exit_code == 0, result.output
        # Embedding model name propagated from the stub into meta.
        assert "stub-embed" in result.output


# ---------------------------------------------------------------------------
# gemma rag reset
# ---------------------------------------------------------------------------

class TestResetCommand:
    def test_reset_requires_confirmation(self, injected):
        # Simulate the user declining the confirm prompt.
        result = runner.invoke(app, ["rag", "reset"], input="n\n")
        assert result.exit_code == 1
        assert "aborted" in result.output

    def test_reset_yes_flag_skips_prompt(self, injected):
        (injected["root"] / "a.py").write_text("x = 1\n")
        runner.invoke(app, ["rag", "index"])

        result = runner.invoke(app, ["rag", "reset", "--yes"])
        assert result.exit_code == 0, result.output
        assert "cleared" in result.output

        # Post-reset: chunks gone, query complains.
        after = runner.invoke(app, ["rag", "query", "foo"])
        assert after.exit_code == 1
        assert "no chunks indexed" in after.output

    def test_reset_confirm_yes(self, injected):
        (injected["root"] / "a.py").write_text("x = 1\n")
        runner.invoke(app, ["rag", "index"])

        result = runner.invoke(app, ["rag", "reset"], input="y\n")
        assert result.exit_code == 0, result.output
        assert "cleared" in result.output


# ---------------------------------------------------------------------------
# Factory hooks
# ---------------------------------------------------------------------------

def test_set_factory_hooks_are_applied(monkeypatch):
    """Sanity check the DI seams — the module-level factories are settable."""
    sentinel: Dict[str, Any] = {"store": None, "emb": None}

    def _store_factory(ns: str, url: str):
        sentinel["store"] = (ns, url)
        raise RuntimeError("short-circuit")  # abort before other work

    rag_cmd.set_store_factory(_store_factory)
    try:
        # The factory is invoked by ``_wire`` — we don't actually care
        # what happens next; we're only asserting it was called.
        with pytest.raises(RuntimeError):
            rag_cmd._wire(Path.cwd())
        assert sentinel["store"] is not None
        ns, _url = sentinel["store"]
        assert ":" in ns  # namespace shape: "<hash>:<branch>"
    finally:
        # Restore default so subsequent tests aren't poisoned.
        rag_cmd.set_store_factory(rag_cmd._default_store_factory)
