"""End-to-end tests for ``gemma storage {info, migrate}``.

Migration tests build a fakeredis source, run the migrate command,
and assert the destination SQLite file holds the expected rows. We
patch redis.Redis.from_url so the migrate command's internal "open
source" path resolves to fakeredis — same trick test_session.py uses.
"""

from __future__ import annotations

from unittest.mock import patch

import fakeredis
import numpy as np
import pytest
from typer.testing import CliRunner

from gemma.config import Config
from gemma.main import app
from gemma.memory.models import MemoryCategory, MemoryRecord


runner = CliRunner()


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------

def test_info_sqlite_renders_counts(tmp_path, monkeypatch):
    """``gemma storage info`` against a fresh SQLite file shows zero rows."""
    sqlite_path = tmp_path / "store.sqlite"
    monkeypatch.setenv("HOME", str(tmp_path))  # not strictly needed; defensive
    # Force the Config default by passing nothing; override path via env-style
    # by writing directly.
    from gemma.storage.sqlite_memory import SQLiteMemoryStore

    cfg = Config(storage_backend="sqlite", sqlite_path=str(sqlite_path))
    SQLiteMemoryStore(cfg)  # initialise the schema

    # Patch Config so the command sees our temp path
    with patch("gemma.commands.storage.Config", return_value=cfg):
        result = runner.invoke(app, ["storage", "info"])

    assert result.exit_code == 0, result.output
    assert "memories" in result.output
    assert "sqlite" in result.output
    # Rich may soft-wrap long paths across newlines, so match a stable
    # substring instead of the absolute path.
    assert "store.sqlite" in result.output


# ---------------------------------------------------------------------------
# migrate redis → sqlite
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_redis_with_memories():
    """Populate a FakeRedis with a couple of memories, then yield it."""
    server = fakeredis.FakeServer()
    client = fakeredis.FakeRedis(server=server, decode_responses=True)

    # We use the legacy MemoryStore directly to populate so the on-wire
    # shape exactly matches what production Redis would have.
    from gemma.memory.store import MemoryStore

    cfg = Config(storage_backend="redis")
    redis_store = MemoryStore(cfg, client=client)
    redis_store.connect()

    rec_a = MemoryRecord(
        content="User prefers Python",
        category=MemoryCategory.USER_PREFERENCE,
        importance=4,
        session_id="src1",
    )
    rec_b = MemoryRecord(
        content="Working on gemma-cli",
        category=MemoryCategory.TASK_STATE,
        importance=5,
        session_id="src1",
    )
    redis_store.save_memory(rec_a)
    redis_store.save_memory(rec_b)
    redis_store.save_embedding(rec_a.memory_id, np.array([0.1, 0.2, 0.3], dtype=np.float32))
    redis_store.save_embedding(rec_b.memory_id, np.array([0.9, 0.8, 0.7], dtype=np.float32))

    return client, [rec_a, rec_b]


def test_migrate_redis_to_sqlite_copies_memories(
    tmp_path, fake_redis_with_memories, monkeypatch,
):
    client, expected = fake_redis_with_memories
    sqlite_path = tmp_path / "store.sqlite"

    # Patch redis.Redis.from_url so the migrate command's internal
    # build_memory_store(redis) resolves to our FakeRedis.
    with patch("redis.Redis.from_url", return_value=client):
        # Patch Config inside the migrate command to use our temp paths
        # for both backends.
        def _cfg_factory(**kw):
            kw.setdefault("storage_backend", "sqlite")
            kw.setdefault("sqlite_path", str(sqlite_path))
            return Config(**kw)

        with patch("gemma.commands.storage.Config", side_effect=_cfg_factory):
            result = runner.invoke(
                app,
                ["storage", "migrate", "--from", "redis", "--to", "sqlite"],
            )

    assert result.exit_code == 0, result.output
    # Expected counts surface in the rendered table.
    assert "memories" in result.output

    # Verify destination state: open the SQLite store directly.
    from gemma.storage.sqlite_memory import SQLiteMemoryStore

    cfg = Config(storage_backend="sqlite", sqlite_path=str(sqlite_path))
    dst = SQLiteMemoryStore(cfg)
    contents = sorted(m.content for m in dst.get_all_active_memories())
    assert contents == sorted(m.content for m in expected)


def test_migrate_dry_run_leaves_destination_empty(
    tmp_path, fake_redis_with_memories,
):
    client, _expected = fake_redis_with_memories
    sqlite_path = tmp_path / "store.sqlite"

    with patch("redis.Redis.from_url", return_value=client):
        def _cfg_factory(**kw):
            kw.setdefault("storage_backend", "sqlite")
            kw.setdefault("sqlite_path", str(sqlite_path))
            return Config(**kw)

        with patch("gemma.commands.storage.Config", side_effect=_cfg_factory):
            result = runner.invoke(
                app,
                ["storage", "migrate", "--from", "redis", "--to", "sqlite", "--dry-run"],
            )

    assert result.exit_code == 0, result.output
    assert "dry run" in result.output

    # SQLite file should be empty (or absent).
    if sqlite_path.exists():
        from gemma.storage.sqlite_memory import SQLiteMemoryStore

        cfg = Config(storage_backend="sqlite", sqlite_path=str(sqlite_path))
        dst = SQLiteMemoryStore(cfg)
        assert dst.count_active_memories() == 0


def test_migrate_same_backend_is_noop():
    result = runner.invoke(
        app,
        ["storage", "migrate", "--from", "sqlite", "--to", "sqlite"],
    )
    assert result.exit_code == 0
    assert "nothing to do" in result.output.lower()


def test_migrate_unknown_backend_errors():
    result = runner.invoke(
        app,
        ["storage", "migrate", "--from", "lancedb", "--to", "sqlite"],
    )
    assert result.exit_code == 1
