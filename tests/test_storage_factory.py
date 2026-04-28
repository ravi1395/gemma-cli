"""Tests for the storage backend factory (:mod:`gemma.storage`).

Confirms each ``build_*`` helper picks the right implementation
based on ``Config.storage_backend`` and that unknown names raise
loudly. We don't reach Redis here — fakeredis isn't needed because
we only assert the *type* of the returned store, not its
connectivity.
"""

from __future__ import annotations

import pytest

from gemma.config import Config


def test_build_memory_store_default_returns_sqlite(tmp_path):
    cfg = Config(
        storage_backend="sqlite",
        sqlite_path=str(tmp_path / "store.sqlite"),
    )
    from gemma.storage import build_memory_store
    from gemma.storage.sqlite_memory import SQLiteMemoryStore

    store = build_memory_store(cfg)
    assert isinstance(store, SQLiteMemoryStore)


def test_build_memory_store_redis_returns_redis_class(tmp_path):
    cfg = Config(
        storage_backend="redis",
        redis_url="redis://localhost:9999",  # never connected; just a type check
    )
    from gemma.memory.store import MemoryStore as RedisMemoryStore
    from gemma.storage import build_memory_store

    store = build_memory_store(cfg)
    assert isinstance(store, RedisMemoryStore)


def test_build_rag_store_default_returns_sqlite(tmp_path):
    cfg = Config(
        storage_backend="sqlite",
        sqlite_path=str(tmp_path / "store.sqlite"),
    )
    from gemma.storage import build_rag_store
    from gemma.storage.sqlite_rag import SQLiteRAGStore

    store = build_rag_store(cfg, "test_ns")
    assert isinstance(store, SQLiteRAGStore)


def test_build_response_cache_returns_none_when_disabled(tmp_path):
    cfg = Config(
        storage_backend="sqlite",
        sqlite_path=str(tmp_path / "store.sqlite"),
        cache_enabled=False,
    )
    from gemma.storage import build_response_cache

    assert build_response_cache(cfg) is None


def test_build_response_cache_returns_none_when_ttl_zero(tmp_path):
    cfg = Config(
        storage_backend="sqlite",
        sqlite_path=str(tmp_path / "store.sqlite"),
        cache_ttl_seconds=0,
    )
    from gemma.storage import build_response_cache

    assert build_response_cache(cfg) is None


def test_unknown_backend_raises():
    cfg = Config(storage_backend="lancedb")  # not implemented
    from gemma.storage import build_memory_store

    with pytest.raises(ValueError, match="Unknown storage_backend"):
        build_memory_store(cfg)


def test_two_factory_calls_open_independent_handles(tmp_path):
    """Multiple ``build_memory_store`` calls share the file but get fresh
    connections — important so closing one doesn't break the other."""
    cfg = Config(
        storage_backend="sqlite",
        sqlite_path=str(tmp_path / "store.sqlite"),
    )
    from gemma.storage import build_memory_store

    s1 = build_memory_store(cfg)
    s2 = build_memory_store(cfg)
    assert s1 is not s2
    assert s1.client is not s2.client
