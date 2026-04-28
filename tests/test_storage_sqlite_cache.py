"""Tests for :class:`gemma.storage.sqlite_cache.SQLiteResponseCache`.

The cache key is computed by ``ResponseCache._compute_key`` (shared
with the Redis path) so we test only the SQLite-specific behaviour:
on-disk persistence, TTL expiry, and the ``ttl_seconds == 0`` no-op
guard.
"""

from __future__ import annotations

import time

import pytest

from gemma.config import Config
from gemma.storage.sqlite_cache import SQLiteResponseCache


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(
        storage_backend="sqlite",
        sqlite_path=str(tmp_path / "store.sqlite"),
        cache_enabled=True,
        cache_ttl_seconds=3600,
    )


@pytest.fixture
def cache(cfg) -> SQLiteResponseCache:
    c = SQLiteResponseCache(cfg)
    yield c
    c.close()


_MESSAGES = [
    {"role": "system", "content": "be terse"},
    {"role": "user", "content": "hello"},
]


def test_put_and_get_round_trip(cache, cfg):
    assert cache.get(_MESSAGES, cfg) is None
    cache.put(_MESSAGES, cfg, "hi there")
    assert cache.get(_MESSAGES, cfg) == "hi there"


def test_different_messages_have_different_keys(cache, cfg):
    cache.put(_MESSAGES, cfg, "answer-a")
    other = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "different question"},
    ]
    assert cache.get(other, cfg) is None
    assert cache.get(_MESSAGES, cfg) == "answer-a"


def test_ttl_expiry_is_observable(cache, cfg):
    cache.put(_MESSAGES, cfg, "soon expires")
    # Force expiry directly so the test doesn't sleep.
    cache._conn.execute(
        "UPDATE response_cache SET expires_at = ?",
        (time.time() - 60,),
    )
    cache._conn.commit()
    assert cache.get(_MESSAGES, cfg) is None


def test_zero_ttl_disables_writes(tmp_path):
    """``cache_ttl_seconds == 0`` short-circuits put/get.

    Mirrors the Redis path's behaviour. The expectation is that the
    factory builds None instead — but if a caller bypasses the factory
    and constructs the cache directly, the guards still apply.
    """
    cfg = Config(
        storage_backend="sqlite",
        sqlite_path=str(tmp_path / "store.sqlite"),
        cache_enabled=True,
        cache_ttl_seconds=0,
    )
    cache = SQLiteResponseCache(cfg)
    cache.put(_MESSAGES, cfg, "ignored")
    assert cache.get(_MESSAGES, cfg) is None
