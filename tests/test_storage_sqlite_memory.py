"""Tests for :class:`gemma.storage.sqlite_memory.SQLiteMemoryStore`.

Each test gets its own temp file so we never touch
``~/.gemma/store.sqlite`` and never interfere with the developer's
own running gemma. Tests focus on behavioural parity with the Redis
``MemoryStore`` — the same call patterns the manager uses today must
produce equivalent state.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from gemma.config import Config
from gemma.memory.models import (
    ConversationTurn,
    MemoryCategory,
    MemoryRecord,
)
from gemma.storage.sqlite_memory import SQLiteMemoryStore


@pytest.fixture
def cfg(tmp_path) -> Config:
    """Config pointing at a per-test SQLite file."""
    return Config(
        storage_backend="sqlite",
        sqlite_path=str(tmp_path / "store.sqlite"),
    )


@pytest.fixture
def store(cfg) -> SQLiteMemoryStore:
    s = SQLiteMemoryStore(cfg)
    assert s.connect()
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Memory CRUD
# ---------------------------------------------------------------------------

def test_save_and_get_round_trip(store):
    rec = MemoryRecord(
        content="User prefers Python",
        category=MemoryCategory.USER_PREFERENCE,
        importance=4,
        session_id="s1",
        turn_range="3-5",
    )
    store.save_memory(rec)
    loaded = store.get_memory(rec.memory_id, bump_access=False)

    assert loaded is not None
    assert loaded.content == rec.content
    assert loaded.category is MemoryCategory.USER_PREFERENCE
    assert loaded.importance == 4
    assert loaded.session_id == "s1"
    assert loaded.turn_range == "3-5"


def test_bump_access_updates_counter(store):
    rec = MemoryRecord(
        content="x", category=MemoryCategory.FACTUAL_CONTEXT,
        importance=2, session_id="s1",
    )
    store.save_memory(rec)
    before = store.get_memory(rec.memory_id, bump_access=False)
    assert before.access_count == 0

    store.get_memory(rec.memory_id, bump_access=True)
    after = store.get_memory(rec.memory_id, bump_access=False)
    assert after.access_count >= 1


def test_get_all_active_memories_skips_superseded(store):
    a = MemoryRecord(content="A", category=MemoryCategory.TASK_STATE, importance=3, session_id="s1")
    b = MemoryRecord(content="B", category=MemoryCategory.TASK_STATE, importance=3, session_id="s1")
    store.save_memory(a)
    store.save_memory(b)
    store.supersede_memory(a.memory_id, b.memory_id)

    active = store.get_all_active_memories()
    contents = [m.content for m in active]
    assert "B" in contents
    assert "A" not in contents


def test_top_memories_orders_by_importance(store):
    high = MemoryRecord(content="H", category=MemoryCategory.CORRECTION, importance=5, session_id="s1")
    low = MemoryRecord(content="L", category=MemoryCategory.FACTUAL_CONTEXT, importance=1, session_id="s1")
    store.save_memory(low)
    store.save_memory(high)

    top = store.get_top_memories(2)
    assert [m.content for m in top] == ["H", "L"]


def test_count_active_memories(store):
    assert store.count_active_memories() == 0
    store.save_memory(MemoryRecord(content="x", category=MemoryCategory.TASK_STATE, importance=3, session_id="s1"))
    store.save_memory(MemoryRecord(content="y", category=MemoryCategory.TASK_STATE, importance=3, session_id="s1"))
    assert store.count_active_memories() == 2


# ---------------------------------------------------------------------------
# TTL sweep
# ---------------------------------------------------------------------------

def test_ttl_expiry_is_swept_on_read(store, cfg, monkeypatch):
    """Memories past ``expires_at`` disappear from queries automatically."""
    # Importance-1 memory has the shortest TTL (6 hours). Build one
    # whose creation timestamp is 7 hours in the past so the computed
    # ``expires_at`` is already behind us.
    ttl_for_low = cfg.ttl_for(1)
    assert ttl_for_low is not None
    rec = MemoryRecord(
        content="ephemeral",
        category=MemoryCategory.FACTUAL_CONTEXT,
        importance=1,
        session_id="s1",
        created_at=time.time() - (ttl_for_low + 60),
    )
    store.save_memory(rec)

    # The save itself runs a sweep; after that, the row should be gone.
    assert store.get_memory(rec.memory_id, bump_access=False) is None
    assert store.count_active_memories() == 0


def test_high_importance_never_expires(store):
    """importance=5 → ``expires_at = NULL`` and stays forever."""
    rec = MemoryRecord(
        content="critical",
        category=MemoryCategory.CORRECTION,
        importance=5,
        session_id="s1",
        created_at=time.time() - 10 * 365 * 86400,  # 10 years ago
    )
    store.save_memory(rec)
    assert store.get_memory(rec.memory_id, bump_access=False) is not None


# ---------------------------------------------------------------------------
# Embedding storage + retrieval
# ---------------------------------------------------------------------------

def test_save_and_get_embedding_round_trip(store):
    rec = MemoryRecord(content="x", category=MemoryCategory.TASK_STATE, importance=3, session_id="s1")
    store.save_memory(rec)
    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    store.save_embedding(rec.memory_id, vec)

    loaded = store.get_embedding(rec.memory_id)
    assert loaded is not None
    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, vec)


def test_get_all_embeddings_returns_only_active(store):
    rec_a = MemoryRecord(content="A", category=MemoryCategory.TASK_STATE, importance=3, session_id="s1")
    rec_b = MemoryRecord(content="B", category=MemoryCategory.TASK_STATE, importance=3, session_id="s1")
    store.save_memory(rec_a)
    store.save_memory(rec_b)
    store.save_embedding(rec_a.memory_id, np.array([1.0, 0.0], dtype=np.float32))
    store.save_embedding(rec_b.memory_id, np.array([0.0, 1.0], dtype=np.float32))

    embeds = store.get_all_embeddings()
    assert set(embeds.keys()) == {rec_a.memory_id, rec_b.memory_id}

    store.supersede_memory(rec_a.memory_id, rec_b.memory_id)
    embeds = store.get_all_embeddings()
    assert rec_a.memory_id not in embeds


def test_embedding_cascade_deletes_with_memory(store, cfg):
    """Deleting a memory row should cascade to its embedding."""
    rec = MemoryRecord(content="x", category=MemoryCategory.TASK_STATE, importance=3, session_id="s1")
    store.save_memory(rec)
    store.save_embedding(rec.memory_id, np.array([1.0, 2.0], dtype=np.float32))

    # Manually delete to validate the FK cascade.
    store.client.execute("DELETE FROM memories WHERE memory_id = ?", (rec.memory_id,))
    store.client.commit()
    assert store.get_embedding(rec.memory_id) is None


# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------

def test_push_and_get_recent_turns(store):
    for i in range(1, 6):
        store.push_turn("s1", ConversationTurn(role="user", content=f"t{i}", turn_number=i))
    recent = store.get_recent_turns("s1", 3)
    assert [t.content for t in recent] == ["t3", "t4", "t5"]


def test_overflow_returns_and_deletes(store):
    """Overflow turns are returned AND removed so the window stays bounded."""
    for i in range(1, 11):
        store.push_turn("s1", ConversationTurn(role="user", content=f"t{i}", turn_number=i))
    overflow = store.get_overflow_turns("s1", window_size=4)
    assert [t.content for t in overflow] == ["t1", "t2", "t3", "t4", "t5", "t6"]

    # Remaining count must be exactly window_size.
    assert store.get_turn_count("s1") == 4
    remaining = store.get_recent_turns("s1", 4)
    assert [t.content for t in remaining] == ["t7", "t8", "t9", "t10"]


def test_clear_session_removes_all_state(store):
    store.push_turn("s1", ConversationTurn(role="user", content="x", turn_number=1))
    store.clear_session("s1")
    assert store.get_turn_count("s1") == 0


# ---------------------------------------------------------------------------
# Generation counter
# ---------------------------------------------------------------------------

def test_generation_counter_increments_on_save(store):
    """save_memory bumps the counter; supersede bumps it too."""
    g0 = store.get_generation()
    rec = MemoryRecord(content="x", category=MemoryCategory.TASK_STATE, importance=3, session_id="s1")
    store.save_memory(rec)
    g1 = store.get_generation()
    assert g1 > g0

    other = MemoryRecord(content="y", category=MemoryCategory.TASK_STATE, importance=3, session_id="s1")
    store.save_memory(other)
    store.supersede_memory(rec.memory_id, other.memory_id)
    g2 = store.get_generation()
    assert g2 > g1
