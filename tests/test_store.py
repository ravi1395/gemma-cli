"""Unit tests for the Redis-backed MemoryStore.

All tests use fakeredis (see conftest.py) so they run without a live Redis.
"""

from __future__ import annotations

import numpy as np
import pytest

from gemma.memory.models import (
    ConversationTurn,
    MemoryCategory,
    MemoryRecord,
)


# ---------------------------------------------------------------------------
# Memory CRUD
# ---------------------------------------------------------------------------

def test_save_and_get_memory_round_trip(store, sample_memories):
    rec = sample_memories[0]
    store.save_memory(rec)

    loaded = store.get_memory(rec.memory_id, bump_access=False)
    assert loaded is not None
    assert loaded.content == rec.content
    assert loaded.category is MemoryCategory.USER_PREFERENCE
    assert loaded.importance == 4


def test_get_memory_returns_none_when_missing(store):
    assert store.get_memory("does-not-exist") is None


def test_save_memory_indexes_by_importance(store, sample_memories):
    for rec in sample_memories:
        store.save_memory(rec)

    top = store.get_top_memories(n=10)
    # Descending by importance (5 -> 4 -> 1)
    scores = [r.importance for r in top]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == 5


def test_get_memory_bumps_access_count(store, sample_memories):
    rec = sample_memories[0]
    store.save_memory(rec)

    first = store.get_memory(rec.memory_id, bump_access=True)
    assert first is not None
    second = store.get_memory(rec.memory_id, bump_access=True)
    assert second is not None
    assert second.access_count == first.access_count + 1


def test_supersede_memory_removes_from_index(store, sample_memories):
    old = sample_memories[0]
    new_id = "replacement-id"
    store.save_memory(old)

    store.supersede_memory(old.memory_id, new_id)

    # Hash still exists with the supersede pointer...
    loaded = store.get_memory(old.memory_id, bump_access=False)
    assert loaded is not None
    assert loaded.superseded_by == new_id
    assert loaded.is_active() is False

    # ...but the active index no longer contains it
    assert store.count_active_memories() == 0


def test_count_active_memories(store, sample_memories):
    assert store.count_active_memories() == 0
    for rec in sample_memories:
        store.save_memory(rec)
    assert store.count_active_memories() == len(sample_memories)


def test_get_all_active_memories(store, sample_memories):
    for rec in sample_memories:
        store.save_memory(rec)

    all_active = store.get_all_active_memories()
    assert len(all_active) == len(sample_memories)
    ids = {r.memory_id for r in all_active}
    assert ids == {r.memory_id for r in sample_memories}


def test_ttl_applied_for_low_importance(store, fake_redis):
    low = MemoryRecord(
        content="trivial",
        category=MemoryCategory.FACTUAL_CONTEXT,
        importance=1,     # 6 hours TTL
        session_id="s1",
    )
    store.save_memory(low)
    key = f"gemma:memory:{low.memory_id}"
    ttl = fake_redis.ttl(key)
    # ttl > 0 means an expiry was set; -1 would mean no expiry, -2 missing
    assert ttl > 0
    assert ttl <= 6 * 3600


def test_no_ttl_for_critical_importance(store, fake_redis):
    critical = MemoryRecord(
        content="user identity",
        category=MemoryCategory.USER_PREFERENCE,
        importance=5,
        session_id="s1",
    )
    store.save_memory(critical)
    key = f"gemma:memory:{critical.memory_id}"
    assert fake_redis.ttl(key) == -1


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def test_embedding_round_trip_preserves_values(store):
    vec = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32)
    store.save_embedding("m1", vec)

    loaded = store.get_embedding("m1")
    assert loaded is not None
    np.testing.assert_array_almost_equal(loaded, vec)


def test_get_embedding_returns_none_when_missing(store):
    assert store.get_embedding("nope") is None


def test_get_all_embeddings_only_returns_indexed_memories(store, sample_memories):
    # Save two memories and their embeddings; index only the first two.
    for i, rec in enumerate(sample_memories):
        store.save_memory(rec)
        store.save_embedding(rec.memory_id, np.array([float(i)] * 4, dtype=np.float32))

    mapping = store.get_all_embeddings()
    assert len(mapping) == len(sample_memories)
    for rec in sample_memories:
        assert rec.memory_id in mapping


# ---------------------------------------------------------------------------
# Session sliding window
# ---------------------------------------------------------------------------

def test_push_and_get_recent_turns(store):
    for i in range(5):
        store.push_turn(
            "session-A",
            ConversationTurn(role="user", content=f"msg {i}", turn_number=i),
        )

    recent = store.get_recent_turns("session-A", n=3)
    assert [t.content for t in recent] == ["msg 2", "msg 3", "msg 4"]


def test_get_recent_turns_handles_fewer_than_requested(store):
    store.push_turn(
        "s",
        ConversationTurn(role="user", content="only one", turn_number=0),
    )
    recent = store.get_recent_turns("s", n=10)
    assert len(recent) == 1
    assert recent[0].content == "only one"


def test_turn_count(store):
    assert store.get_turn_count("empty") == 0
    for i in range(4):
        store.push_turn("s", ConversationTurn("user", f"t{i}", i))
    assert store.get_turn_count("s") == 4


def test_get_overflow_trims_list(store):
    for i in range(10):
        store.push_turn(
            "s",
            ConversationTurn(role="user", content=f"msg {i}", turn_number=i),
        )

    overflow = store.get_overflow_turns("s", window_size=4)
    assert len(overflow) == 6
    assert [t.content for t in overflow] == [f"msg {i}" for i in range(6)]
    assert store.get_turn_count("s") == 4

    # The remaining turns are the most recent ones
    remaining = store.get_recent_turns("s", n=10)
    assert [t.content for t in remaining] == [f"msg {i}" for i in range(6, 10)]


def test_get_overflow_returns_empty_within_window(store):
    for i in range(3):
        store.push_turn("s", ConversationTurn("user", f"m{i}", i))
    overflow = store.get_overflow_turns("s", window_size=5)
    assert overflow == []
    assert store.get_turn_count("s") == 3


def test_clear_session_removes_turns(store):
    for i in range(3):
        store.push_turn("s", ConversationTurn("user", "x", i))
    store.clear_session("s")
    assert store.get_turn_count("s") == 0
