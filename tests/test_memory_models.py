"""Unit tests for memory dataclasses."""

import pytest

from gemma.memory.models import (
    ConversationTurn,
    MemoryCategory,
    MemoryRecord,
)


# ---------------------------------------------------------------------------
# MemoryCategory
# ---------------------------------------------------------------------------

def test_category_parse_known_values():
    assert MemoryCategory.parse("user_preference") is MemoryCategory.USER_PREFERENCE
    assert MemoryCategory.parse("TASK_STATE") is MemoryCategory.TASK_STATE
    assert MemoryCategory.parse("factual context") is MemoryCategory.FACTUAL_CONTEXT
    assert MemoryCategory.parse("tool-usage") is MemoryCategory.TOOL_USAGE


def test_category_parse_unknown_falls_back():
    assert MemoryCategory.parse("") is MemoryCategory.FACTUAL_CONTEXT
    assert MemoryCategory.parse("gibberish") is MemoryCategory.FACTUAL_CONTEXT


# ---------------------------------------------------------------------------
# ConversationTurn
# ---------------------------------------------------------------------------

def test_turn_to_message():
    turn = ConversationTurn(role="user", content="hello", turn_number=1)
    assert turn.to_message() == {"role": "user", "content": "hello"}
    assert turn.timestamp > 0


# ---------------------------------------------------------------------------
# MemoryRecord: importance clamping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [(-5, 1), (0, 1), (1, 1), (3, 3), (5, 5), (9, 5)],
)
def test_importance_is_clamped(raw, expected):
    rec = MemoryRecord(
        content="x",
        category=MemoryCategory.FACTUAL_CONTEXT,
        importance=raw,
        session_id="s1",
    )
    assert rec.importance == expected


def test_category_accepts_raw_string():
    rec = MemoryRecord(
        content="x",
        category="correction",  # type: ignore[arg-type]
        importance=3,
        session_id="s1",
    )
    assert rec.category is MemoryCategory.CORRECTION


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------

def test_round_trip_preserves_all_fields():
    original = MemoryRecord(
        content="User prefers Python",
        category=MemoryCategory.USER_PREFERENCE,
        importance=4,
        session_id="abc",
        turn_range="5-10",
        source_summary="discussed language prefs",
        access_count=3,
    )
    flat = original.to_redis_hash()
    assert all(isinstance(v, str) for v in flat.values())

    rebuilt = MemoryRecord.from_redis_hash(flat)
    assert rebuilt.memory_id == original.memory_id
    assert rebuilt.content == original.content
    assert rebuilt.category is MemoryCategory.USER_PREFERENCE
    assert rebuilt.importance == 4
    assert rebuilt.session_id == "abc"
    assert rebuilt.turn_range == "5-10"
    assert rebuilt.source_summary == "discussed language prefs"
    assert rebuilt.access_count == 3
    assert rebuilt.superseded_by is None


def test_superseded_field_round_trip():
    rec = MemoryRecord(
        content="old fact",
        category=MemoryCategory.FACTUAL_CONTEXT,
        importance=2,
        session_id="s",
        superseded_by="new-memory-id",
    )
    flat = rec.to_redis_hash()
    assert flat["superseded_by"] == "new-memory-id"

    rebuilt = MemoryRecord.from_redis_hash(flat)
    assert rebuilt.superseded_by == "new-memory-id"
    assert rebuilt.is_active() is False


def test_from_redis_hash_rejects_empty():
    with pytest.raises(ValueError):
        MemoryRecord.from_redis_hash({})


# ---------------------------------------------------------------------------
# TTL
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("importance", "expected"),
    [
        (5, None),
        (4, 7 * 86400),
        (3, 3 * 86400),
        (2, 86400),
        (1, 6 * 3600),
    ],
)
def test_ttl_default_mapping(importance, expected):
    rec = MemoryRecord(
        content="x",
        category=MemoryCategory.FACTUAL_CONTEXT,
        importance=importance,
        session_id="s",
    )
    assert rec.ttl_seconds() == expected


def test_ttl_custom_map_overrides_default():
    rec = MemoryRecord(
        content="x",
        category=MemoryCategory.FACTUAL_CONTEXT,
        importance=3,
        session_id="s",
    )
    custom = {1: 10, 2: 20, 3: 30, 4: 40, 5: 50}
    assert rec.ttl_seconds(custom) == 30


def test_is_active_true_by_default():
    rec = MemoryRecord(
        content="x",
        category=MemoryCategory.FACTUAL_CONTEXT,
        importance=3,
        session_id="s",
    )
    assert rec.is_active() is True
