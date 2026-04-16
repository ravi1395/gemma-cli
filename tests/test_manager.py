"""Unit / integration tests for the MemoryManager orchestrator.

Uses fakeredis + stub Embedder + stub CondensationPipeline so no external
services are required. Focuses on the coordination logic: sliding window,
condensation triggers, prompt assembly, degraded mode.
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from gemma.config import Config
from gemma.memory.manager import MemoryManager
from gemma.memory.models import (
    ConversationTurn,
    MemoryCategory,
    MemoryRecord,
)


class StubEmbedder:
    """Returns a deterministic vector based on text hash."""

    @property
    def model(self) -> str:
        return "stub"

    def embed(self, text: str) -> np.ndarray:
        # Map any input to a small deterministic vector
        seed = abs(hash(text)) % 1000 / 1000.0
        return np.array([1.0 - seed, seed, 0.0, 0.0], dtype=np.float32)


def _build_manager(
    store,
    cfg,
    *,
    extracted_records: Optional[list[MemoryRecord]] = None,
) -> MemoryManager:
    """Wire a MemoryManager with stubs for condensation and embedding."""
    embedder = StubEmbedder()
    pipeline = MagicMock()
    pipeline.condense_turns.return_value = extracted_records or []
    pipeline.reconsolidate.return_value = extracted_records or []

    # Run condensation synchronously in tests so we can assert on outcomes.
    cfg.condensation_async = False

    mgr = MemoryManager(
        cfg,
        store=store,
        embedder=embedder,  # type: ignore[arg-type]
        pipeline=pipeline,
        session_id="test-session",
    )
    mgr.initialize()
    return mgr


# ---------------------------------------------------------------------------
# Initialization + degraded mode
# ---------------------------------------------------------------------------

def test_initialize_sets_available(store, cfg):
    mgr = _build_manager(store, cfg)
    assert mgr.available is True
    assert mgr.degraded is False


def test_degraded_mode_when_memory_disabled(store, cfg):
    cfg.memory_enabled = False
    mgr = _build_manager(store, cfg)
    assert mgr.available is False
    assert mgr.degraded is True


def test_degraded_records_fallback_turns(store, cfg):
    cfg.memory_enabled = False
    mgr = _build_manager(store, cfg)
    mgr.record_turn("user", "hello")
    mgr.record_turn("assistant", "hi")

    messages = mgr.get_context_messages("next query", system_prompt="sys")
    assert messages[0]["role"] == "system"
    # Both fallback turns flow through
    contents = [m["content"] for m in messages]
    assert "hello" in contents
    assert "hi" in contents


# ---------------------------------------------------------------------------
# Sliding window + condensation trigger
# ---------------------------------------------------------------------------

def test_record_turn_within_window_no_condensation(store, cfg):
    cfg.sliding_window_size = 8
    mgr = _build_manager(store, cfg)
    for i in range(5):
        mgr.record_turn("user", f"msg {i}")

    assert store.get_turn_count("test-session") == 5
    assert store.count_active_memories() == 0
    # Pipeline.condense_turns should NOT have been called
    mgr._pipeline.condense_turns.assert_not_called()


def test_condensation_triggered_on_overflow(store, cfg):
    cfg.sliding_window_size = 4
    extracted = [
        MemoryRecord(
            content="Extracted fact",
            category=MemoryCategory.FACTUAL_CONTEXT,
            importance=3,
            session_id="test-session",
        )
    ]
    mgr = _build_manager(store, cfg, extracted_records=extracted)

    # Push more turns than the window allows
    for i in range(6):
        mgr.record_turn("user", f"turn {i}")

    # Overflow should have been condensed; pipeline called at least once
    assert mgr._pipeline.condense_turns.called
    # Window should now contain exactly the configured size
    assert store.get_turn_count("test-session") == cfg.sliding_window_size
    # New memory was saved
    assert store.count_active_memories() == 1


def test_condensation_embeds_and_stores_memories(store, cfg):
    cfg.sliding_window_size = 2
    extracted = [
        MemoryRecord(
            content="New memory",
            category=MemoryCategory.USER_PREFERENCE,
            importance=4,
            session_id="test-session",
        )
    ]
    mgr = _build_manager(store, cfg, extracted_records=extracted)

    for i in range(5):
        mgr.record_turn("user", f"t{i}")

    memories = store.get_all_active_memories()
    assert len(memories) == 1
    # Embedding should be persisted alongside the memory
    vec = store.get_embedding(memories[0].memory_id)
    assert vec is not None
    assert vec.size > 0


# ---------------------------------------------------------------------------
# get_context_messages
# ---------------------------------------------------------------------------

def test_context_includes_recent_turns(store, cfg):
    mgr = _build_manager(store, cfg)
    mgr.record_turn("user", "hello")
    mgr.record_turn("assistant", "hi there")

    messages = mgr.get_context_messages("next?", system_prompt="sys")
    # system + two turns
    assert messages[0]["role"] == "system"
    contents = [m["content"] for m in messages[1:]]
    assert "hello" in contents
    assert "hi there" in contents


def test_context_includes_retrieved_memories(store, cfg):
    mgr = _build_manager(store, cfg)
    # Pre-seed a relevant memory with an embedding
    rec = MemoryRecord(
        content="User likes dark mode",
        category=MemoryCategory.USER_PREFERENCE,
        importance=4,
        session_id="test-session",
    )
    store.save_memory(rec)
    vec = mgr._embedder.embed(rec.content)  # same stub produces same vec for same text
    store.save_embedding(rec.memory_id, vec)

    cfg.memory_min_similarity = 0.0  # accept anything
    messages = mgr.get_context_messages(rec.content, system_prompt="sys")
    sys_content = messages[0]["content"]
    assert "User likes dark mode" in sys_content


# ---------------------------------------------------------------------------
# Stats & introspection
# ---------------------------------------------------------------------------

def test_stats_when_degraded(store, cfg):
    cfg.memory_enabled = False
    mgr = _build_manager(store, cfg)
    mgr.record_turn("user", "x")

    stats = mgr.get_stats()
    assert stats["available"] is False
    assert stats["fallback_turns"] == 1


def test_stats_counts_active_memories(store, cfg, sample_memories):
    mgr = _build_manager(store, cfg)
    for rec in sample_memories:
        store.save_memory(rec)

    stats = mgr.get_stats()
    assert stats["available"] is True
    assert stats["active_memories"] == len(sample_memories)
    assert sum(stats["by_category"].values()) == len(sample_memories)


def test_list_memories_respects_limit(store, cfg, sample_memories):
    mgr = _build_manager(store, cfg)
    for rec in sample_memories:
        store.save_memory(rec)

    listed = mgr.list_memories(limit=2)
    assert len(listed) == 2


def test_clear_session_empties_window_but_keeps_memories(store, cfg, sample_memories):
    mgr = _build_manager(store, cfg)
    for rec in sample_memories:
        store.save_memory(rec)
    mgr.record_turn("user", "x")
    mgr.record_turn("assistant", "y")

    mgr.clear_session()
    assert store.get_turn_count("test-session") == 0
    # Condensed memories survive
    assert store.count_active_memories() == len(sample_memories)
