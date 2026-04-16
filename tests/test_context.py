"""Unit tests for the ContextAssembler."""

from __future__ import annotations

import pytest

from gemma.config import Config
from gemma.memory.context import ContextAssembler
from gemma.memory.models import ConversationTurn, MemoryCategory, MemoryRecord


@pytest.fixture
def assembler() -> ContextAssembler:
    return ContextAssembler(Config())


# ---------------------------------------------------------------------------
# build_messages
# ---------------------------------------------------------------------------

def test_build_messages_without_memories(assembler):
    turns = [
        ConversationTurn("user", "q1", turn_number=1),
        ConversationTurn("assistant", "a1", turn_number=2),
    ]
    messages = assembler.build_messages(
        system_prompt="You help users.",
        relevant_memories=[],
        recent_turns=turns,
    )
    assert len(messages) == 3
    assert messages[0] == {"role": "system", "content": "You help users."}
    assert messages[1] == {"role": "user", "content": "q1"}
    assert messages[2] == {"role": "assistant", "content": "a1"}


def test_build_messages_folds_memory_block_into_system(assembler, sample_memories):
    messages = assembler.build_messages(
        system_prompt="You help users.",
        relevant_memories=sample_memories,
        recent_turns=[],
    )
    assert len(messages) == 1
    assert messages[0]["role"] == "system"
    system_content = messages[0]["content"]
    assert system_content.startswith("You help users.")
    assert "Relevant memories" in system_content
    for mem in sample_memories:
        assert mem.content in system_content


def test_memory_block_sorted_by_importance_descending(assembler):
    memories = [
        MemoryRecord(content="low", category=MemoryCategory.FACTUAL_CONTEXT, importance=1, session_id="s"),
        MemoryRecord(content="high", category=MemoryCategory.USER_PREFERENCE, importance=5, session_id="s"),
        MemoryRecord(content="mid", category=MemoryCategory.TASK_STATE, importance=3, session_id="s"),
    ]
    messages = assembler.build_messages("sys", memories, [])
    system = messages[0]["content"]
    idx_high = system.index("high")
    idx_mid = system.index("mid")
    idx_low = system.index("low")
    assert idx_high < idx_mid < idx_low


def test_build_messages_drops_stored_system_turns(assembler):
    """System turns stored in Redis shouldn't duplicate our fresh system message."""
    turns = [
        ConversationTurn("system", "old system prompt", turn_number=0),
        ConversationTurn("user", "hi", turn_number=1),
    ]
    messages = assembler.build_messages("new system prompt", [], turns)
    roles = [m["role"] for m in messages]
    # Only one system message, and it's the fresh one
    assert roles.count("system") == 1
    assert messages[0]["content"] == "new system prompt"


# ---------------------------------------------------------------------------
# token estimation
# ---------------------------------------------------------------------------

def test_estimate_token_count_scales_with_content():
    small = [{"role": "user", "content": "hi"}]
    large = [{"role": "user", "content": "x" * 3500}]
    s = ContextAssembler.estimate_token_count(small)
    l = ContextAssembler.estimate_token_count(large)
    assert l > s
    # 3500 chars / 3.5 ~= 1000 tokens, plus overhead
    assert l >= 1000


# ---------------------------------------------------------------------------
# trim_to_budget
# ---------------------------------------------------------------------------

def test_trim_to_budget_returns_unchanged_when_within(assembler):
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]
    out = assembler.trim_to_budget(list(messages), max_tokens=10000)
    assert out == messages


def test_trim_drops_old_turns_but_keeps_last_two(assembler):
    big = "x" * 400
    messages = [{"role": "system", "content": "sys"}]
    # Add several big turns -- total chars ~ 2400, ~ 685 tokens
    for i in range(6):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": big})

    trimmed = assembler.trim_to_budget(list(messages), max_tokens=250)
    # System stays + at least 2 turns remain
    assert trimmed[0]["role"] == "system"
    assert len(trimmed) >= 3
    # The last two preserved turns are the most recent
    assert trimmed[-1] == messages[-1]
    assert trimmed[-2] == messages[-2]


def test_trim_shrinks_memory_block_first(assembler, sample_memories):
    # Build messages WITH a memory block via the assembler
    sys_prompt = "You help."
    messages = assembler.build_messages(sys_prompt, sample_memories, [])
    original_sys_len = len(messages[0]["content"])

    # Budget small enough to force memory-block trimming
    trimmed = assembler.trim_to_budget(list(messages), max_tokens=20)
    # System message should still exist, but shorter
    assert trimmed[0]["role"] == "system"
    assert len(trimmed[0]["content"]) < original_sys_len
