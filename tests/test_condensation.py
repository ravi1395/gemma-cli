"""Unit tests for the condensation pipeline.

Mocks the Ollama chat call so we can validate parsing behavior end-to-end
without running Gemma.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gemma.config import Config
from gemma.memory.condensation import CondensationPipeline
from gemma.memory.models import ConversationTurn, MemoryCategory


def _mock_client(response_text: str) -> MagicMock:
    client = MagicMock()
    client.chat.return_value = {"message": {"content": response_text}}
    return client


@pytest.fixture
def cfg() -> Config:
    return Config()


# ---------------------------------------------------------------------------
# Clean JSON
# ---------------------------------------------------------------------------

def test_parses_clean_json_array(cfg):
    client = _mock_client(
        '[{"content": "User prefers Python", "category": "user_preference", "importance": 4}]'
    )
    pipeline = CondensationPipeline(cfg, client=client)
    turns = [ConversationTurn("user", "I like Python", turn_number=1)]
    records = pipeline.condense_turns(turns, session_id="s1")

    assert len(records) == 1
    rec = records[0]
    assert rec.content == "User prefers Python"
    assert rec.category is MemoryCategory.USER_PREFERENCE
    assert rec.importance == 4
    assert rec.session_id == "s1"
    assert rec.turn_range == "1-1"


def test_multiple_memories_extracted(cfg):
    client = _mock_client(
        """[
          {"content": "Fact A", "category": "task_state", "importance": 3},
          {"content": "Fact B", "category": "factual_context", "importance": 2}
        ]"""
    )
    pipeline = CondensationPipeline(cfg, client=client)
    turns = [
        ConversationTurn("user", "A", turn_number=5),
        ConversationTurn("assistant", "B", turn_number=6),
    ]
    records = pipeline.condense_turns(turns, session_id="s")
    assert len(records) == 2
    assert records[0].category is MemoryCategory.TASK_STATE
    assert records[1].category is MemoryCategory.FACTUAL_CONTEXT
    assert all(r.turn_range == "5-6" for r in records)


# ---------------------------------------------------------------------------
# Fallback parsing
# ---------------------------------------------------------------------------

def test_parses_markdown_fenced_json(cfg):
    client = _mock_client(
        """Sure, here is the output:
```json
[{"content": "X", "category": "factual_context", "importance": 2}]
```
Let me know if you need more!"""
    )
    pipeline = CondensationPipeline(cfg, client=client)
    records = pipeline.condense_turns(
        [ConversationTurn("user", "q", 1)], session_id="s"
    )
    assert len(records) == 1
    assert records[0].content == "X"


def test_parses_json_with_leading_text(cfg):
    client = _mock_client(
        'Here are the memories:\n[{"content": "Y", "category": "user_preference", "importance": 3}]'
    )
    pipeline = CondensationPipeline(cfg, client=client)
    records = pipeline.condense_turns(
        [ConversationTurn("user", "q", 1)], session_id="s"
    )
    assert len(records) == 1
    assert records[0].content == "Y"


def test_returns_empty_on_malformed_json(cfg):
    client = _mock_client("{not json at all")
    pipeline = CondensationPipeline(cfg, client=client)
    records = pipeline.condense_turns(
        [ConversationTurn("user", "q", 1)], session_id="s"
    )
    assert records == []


def test_returns_empty_on_empty_response(cfg):
    client = _mock_client("")
    pipeline = CondensationPipeline(cfg, client=client)
    records = pipeline.condense_turns(
        [ConversationTurn("user", "q", 1)], session_id="s"
    )
    assert records == []


def test_returns_empty_when_no_turns(cfg):
    client = _mock_client("[]")
    pipeline = CondensationPipeline(cfg, client=client)
    assert pipeline.condense_turns([], session_id="s") == []
    # With no turns, the model shouldn't even be called
    client.chat.assert_not_called()


# ---------------------------------------------------------------------------
# Sanitization
# ---------------------------------------------------------------------------

def test_unknown_category_falls_back(cfg):
    client = _mock_client(
        '[{"content": "Z", "category": "invented_category", "importance": 3}]'
    )
    pipeline = CondensationPipeline(cfg, client=client)
    records = pipeline.condense_turns(
        [ConversationTurn("user", "q", 1)], session_id="s"
    )
    assert len(records) == 1
    assert records[0].category is MemoryCategory.FACTUAL_CONTEXT


def test_importance_is_clamped(cfg):
    client = _mock_client(
        '[{"content": "A", "category": "task_state", "importance": 99}]'
    )
    pipeline = CondensationPipeline(cfg, client=client)
    records = pipeline.condense_turns(
        [ConversationTurn("user", "q", 1)], session_id="s"
    )
    assert records[0].importance == 5


def test_skips_blank_content(cfg):
    client = _mock_client(
        """[
          {"content": "", "category": "factual_context", "importance": 3},
          {"content": "  ", "category": "factual_context", "importance": 3},
          {"content": "kept", "category": "factual_context", "importance": 3}
        ]"""
    )
    pipeline = CondensationPipeline(cfg, client=client)
    records = pipeline.condense_turns(
        [ConversationTurn("user", "q", 1)], session_id="s"
    )
    assert len(records) == 1
    assert records[0].content == "kept"


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def test_prompt_includes_existing_memories(cfg, sample_memories):
    client = _mock_client("[]")
    pipeline = CondensationPipeline(cfg, client=client)
    pipeline.condense_turns(
        [ConversationTurn("user", "q", 1)],
        existing_memories=sample_memories,
        session_id="s",
    )
    prompt = client.chat.call_args.kwargs["messages"][0]["content"]
    for mem in sample_memories:
        assert mem.content in prompt


def test_prompt_shows_none_when_no_existing(cfg):
    client = _mock_client("[]")
    pipeline = CondensationPipeline(cfg, client=client)
    pipeline.condense_turns(
        [ConversationTurn("user", "q", 1)], session_id="s"
    )
    prompt = client.chat.call_args.kwargs["messages"][0]["content"]
    assert "(none)" in prompt


# ---------------------------------------------------------------------------
# Reconsolidation
# ---------------------------------------------------------------------------

def test_reconsolidate_returns_merged(cfg, sample_memories):
    client = _mock_client(
        '[{"content": "Combined fact", "category": "factual_context", "importance": 3}]'
    )
    pipeline = CondensationPipeline(cfg, client=client)
    merged = pipeline.reconsolidate(sample_memories)
    assert len(merged) == 1
    assert merged[0].content == "Combined fact"
    # Session id is inherited from the originals
    assert merged[0].session_id == sample_memories[0].session_id


def test_reconsolidate_with_one_memory_noop(cfg, sample_memories):
    client = _mock_client("[]")
    pipeline = CondensationPipeline(cfg, client=client)
    result = pipeline.reconsolidate([sample_memories[0]])
    assert result == [sample_memories[0]]
    client.chat.assert_not_called()


def test_reconsolidate_preserves_originals_on_bad_response(cfg, sample_memories):
    client = _mock_client("not json")
    pipeline = CondensationPipeline(cfg, client=client)
    result = pipeline.reconsolidate(sample_memories)
    assert result == sample_memories
