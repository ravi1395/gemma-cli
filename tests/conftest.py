"""Shared pytest fixtures for gemma-cli tests.

Provides:
  - `fake_redis`: an in-memory fakeredis instance configured like the real one
  - `store`: a MemoryStore wired to that fakeredis instance
  - `cfg`: a Config with a temp history file
  - `sample_turns`: a list of ConversationTurn for condensation/retrieval tests
  - `sample_memories`: a list of MemoryRecord across categories
"""

from __future__ import annotations

from typing import Iterator

import fakeredis
import pytest

from gemma.config import Config
from gemma.memory.models import (
    ConversationTurn,
    MemoryCategory,
    MemoryRecord,
)
from gemma.memory.store import MemoryStore


# ---------------------------------------------------------------------------
# Warm-start guard (item #4)
# ---------------------------------------------------------------------------
#
# Production defaults to ``warm_start=True`` so CLI invocations prime Ollama
# in the background. Under pytest that would spawn real HTTP threads every
# time a CliRunner exercises ``main_callback``; worse, it would hit whatever
# Ollama happens to be on the dev machine. We short-circuit the spawn at
# its single call site so no test is affected regardless of how it builds
# its ``Config``.
#
# Tests that specifically exercise warm-start (see tests/test_warm_start.py)
# opt in locally by re-patching ``_spawn_warm_start`` back to the real
# implementation on their own fixtures.

@pytest.fixture(autouse=True)
def _disable_warm_start_in_tests(monkeypatch):
    """Make ``_spawn_warm_start`` a no-op for the entire test session."""
    import gemma.main as _main

    monkeypatch.setattr(_main, "_spawn_warm_start", lambda cfg: None)


@pytest.fixture
def fake_redis() -> Iterator:
    """In-memory Redis. decode_responses matches what MemoryStore uses."""
    client = fakeredis.FakeRedis(decode_responses=True)
    yield client
    client.flushall()


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(history_file=str(tmp_path / "history.json"))


@pytest.fixture
def store(fake_redis, cfg) -> MemoryStore:
    s = MemoryStore(cfg, client=fake_redis)
    # Force health to True without a live PING -- fakeredis responds to ping
    assert s.connect()
    return s


@pytest.fixture
def sample_turns() -> list[ConversationTurn]:
    return [
        ConversationTurn(role="user", content="I prefer Python over Java.", turn_number=1),
        ConversationTurn(role="assistant", content="Noted. Any frameworks you use?", turn_number=2),
        ConversationTurn(role="user", content="FastAPI and Typer.", turn_number=3),
        ConversationTurn(role="assistant", content="Great, both are modern.", turn_number=4),
    ]


@pytest.fixture
def sample_memories() -> list[MemoryRecord]:
    return [
        MemoryRecord(
            content="User prefers Python.",
            category=MemoryCategory.USER_PREFERENCE,
            importance=4,
            session_id="s1",
        ),
        MemoryRecord(
            content="User is building a CLI called gemma-cli.",
            category=MemoryCategory.TASK_STATE,
            importance=5,
            session_id="s1",
        ),
        MemoryRecord(
            content="Discussed the weather briefly.",
            category=MemoryCategory.FACTUAL_CONTEXT,
            importance=1,
            session_id="s1",
        ),
    ]
