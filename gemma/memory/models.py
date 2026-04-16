"""Data models for the memory system.

Pure dataclasses with no I/O. Includes serialization helpers for Redis
hash storage (all string values) and TTL calculation by importance.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class MemoryCategory(str, Enum):
    """Fixed categories for condensed memories.

    Using an enum (instead of free-form strings) keeps the extraction prompt
    constrained and makes downstream consumers reliable. The string values
    are what get persisted in Redis.
    """

    USER_PREFERENCE = "user_preference"
    TASK_STATE = "task_state"
    FACTUAL_CONTEXT = "factual_context"
    INSTRUCTION = "instruction"
    CORRECTION = "correction"
    RELATIONSHIP = "relationship"
    TOOL_USAGE = "tool_usage"

    @classmethod
    def parse(cls, value: str) -> "MemoryCategory":
        """Lenient parser: accept any case, fall back to FACTUAL_CONTEXT."""
        if not value:
            return cls.FACTUAL_CONTEXT
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == normalized:
                return member
        return cls.FACTUAL_CONTEXT


@dataclass
class ConversationTurn:
    """A single user or assistant turn in the sliding window."""

    role: str                    # "user" | "assistant" | "system"
    content: str
    turn_number: int
    timestamp: float = field(default_factory=time.time)

    def to_message(self) -> dict[str, str]:
        """Render as an Ollama chat message."""
        return {"role": self.role, "content": self.content}


@dataclass
class MemoryRecord:
    """A single condensed memory persisted in Redis.

    Fields map 1:1 to the Redis hash at `gemma:memory:{memory_id}`. All
    values are stringified by `to_redis_hash()` because Redis hashes are
    flat string->string maps.
    """

    content: str
    category: MemoryCategory
    importance: int                          # 1 (trivial) -- 5 (critical)
    session_id: str
    turn_range: str = ""                     # e.g. "12-18"
    source_summary: str = ""
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    superseded_by: Optional[str] = None      # memory_id that replaced this one

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        # Clamp importance into 1-5 so downstream TTL / indexing is safe
        self.importance = max(1, min(5, int(self.importance)))
        # Accept raw strings for category and normalize via the enum parser
        if not isinstance(self.category, MemoryCategory):
            self.category = MemoryCategory.parse(str(self.category))

    # ------------------------------------------------------------------
    # Redis (de)serialization
    # ------------------------------------------------------------------

    def to_redis_hash(self) -> dict[str, str]:
        """Serialize to a flat string dict suitable for Redis HSET."""
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "category": self.category.value,
            "importance": str(self.importance),
            "session_id": self.session_id,
            "turn_range": self.turn_range,
            "source_summary": self.source_summary,
            "created_at": f"{self.created_at:.6f}",
            "last_accessed": f"{self.last_accessed:.6f}",
            "access_count": str(self.access_count),
            "superseded_by": self.superseded_by or "",
        }

    @classmethod
    def from_redis_hash(cls, data: dict[str, str]) -> "MemoryRecord":
        """Reconstruct from an HGETALL result. Tolerates missing fields."""
        if not data:
            raise ValueError("cannot build MemoryRecord from empty hash")

        superseded = data.get("superseded_by") or None
        return cls(
            memory_id=data["memory_id"],
            content=data.get("content", ""),
            category=MemoryCategory.parse(data.get("category", "")),
            importance=int(data.get("importance", 1)),
            session_id=data.get("session_id", ""),
            turn_range=data.get("turn_range", ""),
            source_summary=data.get("source_summary", ""),
            created_at=float(data.get("created_at", 0.0) or 0.0),
            last_accessed=float(data.get("last_accessed", 0.0) or 0.0),
            access_count=int(data.get("access_count", 0) or 0),
            superseded_by=superseded,
        )

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    def ttl_seconds(self, ttl_map: Optional[dict[int, Optional[int]]] = None) -> Optional[int]:
        """Return the TTL for this record in seconds, or None for no expiry.

        If a custom ttl_map is provided (typically from Config), it takes
        precedence; otherwise the default mapping is used.
        """
        default_map: dict[int, Optional[int]] = {
            5: None,
            4: 7 * 86400,
            3: 3 * 86400,
            2: 86400,
            1: 6 * 3600,
        }
        effective = ttl_map if ttl_map is not None else default_map
        return effective.get(self.importance)

    def is_active(self) -> bool:
        """A memory is active if it has not been superseded by a newer one."""
        return not self.superseded_by
