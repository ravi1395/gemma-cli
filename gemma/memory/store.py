"""Redis-backed storage for condensed memories, embeddings, and session turns.

Key layout (see plan.md for full data model):
    gemma:session:{sid}:turns        List    raw conversation turns
    gemma:session:{sid}:meta         Hash    session metadata
    gemma:memory:{mid}               Hash    single MemoryRecord
    gemma:memory:index               ZSet    active memory_ids scored by importance
    gemma:memory:embed:{mid}         String  numpy float32 bytes (embedding)

Embeddings are stored as raw numpy bytes rather than via RedisSearch/HNSW
so the project keeps a minimal footprint (plain Redis, not redis-stack).
Similarity is computed client-side; this scales fine to thousands of records.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any, Iterable, Optional

import numpy as np

try:
    import redis  # type: ignore
except ImportError:  # pragma: no cover -- redis is in the memory extra
    redis = None  # type: ignore

from gemma.config import Config
from gemma.memory.models import ConversationTurn, MemoryRecord


# -----------------------------------------------------------------------------
# Key helpers
# -----------------------------------------------------------------------------

def _k_memory(mid: str) -> str:
    return f"gemma:memory:{mid}"


def _k_memory_embed(mid: str) -> str:
    return f"gemma:memory:embed:{mid}"


_K_MEMORY_INDEX = "gemma:memory:index"

# Monotonic counter incremented on every save/supersede so MemoryRetriever
# can detect stale in-memory embedding caches without a full fetch.
_K_MEMORY_GENERATION = "gemma:memory:generation"


def _k_session_turns(sid: str) -> str:
    return f"gemma:session:{sid}:turns"


def _k_session_meta(sid: str) -> str:
    return f"gemma:session:{sid}:meta"


# -----------------------------------------------------------------------------
# MemoryStore
# -----------------------------------------------------------------------------

class MemoryStore:
    """Thin wrapper around redis.Redis tailored to the memory data model.

    The store is connection-aware: `available` returns False when Redis is
    unreachable, letting callers degrade gracefully instead of crashing.
    """

    def __init__(self, config: Config, client: Optional[Any] = None):
        self._config = config
        self._client = client  # injected for tests (fakeredis)
        self._health_checked_at: float = 0.0
        self._healthy: bool = False

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Attempt to establish a connection. Returns True on success."""
        if self._client is None:
            if redis is None:
                self._healthy = False
                return False
            try:
                self._client = redis.Redis.from_url(
                    self._config.redis_url,
                    decode_responses=True,  # hashes return str, not bytes
                )
            except Exception:
                self._healthy = False
                return False
        self._healthy = self._ping()
        self._health_checked_at = time.time()
        return self._healthy

    def _ping(self) -> bool:
        if self._client is None:
            return False
        try:
            return bool(self._client.ping())
        except Exception:
            return False

    @property
    def available(self) -> bool:
        """Cached health status (5-second window) to avoid per-call PING spam."""
        now = time.time()
        if now - self._health_checked_at > 5.0:
            self._healthy = self._ping()
            self._health_checked_at = now
        return self._healthy

    @property
    def client(self):
        """Raw Redis client (assumes connect() succeeded). For tests/debug."""
        return self._client

    # ------------------------------------------------------------------
    # Memory CRUD
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Generation counter (task 5.2 — lazy embedding cache invalidation)
    # ------------------------------------------------------------------

    def _bump_generation(self) -> None:
        """Atomically increment the global embedding-generation counter.

        Called on every save_memory and supersede_memory so that
        MemoryRetriever can detect stale in-memory embedding caches by
        comparing its stored generation against this counter.
        """
        if self._client is None:
            return
        try:
            self._client.incr(_K_MEMORY_GENERATION)
        except Exception:
            pass

    def get_generation(self) -> Optional[int]:
        """Return the current embedding-generation counter value.

        Returns:
            The counter as an int (0 if the key does not exist yet), or
            None when Redis is unavailable.
        """
        if self._client is None:
            return None
        try:
            val = self._client.get(_K_MEMORY_GENERATION)
            return int(val) if val is not None else 0
        except Exception:
            return None

    def save_memory(self, record: MemoryRecord) -> None:
        """Persist a MemoryRecord: HSET fields, ZADD to index, apply TTL."""
        if self._client is None:
            raise RuntimeError("MemoryStore not connected")

        key = _k_memory(record.memory_id)
        pipe = self._client.pipeline()
        pipe.hset(key, mapping=record.to_redis_hash())

        # Only index active memories (not superseded) so retrieval skips stale ones
        if record.is_active():
            pipe.zadd(_K_MEMORY_INDEX, {record.memory_id: float(record.importance)})
        else:
            pipe.zrem(_K_MEMORY_INDEX, record.memory_id)

        ttl = record.ttl_seconds(self._config.ttl_map)
        if ttl is not None and ttl > 0:
            pipe.expire(key, ttl)
            pipe.expire(_k_memory_embed(record.memory_id), ttl)
        pipe.execute()
        # Invalidate retriever caches that may hold stale embeddings.
        self._bump_generation()

    def get_memory(
        self, memory_id: str, *, bump_access: bool = True
    ) -> Optional[MemoryRecord]:
        """Load a MemoryRecord. If bump_access, updates last_accessed + count."""
        if self._client is None:
            return None
        raw = self._client.hgetall(_k_memory(memory_id))
        if not raw:
            return None
        record = MemoryRecord.from_redis_hash(raw)

        if bump_access:
            record.last_accessed = time.time()
            record.access_count += 1
            self._client.hset(
                _k_memory(memory_id),
                mapping={
                    "last_accessed": f"{record.last_accessed:.6f}",
                    "access_count": str(record.access_count),
                },
            )
        return record

    def supersede_memory(self, old_id: str, new_id: str) -> None:
        """Mark `old_id` as replaced by `new_id` and drop it from the index."""
        if self._client is None:
            return
        pipe = self._client.pipeline()
        pipe.hset(_k_memory(old_id), "superseded_by", new_id)
        pipe.zrem(_K_MEMORY_INDEX, old_id)
        pipe.execute()
        # Invalidate retriever caches; the index changed.
        self._bump_generation()

    def get_all_active_memories(self) -> list[MemoryRecord]:
        """Return every MemoryRecord currently in the index. For reconsolidation."""
        if self._client is None:
            return []
        ids = self._client.zrange(_K_MEMORY_INDEX, 0, -1)
        records: list[MemoryRecord] = []
        for mid in ids:
            raw = self._client.hgetall(_k_memory(mid))
            if raw:
                records.append(MemoryRecord.from_redis_hash(raw))
        return records

    def get_top_memories(self, n: int = 20) -> list[MemoryRecord]:
        """Top-N active memories by importance (descending)."""
        if self._client is None:
            return []
        ids = self._client.zrevrange(_K_MEMORY_INDEX, 0, max(0, n - 1))
        out: list[MemoryRecord] = []
        for mid in ids:
            raw = self._client.hgetall(_k_memory(mid))
            if raw:
                out.append(MemoryRecord.from_redis_hash(raw))
        return out

    def count_active_memories(self) -> int:
        """ZCARD of the active index."""
        if self._client is None:
            return 0
        return int(self._client.zcard(_K_MEMORY_INDEX))

    # ------------------------------------------------------------------
    # Embedding CRUD
    # ------------------------------------------------------------------

    def save_embedding(self, memory_id: str, vector: np.ndarray) -> None:
        """Store a numpy float32 vector.

        Base64-encoded because the MemoryStore Redis client is configured
        with `decode_responses=True` (needed for hash operations), which
        mangles raw binary. Base64 round-trips cleanly through the string
        decoder and adds ~33% overhead -- negligible for 768-float vectors
        (~4KB raw -> ~5.5KB encoded).
        """
        if self._client is None:
            raise RuntimeError("MemoryStore not connected")
        arr = np.asarray(vector, dtype=np.float32)
        encoded = base64.b64encode(arr.tobytes()).decode("ascii")
        self._client.set(_k_memory_embed(memory_id), encoded)

    def get_embedding(self, memory_id: str) -> Optional[np.ndarray]:
        """Load a vector, reconstructing the numpy array from base64."""
        if self._client is None:
            return None
        raw = self._client.get(_k_memory_embed(memory_id))
        if not raw:
            return None
        try:
            decoded = base64.b64decode(raw)
        except Exception:
            return None
        return np.frombuffer(decoded, dtype=np.float32)

    def get_all_embeddings(self) -> dict[str, np.ndarray]:
        """Map of active memory_id -> embedding vector.

        Memories without a stored embedding are silently skipped -- they just
        won't be candidates for semantic retrieval.
        """
        if self._client is None:
            return {}
        ids = self._client.zrange(_K_MEMORY_INDEX, 0, -1)
        out: dict[str, np.ndarray] = {}
        for mid in ids:
            vec = self.get_embedding(mid)
            if vec is not None and vec.size > 0:
                out[mid] = vec
        return out

    # ------------------------------------------------------------------
    # Session sliding window
    # ------------------------------------------------------------------

    def push_turn(self, session_id: str, turn: ConversationTurn) -> None:
        """Append a turn to the session list (RPUSH)."""
        if self._client is None:
            return
        payload = json.dumps(
            {
                "role": turn.role,
                "content": turn.content,
                "turn_number": turn.turn_number,
                "timestamp": turn.timestamp,
            }
        )
        self._client.rpush(_k_session_turns(session_id), payload)
        self._client.hset(
            _k_session_meta(session_id),
            mapping={"last_active": f"{time.time():.6f}"},
        )

    def get_turn_count(self, session_id: str) -> int:
        if self._client is None:
            return 0
        return int(self._client.llen(_k_session_turns(session_id)))

    def get_recent_turns(self, session_id: str, n: int) -> list[ConversationTurn]:
        """Return the last N turns as ConversationTurn objects (chronological)."""
        if self._client is None or n <= 0:
            return []
        # LRANGE supports negative indices: -n to -1 yields the last N entries
        raw = self._client.lrange(_k_session_turns(session_id), -n, -1)
        return [self._decode_turn(item) for item in raw]

    def get_overflow_turns(
        self, session_id: str, window_size: int
    ) -> list[ConversationTurn]:
        """Pop the turns that fall outside the sliding window.

        If the session has more than `window_size` turns, the oldest excess
        turns are returned AND removed (LTRIM) so the window stays fixed.
        Returns [] if the session is within bounds.
        """
        if self._client is None:
            return []
        key = _k_session_turns(session_id)
        length = int(self._client.llen(key))
        excess = length - window_size
        if excess <= 0:
            return []
        # LRANGE 0 .. excess-1 are the overflow items
        raw = self._client.lrange(key, 0, excess - 1)
        # LTRIM keeps only [excess, -1], i.e., the most recent `window_size` turns
        self._client.ltrim(key, excess, -1)
        return [self._decode_turn(item) for item in raw]

    @staticmethod
    def _decode_turn(payload: str) -> ConversationTurn:
        data = json.loads(payload)
        return ConversationTurn(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            turn_number=int(data.get("turn_number", 0)),
            timestamp=float(data.get("timestamp", time.time())),
        )

    def clear_session(self, session_id: str) -> None:
        """Delete the raw turn list and metadata for a session."""
        if self._client is None:
            return
        self._client.delete(_k_session_turns(session_id), _k_session_meta(session_id))
