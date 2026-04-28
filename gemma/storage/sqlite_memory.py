"""SQLite-backed memory store — drop-in replacement for the Redis one.

Mirrors every public method of :class:`gemma.memory.store.MemoryStore`
so :mod:`gemma.memory.manager` and the CLI memory commands work
unchanged regardless of which backend is active. Notable design
choices:

* **Single file, two tables** — ``memories`` for metadata,
  ``memory_embeddings`` for vectors. ``ON DELETE CASCADE`` keeps them
  consistent when a memory is hard-deleted.
* **TTL via timestamp column** — SQLite has no native EXPIRE, so we
  compute ``expires_at = created_at + ttl`` at write time and run a
  cheap range-delete sweep on every read/write. Same observable
  behaviour as Redis EXPIRE without a background job.
* **Soft delete via ``superseded_by``** — kept for symmetry with the
  Redis path; queries filter ``WHERE superseded_by = ''`` to skip
  superseded rows.
* **Brute-force vector search** — vectors stored as raw float32 BLOBs
  (~3 KB / 768-dim row), loaded into a numpy matrix for cosine
  similarity. At ≤500 active memories this is sub-millisecond and
  needs no extension.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Optional

import numpy as np

from gemma.memory.models import ConversationTurn, MemoryRecord
from gemma.storage.sqlite_db import open_db, sweep_expired

if TYPE_CHECKING:
    from gemma.config import Config


class SQLiteMemoryStore:
    """SQLite implementation of the memory store contract.

    Public API is identical to :class:`gemma.memory.store.MemoryStore`
    so call sites don't change. The constructor differs only in that
    it ignores the ``client`` / ``pool`` kwargs (Redis-specific) when
    they're provided — keeps the factory signature clean.
    """

    def __init__(
        self,
        config: "Config",
        *,
        client=None,    # accepted, ignored — for factory parity
        pool=None,      # accepted, ignored — for factory parity
    ) -> None:
        _ = client, pool
        self._config = config
        self._conn = open_db(config)

    # ------------------------------------------------------------------
    # Connection lifecycle (mirrors RedisMemoryStore for drop-in use)
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Always succeeds: the file-backed DB is opened in __init__."""
        return True

    @property
    def available(self) -> bool:
        """True once the DB connection is open. SQLite has no ping."""
        return self._conn is not None

    @property
    def client(self):
        """Raw sqlite3 connection — only used by tests and ``storage info``."""
        return self._conn

    # ------------------------------------------------------------------
    # Generation counter (lazy embedding-cache invalidation)
    # ------------------------------------------------------------------

    def _bump_generation(self) -> None:
        """Atomically increment the embedding-generation counter.

        The MemoryRetriever reads this to decide whether its in-process
        embedding matrix is stale (vs Redis where INCR was atomic by
        nature). UPSERT keeps the row count at one.
        """
        self._conn.execute(
            """
            INSERT INTO counters(name, value) VALUES('memory_generation', 1)
            ON CONFLICT(name) DO UPDATE SET value = counters.value + 1
            """
        )
        self._conn.commit()

    def get_generation(self) -> Optional[int]:
        """Return the current counter value, or 0 if the row is absent."""
        row = self._conn.execute(
            "SELECT value FROM counters WHERE name = 'memory_generation'"
        ).fetchone()
        return int(row["value"]) if row else 0

    # ------------------------------------------------------------------
    # Memory CRUD
    # ------------------------------------------------------------------

    def save_memory(self, record: MemoryRecord) -> None:
        """Persist a memory record. Computes ``expires_at`` from the TTL map."""
        ttl = record.ttl_seconds(self._config.ttl_map)
        # ``expires_at = NULL`` means "never" — matches Redis's "no EXPIRE".
        expires_at: Optional[float]
        if ttl is None or ttl <= 0:
            expires_at = None
        else:
            expires_at = float(record.created_at) + float(ttl)

        self._conn.execute(
            """
            INSERT INTO memories(
              memory_id, content, category, importance,
              session_id, turn_range, source_summary,
              created_at, last_accessed, access_count,
              superseded_by, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
              content       = excluded.content,
              category      = excluded.category,
              importance    = excluded.importance,
              session_id    = excluded.session_id,
              turn_range    = excluded.turn_range,
              source_summary= excluded.source_summary,
              last_accessed = excluded.last_accessed,
              access_count  = excluded.access_count,
              superseded_by = excluded.superseded_by,
              expires_at    = excluded.expires_at
            """,
            (
                record.memory_id,
                record.content,
                record.category.value,
                int(record.importance),
                record.session_id or "",
                record.turn_range or "",
                getattr(record, "source_summary", "") or "",
                float(record.created_at),
                float(record.last_accessed),
                int(record.access_count),
                record.superseded_by or "",
                expires_at,
            ),
        )
        self._conn.commit()
        self._bump_generation()
        # Cheap sweep so expired rows don't accumulate between reads.
        sweep_expired(self._conn)

    def get_memory(
        self, memory_id: str, *, bump_access: bool = True
    ) -> Optional[MemoryRecord]:
        sweep_expired(self._conn)
        row = self._conn.execute(
            "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            return None
        record = self._row_to_record(row)
        if bump_access:
            now = time.time()
            record.last_accessed = now
            record.access_count += 1
            self._conn.execute(
                """
                UPDATE memories
                   SET last_accessed = ?, access_count = ?
                 WHERE memory_id = ?
                """,
                (now, record.access_count, memory_id),
            )
            self._conn.commit()
        return record

    def supersede_memory(self, old_id: str, new_id: str) -> None:
        """Mark ``old_id`` as replaced by ``new_id``; drops it from queries."""
        self._conn.execute(
            "UPDATE memories SET superseded_by = ? WHERE memory_id = ?",
            (new_id, old_id),
        )
        self._conn.commit()
        self._bump_generation()

    def get_all_active_memories(self) -> list[MemoryRecord]:
        sweep_expired(self._conn)
        rows = self._conn.execute(
            """
            SELECT * FROM memories
             WHERE superseded_by = ''
             ORDER BY importance DESC, last_accessed DESC
            """
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get_top_memories(self, n: int = 20) -> list[MemoryRecord]:
        sweep_expired(self._conn)
        rows = self._conn.execute(
            """
            SELECT * FROM memories
             WHERE superseded_by = ''
             ORDER BY importance DESC, last_accessed DESC
             LIMIT ?
            """,
            (max(0, int(n)),),
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def count_active_memories(self) -> int:
        sweep_expired(self._conn)
        row = self._conn.execute(
            "SELECT COUNT(*) AS c FROM memories WHERE superseded_by = ''"
        ).fetchone()
        return int(row["c"]) if row else 0

    # ------------------------------------------------------------------
    # Embedding CRUD
    # ------------------------------------------------------------------

    def save_embedding(self, memory_id: str, vector: np.ndarray) -> None:
        """Persist a numpy float32 vector as a raw BLOB."""
        arr = np.asarray(vector, dtype=np.float32)
        self._conn.execute(
            """
            INSERT INTO memory_embeddings(memory_id, vector, dim)
            VALUES (?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
              vector = excluded.vector,
              dim    = excluded.dim
            """,
            (memory_id, arr.tobytes(), int(arr.size)),
        )
        self._conn.commit()

    def get_embedding(self, memory_id: str) -> Optional[np.ndarray]:
        row = self._conn.execute(
            "SELECT vector FROM memory_embeddings WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row["vector"], dtype=np.float32)

    def get_all_embeddings(self) -> dict[str, np.ndarray]:
        """Return ``{memory_id: vector}`` for every active memory.

        One JOIN against the active-memory predicate keeps the round-trip
        count to one regardless of memory count.
        """
        sweep_expired(self._conn)
        rows = self._conn.execute(
            """
            SELECT m.memory_id AS mid, e.vector AS vec
              FROM memories m
              JOIN memory_embeddings e ON e.memory_id = m.memory_id
             WHERE m.superseded_by = ''
            """
        ).fetchall()
        out: dict[str, np.ndarray] = {}
        for r in rows:
            vec = np.frombuffer(r["vec"], dtype=np.float32)
            if vec.size > 0:
                out[r["mid"]] = vec
        return out

    # ------------------------------------------------------------------
    # Session sliding window
    # ------------------------------------------------------------------

    def push_turn(self, session_id: str, turn: ConversationTurn) -> None:
        self._conn.execute(
            """
            INSERT INTO session_turns(session_id, turn_number, role, content, timestamp)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(session_id, turn_number) DO UPDATE SET
              role = excluded.role,
              content = excluded.content,
              timestamp = excluded.timestamp
            """,
            (
                session_id,
                int(turn.turn_number),
                turn.role,
                turn.content,
                float(turn.timestamp),
            ),
        )
        self._conn.execute(
            """
            INSERT INTO session_meta(session_id, last_active) VALUES (?, ?)
            ON CONFLICT(session_id) DO UPDATE SET last_active = excluded.last_active
            """,
            (session_id, time.time()),
        )
        self._conn.commit()

    def get_turn_count(self, session_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS c FROM session_turns WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row["c"]) if row else 0

    def get_recent_turns(self, session_id: str, n: int) -> list[ConversationTurn]:
        if n <= 0:
            return []
        # Pull in descending order then reverse: cheaper than a window function
        # and still O(log + n).
        rows = self._conn.execute(
            """
            SELECT role, content, turn_number, timestamp
              FROM session_turns
             WHERE session_id = ?
             ORDER BY turn_number DESC
             LIMIT ?
            """,
            (session_id, int(n)),
        ).fetchall()
        rows = list(reversed(rows))
        return [self._row_to_turn(r) for r in rows]

    def get_overflow_turns(
        self, session_id: str, window_size: int
    ) -> list[ConversationTurn]:
        """Return + delete the turns older than the most recent ``window_size``.

        Mirrors Redis LTRIM semantics: the call both *returns* the
        excess turns (so condensation can extract memories from them)
        and *removes* them (so the window stays bounded).
        """
        total = self.get_turn_count(session_id)
        excess = total - int(window_size)
        if excess <= 0:
            return []
        # Identify the oldest ``excess`` turn_numbers in one round-trip.
        rows = self._conn.execute(
            """
            SELECT role, content, turn_number, timestamp
              FROM session_turns
             WHERE session_id = ?
             ORDER BY turn_number ASC
             LIMIT ?
            """,
            (session_id, excess),
        ).fetchall()
        if not rows:
            return []
        nums = [int(r["turn_number"]) for r in rows]
        placeholders = ",".join("?" * len(nums))
        self._conn.execute(
            f"""
            DELETE FROM session_turns
             WHERE session_id = ?
               AND turn_number IN ({placeholders})
            """,
            (session_id, *nums),
        )
        self._conn.commit()
        return [self._row_to_turn(r) for r in rows]

    def clear_session(self, session_id: str) -> None:
        self._conn.execute(
            "DELETE FROM session_turns WHERE session_id = ?", (session_id,)
        )
        self._conn.execute(
            "DELETE FROM session_meta WHERE session_id = ?", (session_id,)
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Row → domain model adapters
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_record(row) -> MemoryRecord:
        """Reuse ``MemoryRecord.from_redis_hash`` for serialization parity.

        The field set is identical between Redis hashes and our
        ``memories`` table, so we marshal one into the other and lean
        on the existing parser. Keeps a single source of truth for
        type coercion.
        """
        hash_form = {
            "memory_id": row["memory_id"],
            "content": row["content"],
            "category": row["category"],
            "importance": str(int(row["importance"])),
            "session_id": row["session_id"],
            "turn_range": row["turn_range"],
            "source_summary": row["source_summary"],
            "created_at": f"{float(row['created_at']):.6f}",
            "last_accessed": f"{float(row['last_accessed']):.6f}",
            "access_count": str(int(row["access_count"])),
            "superseded_by": row["superseded_by"] or "",
        }
        return MemoryRecord.from_redis_hash(hash_form)

    @staticmethod
    def _row_to_turn(row) -> ConversationTurn:
        return ConversationTurn(
            role=row["role"],
            content=row["content"],
            turn_number=int(row["turn_number"]),
            timestamp=float(row["timestamp"]),
        )

    # ------------------------------------------------------------------
    # House-keeping
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection — for tests / shutdown."""
        try:
            self._conn.close()
        except Exception:
            pass
