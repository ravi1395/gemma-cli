"""SQLite-backed response cache.

Mirrors the ``get``/``put`` surface of :class:`gemma.cache.ResponseCache`
so the cache-eligibility helper in ``gemma.cache.eligible`` can hand
back either backend interchangeably. Keys are computed by the existing
``ResponseCache._compute_key`` (SHA over model / temperature / system /
user / keep_alive) so a Redis-cached entry and a SQLite-cached entry
hash identically — the migrate command can copy one into the other
without any rehashing.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Optional

from gemma.cache import ResponseCache as _RedisResponseCache
from gemma.storage.sqlite_db import open_db, sweep_expired

if TYPE_CHECKING:
    from gemma.config import Config


class SQLiteResponseCache:
    """Persistent prompt → response cache stored in the gemma SQLite file."""

    def __init__(self, config: "Config") -> None:
        self._config = config
        self._ttl = int(getattr(config, "cache_ttl_seconds", 0) or 0)
        self._conn = open_db(config)

    # ------------------------------------------------------------------
    # Public API (matches :class:`gemma.cache.ResponseCache`)
    # ------------------------------------------------------------------

    def get(self, messages: list[dict], config: "Config") -> Optional[str]:
        if self._ttl <= 0:
            return None
        try:
            key = _RedisResponseCache._compute_key(messages, config)
        except Exception:
            return None
        sweep_expired(self._conn)
        row = self._conn.execute(
            """
            SELECT response FROM response_cache
             WHERE cache_key = ? AND expires_at > ?
            """,
            (key, time.time()),
        ).fetchone()
        if row is None:
            return None
        # Stored payload is plain text; the wrapping JSON the Redis path
        # used was an artefact of needing a single-string Redis VALUE, not
        # a contract. Tolerate both shapes for in-place migration.
        raw = row["response"]
        if raw and raw.startswith("{") and raw.endswith("}"):
            try:
                return json.loads(raw).get("content", raw)
            except Exception:
                return raw
        return raw

    def put(self, messages: list[dict], config: "Config", content: str) -> None:
        if self._ttl <= 0:
            return
        try:
            key = _RedisResponseCache._compute_key(messages, config)
        except Exception:
            return
        now = time.time()
        self._conn.execute(
            """
            INSERT INTO response_cache(cache_key, response, created_at, expires_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
              response = excluded.response,
              created_at = excluded.created_at,
              expires_at = excluded.expires_at
            """,
            (key, content, now, now + self._ttl),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # House-keeping
    # ------------------------------------------------------------------

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
