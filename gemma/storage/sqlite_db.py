"""Shared SQLite connection + schema bootstrap for gemma-cli storage.

Every SQLite-backed store (``SQLiteMemoryStore``, ``SQLiteRAGStore``,
``SQLiteResponseCache``) gets its connection from :func:`open_db`.
The module owns:

* **Schema creation** — idempotent ``CREATE TABLE IF NOT EXISTS``
  against the file at ``Config.sqlite_path`` (default
  ``~/.gemma/store.sqlite``).
* **Pragmas** — WAL journal mode for concurrent reads, ``foreign_keys
  = ON`` so cascade deletes work on the embedding tables, ``synchronous
  = NORMAL`` for the right durability/perf trade-off on a personal CLI.
* **TTL sweeps** — :func:`sweep_expired` deletes rows whose
  ``expires_at`` (unix timestamp) is past. Called from each store's
  hot-paths so we don't need a background thread.

The schema is laid out as one single file with multiple tables rather
than one file per concern — backups and ``cp`` semantics are simpler
with a single file, and SQLite handles tens of tables in one DB
without breaking a sweat.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemma.config import Config


# ---------------------------------------------------------------------------
# DDL — every CREATE is IF NOT EXISTS so calling open_db twice is a no-op.
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
-- Memory records (the structured facts gemma extracts from your sessions).
CREATE TABLE IF NOT EXISTS memories (
  memory_id     TEXT PRIMARY KEY,
  content       TEXT NOT NULL,
  category      TEXT NOT NULL,
  importance    INTEGER NOT NULL,
  session_id    TEXT NOT NULL DEFAULT '',
  turn_range    TEXT NOT NULL DEFAULT '',
  source_summary TEXT NOT NULL DEFAULT '',
  created_at    REAL NOT NULL,
  last_accessed REAL NOT NULL,
  access_count  INTEGER NOT NULL DEFAULT 0,
  superseded_by TEXT NOT NULL DEFAULT '',
  expires_at    REAL  -- NULL = never; sweep deletes when NOW > expires_at
);
CREATE INDEX IF NOT EXISTS idx_memories_active_importance
  ON memories(superseded_by, importance);
CREATE INDEX IF NOT EXISTS idx_memories_expires
  ON memories(expires_at);

-- Memory embeddings stored as raw float32 BLOBs (12 KB / 768-dim row).
-- ON DELETE CASCADE keeps the two tables in sync without app-level code.
CREATE TABLE IF NOT EXISTS memory_embeddings (
  memory_id TEXT PRIMARY KEY
    REFERENCES memories(memory_id) ON DELETE CASCADE,
  vector    BLOB NOT NULL,
  dim       INTEGER NOT NULL
);

-- Per-session sliding-window of raw turns (the recent N visible to the model).
CREATE TABLE IF NOT EXISTS session_turns (
  session_id  TEXT NOT NULL,
  turn_number INTEGER NOT NULL,
  role        TEXT NOT NULL,
  content     TEXT NOT NULL,
  timestamp   REAL NOT NULL,
  PRIMARY KEY (session_id, turn_number)
);

-- Per-session metadata; right now just last_active for cleanup heuristics.
CREATE TABLE IF NOT EXISTS session_meta (
  session_id  TEXT PRIMARY KEY,
  last_active REAL NOT NULL
);

-- Monotonic counters (lazy embedding-cache invalidation, future use).
CREATE TABLE IF NOT EXISTS counters (
  name  TEXT PRIMARY KEY,
  value INTEGER NOT NULL
);

-- RAG chunks scoped by ``namespace`` so multiple workspaces share the file.
CREATE TABLE IF NOT EXISTS rag_chunks (
  namespace  TEXT NOT NULL,
  chunk_id   TEXT NOT NULL,
  path       TEXT NOT NULL,
  start_line INTEGER NOT NULL,
  end_line   INTEGER NOT NULL,
  text       TEXT NOT NULL,
  header     TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (namespace, chunk_id)
);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_path
  ON rag_chunks(namespace, path);

CREATE TABLE IF NOT EXISTS rag_embeddings (
  namespace TEXT NOT NULL,
  chunk_id  TEXT NOT NULL,
  vector    BLOB NOT NULL,
  dim       INTEGER NOT NULL,
  PRIMARY KEY (namespace, chunk_id),
  FOREIGN KEY (namespace, chunk_id)
    REFERENCES rag_chunks(namespace, chunk_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS rag_manifest (
  namespace TEXT NOT NULL,
  path      TEXT NOT NULL,
  blob_sha  TEXT NOT NULL,
  PRIMARY KEY (namespace, path)
);

CREATE TABLE IF NOT EXISTS rag_meta (
  namespace TEXT NOT NULL,
  key       TEXT NOT NULL,
  value     TEXT NOT NULL,
  PRIMARY KEY (namespace, key)
);

-- Embed-vector cache (content-hash keyed; survives indexing across branches).
CREATE TABLE IF NOT EXISTS embed_cache (
  model        TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  vector       BLOB NOT NULL,
  dim          INTEGER NOT NULL,
  expires_at   REAL NOT NULL,
  PRIMARY KEY (model, content_hash)
);
CREATE INDEX IF NOT EXISTS idx_embed_cache_expires
  ON embed_cache(expires_at);

-- Response cache (SHA-keyed prompt → response).
CREATE TABLE IF NOT EXISTS response_cache (
  cache_key  TEXT PRIMARY KEY,
  response   TEXT NOT NULL,
  created_at REAL NOT NULL,
  expires_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_response_cache_expires
  ON response_cache(expires_at);
"""


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _resolve_path(config: "Config") -> Path:
    """Expand ``Config.sqlite_path`` to an absolute path, creating the dir."""
    raw = getattr(config, "sqlite_path", None) or "~/.gemma/store.sqlite"
    path = Path(raw).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def open_db(config: "Config") -> sqlite3.Connection:
    """Open (and bootstrap on first use) the gemma-cli SQLite database.

    Pragmas
    -------
    * ``journal_mode = WAL`` — multiple readers + single writer; the
      write-ahead log is what makes SQLite usable as a concurrent
      store at our scale.
    * ``synchronous = NORMAL`` — fsync on every commit's WAL append
      but skip the per-checkpoint fsync. Adequate durability for a
      single-user CLI; bumps insert throughput substantially.
    * ``foreign_keys = ON`` — enforce ``ON DELETE CASCADE`` on the
      embedding tables so deleting a memory drops its vector.
    * ``temp_store = MEMORY`` — keep temp B-trees in RAM; harmless on
      a desktop and trims a few µs off complex queries.
    """
    conn = sqlite3.connect(
        _resolve_path(config),
        # Allow other threads to use the connection (we hand it to
        # daemon threads from the warm-start path).
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA temp_store = MEMORY;")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# TTL sweep
# ---------------------------------------------------------------------------

def sweep_expired(conn: sqlite3.Connection, *, now: float | None = None) -> int:
    """Delete rows past their ``expires_at`` across every table that has one.

    Idempotent and cheap — every ``expires_at`` column has an index so
    the sweep is one quick range delete per table. Returns the total
    number of rows removed (useful for ``gemma storage info``).
    """
    n = float(now if now is not None else time.time())
    cur = conn.cursor()
    deleted = 0
    deleted += cur.execute(
        "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at < ?",
        (n,),
    ).rowcount or 0
    deleted += cur.execute(
        "DELETE FROM embed_cache WHERE expires_at < ?", (n,)
    ).rowcount or 0
    deleted += cur.execute(
        "DELETE FROM response_cache WHERE expires_at < ?", (n,)
    ).rowcount or 0
    conn.commit()
    return deleted
