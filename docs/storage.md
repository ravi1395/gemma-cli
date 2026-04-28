# Storage backends

gemma-cli persists three things between sessions:

1. **Memories** — structured facts the condensation pipeline extracts from your conversations, plus their embeddings, plus the per-session sliding window of recent turns.
2. **RAG vectors** — chunked workspace files plus their embeddings, partitioned by namespace (`{repo_hash}:{branch}`).
3. **Caches** — content-hash → embedding (so re-indexing skips work) and SHA-keyed prompt → response (so identical low-temperature calls return instantly).

Two backends implement these:

| Backend | When to pick | Where data lives |
|---|---|---|
| **`sqlite`** (default) | New installs. Single-user. Backup = one `cp`. | `~/.gemma/store.sqlite` |
| **`redis`** (legacy) | You already have a populated Redis store, or you share state across machines. | Whatever your `redis_url` points at. |

Switch backends with:

| Scope | How |
|---|---|
| Persistent | `storage_backend = "redis"` in `~/.config/gemma/profiles/<name>.toml` |
| Per-call | (no top-level flag yet — set via profile) |
| Process-wide | `Config(storage_backend="redis")` in any embedding code |

`gemma storage info` prints the resolved backend, file path / Redis URL, and row counts.

---

## SQLite layout

One file at `~/.gemma/store.sqlite` holds every concern in normalised tables. Vectors are raw `float32` BLOBs (`np.frombuffer(blob, dtype=np.float32)` round-trips them).

### Tables

| Table | Purpose | Key fields |
|---|---|---|
| `memories` | One row per condensed memory. | `memory_id` PK, `importance`, `expires_at` |
| `memory_embeddings` | Vector for each memory. | `memory_id` PK FK → `memories` (CASCADE) |
| `session_turns` | Sliding window of raw turns. | `(session_id, turn_number)` PK |
| `session_meta` | Per-session metadata (last_active). | `session_id` PK |
| `counters` | Monotonic counters (used for embedding-cache invalidation). | `name` PK |
| `rag_chunks` | One row per indexed chunk. | `(namespace, chunk_id)` PK |
| `rag_embeddings` | Vector for each chunk. | `(namespace, chunk_id)` PK FK → `rag_chunks` (CASCADE) |
| `rag_manifest` | `path → blob_sha` for incremental indexing. | `(namespace, path)` PK |
| `rag_meta` | Per-namespace meta (model, dim). | `(namespace, key)` PK |
| `embed_cache` | Content-hash → vector cache. | `(model, content_hash)` PK |
| `response_cache` | SHA-keyed prompt → response. | `cache_key` PK |

### Pragmas

Set on every connection by [`gemma/storage/sqlite_db.py`](../gemma/storage/sqlite_db.py):

| Pragma | Value | Why |
|---|---|---|
| `journal_mode` | `WAL` | Multiple readers + single writer. Essential for the warm-start daemon thread. |
| `synchronous` | `NORMAL` | Fsync on WAL append; skip the per-checkpoint fsync. Right durability/perf trade for a single-user CLI. |
| `foreign_keys` | `ON` | Enables `ON DELETE CASCADE` on the embedding tables — deleting a memory drops its vector. |
| `temp_store` | `MEMORY` | Keep temp B-trees in RAM. |

### TTL handling

SQLite has no `EXPIRE`. We use an `expires_at REAL` column and run a sweep on every read/write hot path:

```python
DELETE FROM memories     WHERE expires_at IS NOT NULL AND expires_at < NOW;
DELETE FROM embed_cache  WHERE expires_at < NOW;
DELETE FROM response_cache WHERE expires_at < NOW;
```

Each `expires_at` column is indexed so the sweep is one quick range delete per table. No background thread.

The TTL map for memories is importance-tiered (see [`gemma/config.py`](../gemma/config.py)):

| Importance | TTL |
|---|---|
| 5 (critical) | never expires |
| 4 (high) | 7 days |
| 3 (medium) | 3 days |
| 2 (low) | 1 day |
| 1 (trivial) | 6 hours |

### File layout

```
~/.gemma/
└── store.sqlite          (the main database)
└── store.sqlite-wal      (write-ahead log; rolls forward on close)
└── store.sqlite-shm      (shared-memory index for WAL readers)
```

`store.sqlite-wal` and `store.sqlite-shm` are sidecars. They're harmless but not optional — don't delete them while gemma is running. To back up cleanly, run `gemma storage info` first (forces a checkpoint), then `cp store.sqlite backup.sqlite`.

---

## Redis layout (legacy)

Documented in detail in the docstring at the top of [`gemma/memory/store.py`](../gemma/memory/store.py) and [`gemma/rag/store.py`](../gemma/rag/store.py). Summary:

```
gemma:memory:{mid}                 HASH   memory record fields
gemma:memory:embed:{mid}           STR    base64 float32
gemma:memory:index                 ZSET   {mid: importance_score}
gemma:memory:generation            STR    monotonic counter
gemma:session:{sid}:turns          LIST   JSON-encoded ConversationTurn
gemma:session:{sid}:meta           HASH   last_active

gemma:rag:{ns}:index               SET    chunk_ids for the namespace
gemma:rag:{ns}:chunk:{cid}         HASH   chunk metadata
gemma:rag:{ns}:embed:{cid}         STR    raw float32 bytes
gemma:rag:{ns}:manifest            HASH   path → blob_sha
gemma:rag:{ns}:meta                HASH   dim, model, last_indexed_at
gemma:rag:{ns}:file_chunks:{path}  SET    reverse index for delete-by-file

gemma:rag:embed_cache:v1:{model}:{sha} STR float32 (TTL via EXPIRE)
gemma:cache:{sha256}                JSON  response cache (TTL via EXPIRE)
```

---

## Performance

Brute-force cosine similarity at our scale (768-dim float32, single-machine):

| Vector count | Search latency |
|---|---|
| 500 (memory cap) | ~0.3 ms |
| 5,000 (medium RAG index) | ~3 ms |
| 50,000 | ~30 ms |

Search is `numpy` `matrix @ query` after a single SELECT loads candidates into RAM. The query and stored vectors are L2-normalised at write time so cosine reduces to a dot product — no per-call normalisation in the hot loop.

If your RAG index ever grows past ~50k chunks the brute-force step starts to dominate. Two upgrade paths, both swappable behind the same factory:

1. Add a [`sqlite-vec`](https://github.com/asg017/sqlite-vec) virtual table and let SQLite's optimised k-NN do the work.
2. Adopt a sidecar HNSW index ([`usearch`](https://github.com/unum-cloud/usearch)).

Both are deferrable — neither is needed for the workloads gemma-cli targets today.

---

## Migration

```bash
# Default: redis → sqlite. Idempotent; safe to re-run.
gemma storage migrate

# Reverse direction (re-publish to a Redis backup).
gemma storage migrate --from sqlite --to redis

# Smoke-test without touching the destination.
gemma storage migrate --dry-run
```

`gemma storage migrate` walks every memory, embedding, RAG chunk, manifest entry, and meta row from the source and upserts into the destination. Primary keys are stable across backends, so partial runs resume cleanly. See [`migrating-from-redis.md`](migrating-from-redis.md) for the full upgrade walkthrough.

---

## Useful inspection queries

`sqlite3 ~/.gemma/store.sqlite` is plain SQLite — every standard tool works:

```sql
-- 10 most-accessed memories
SELECT content, access_count, importance
  FROM memories
 WHERE superseded_by = ''
 ORDER BY access_count DESC
 LIMIT 10;

-- Disk usage by table (rough)
SELECT name, SUM(pgsize) AS bytes
  FROM dbstat
 GROUP BY name
 ORDER BY bytes DESC;

-- Active namespaces in the RAG index
SELECT namespace, COUNT(*) AS chunks
  FROM rag_chunks
 GROUP BY namespace
 ORDER BY chunks DESC;
```
