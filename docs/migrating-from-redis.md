# Migrating from Redis to SQLite

gemma-cli 0.3 makes **SQLite the default** storage backend. Memories, RAG vectors, and the response cache all live in a single file at `~/.gemma/store.sqlite`. Redis remains supported as an opt-in backend for users who already have a populated store.

This document walks through the upgrade path.

---

## TL;DR

```bash
git pull
uv sync                                  # picks up the new SQLite path automatically

# Copy your existing Redis state into the SQLite file
gemma storage migrate --from redis --to sqlite

# Confirm
gemma storage info                       # shows row counts in the new file

# (Optional) shut down Redis if you only used it for gemma
brew services stop redis                 # macOS
sudo systemctl stop redis                # Linux
docker compose down redis                # Docker
```

After the migration, every existing CLI command (`gemma ask`, `gemma chat`, `gemma rag query`, `gemma history memories`, etc.) reads from SQLite without you doing anything else.

---

## What changed and why

| Before (Redis-only) | After (SQLite-default) |
|---|---|
| Required a running Redis server (`redis-server`, Docker, or Memurai on Windows) | Single file; no server. SQLite is in the Python stdlib. |
| `decode_responses=True` plus a separate binary client for vectors | Vectors are raw `BLOB` columns; one connection per process. |
| Native `EXPIRE` for TTLs | `expires_at` column + sweep-on-touch (same observable behaviour). |
| `gemma:rag:embed_cache:v1:...` keys | `embed_cache` table with `(model, content_hash)` primary key. |
| Stale memories evicted automatically by Redis | Stale memories deleted on every read/write hot path. |
| Backup = `BGSAVE` + copy `dump.rdb` | Backup = `cp ~/.gemma/store.sqlite ~/backup.sqlite`. |

The migration is **opt-out, not opt-in**. New installs get SQLite for free; existing users with a running Redis can either migrate (recommended) or set `storage_backend = "redis"` to keep their current setup.

---

## Step 1 — Pull and sync

```bash
cd ~/path/to/gemma-cli
git pull                                # or `git checkout v0.3.x`
uv sync                                 # or: pip install -e ".[memory]"
```

`Config.storage_backend` defaults to `"sqlite"` and `Config.sqlite_path` defaults to `~/.gemma/store.sqlite`. You don't need to set anything for the new default to take effect.

**Sanity check:**

```bash
gemma storage info
```

You'll see a fresh empty SQLite file:

```
backend:        sqlite
file:           /Users/you/.gemma/store.sqlite
size:           112.0 KB
memories          0
session_turns     0
…
```

---

## Step 2 — Migrate your Redis state

```bash
# Defaults: --from redis --to sqlite
gemma storage migrate

# Or be explicit:
gemma storage migrate --from redis --to sqlite

# Check the proposed transfer first without writing:
gemma storage migrate --dry-run
```

What gets copied:

| Source key set (Redis) | Destination table (SQLite) |
|---|---|
| `gemma:memory:{mid}` HASH | `memories` |
| `gemma:memory:embed:{mid}` STR | `memory_embeddings` |
| `gemma:memory:index` ZSET (membership only) | implicit — recomputed from `superseded_by` |
| `gemma:rag:{ns}:chunk:{cid}` HASH | `rag_chunks` |
| `gemma:rag:{ns}:embed:{cid}` STR | `rag_embeddings` |
| `gemma:rag:{ns}:manifest` HASH | `rag_manifest` |
| `gemma:rag:{ns}:meta` HASH | `rag_meta` |

The migration is **idempotent** — primary keys are stable, so re-running upserts in place. A partial run (e.g. terminated by Ctrl+C) resumes cleanly when you re-invoke.

What does **not** migrate automatically:

- **Embed cache** (`gemma:rag:embed_cache:v1:...`) — these are content-hash keyed; the SQLite path will rebuild them on the next index call. Skip the manual copy unless you have a cold cache and a slow embedder.
- **Response cache** (`gemma:cache:...`) — same logic. Identical low-temp prompts will get a fresh answer the first time and a cache hit thereafter.

If you want both: add `--include-caches` (planned; not in v0.3.0). For now they self-heal on first use.

---

## Step 3 — Verify

```bash
gemma storage info
```

You should see your memory + chunk counts populated:

```
backend:        sqlite
file:           /Users/you/.gemma/store.sqlite
size:           14.8 MB
memories          47
memory_embeddings 47
session_turns     12
rag_chunks        2,341
rag_embeddings    2,341
rag_manifest      189
…
```

Smoke-test a query that you know should hit memory:

```bash
gemma ask "what's my preferred Python version?"
```

If a memory was successfully migrated, the model should answer based on it. If not, run `gemma history memories` to confirm the records are present.

---

## Step 4 — (Optional) shut down Redis

If you were only running Redis for gemma-cli, you can stop it:

| Platform | Command |
|---|---|
| macOS Homebrew | `brew services stop redis` |
| Linux systemd | `sudo systemctl stop redis-server` |
| Docker Compose | `docker compose down redis` (or `docker stop <container>`) |
| Windows / Memurai | Services panel → Memurai → Stop |

You can also uninstall Redis entirely. The SQLite backend has zero runtime dependencies beyond the `sqlite3` module that ships with Python.

---

## Keeping Redis (if you prefer)

If you have a workflow that depends on Redis (e.g. shared state across machines, a separate observability stack on Redis), you can keep it:

```toml
# ~/.config/gemma/profiles/default.toml
storage_backend = "redis"
redis_url = "redis://localhost:6379/0"
```

Or per-call:

```bash
# ``--storage-backend`` is not yet a top-level flag; use a profile.
gemma --profile redis-only ask "..."
```

Both backends share the same `MemoryStore`/`VectorStore` contract — feature parity is by design.

---

## Behavioural differences worth knowing

### TTL semantics

- **Redis**: per-key `EXPIRE`; the server evicts in the background.
- **SQLite**: `expires_at REAL` column; rows past expiry are deleted on every read/write hot path.

Net effect for the user: identical. A memory pinned at importance 5 never expires in either backend. A trivial memory at importance 1 disappears after 6 hours in either backend.

### Connection model

- **Redis**: TCP/IP to a server process. Latency is dominated by the round-trip.
- **SQLite**: in-process file I/O. Latency is dominated by deserialization.

For our workloads (≤500 memories, ≤50k RAG chunks) both are sub-millisecond per call.

### Concurrency

- **Redis**: many clients, full concurrency.
- **SQLite**: WAL mode allows multiple readers + one writer. The warm-start daemon thread reads concurrently with the foreground writer fine; heavy concurrent writes (>1 process) would serialise.

For a single-user CLI this is a non-issue. If you ever invoke `gemma` from multiple terminals at once, the second writer waits briefly — not a correctness problem.

### Backup / restore

- **Redis**: `BGSAVE` + ship `dump.rdb` somewhere safe.
- **SQLite**: `cp ~/.gemma/store.sqlite ~/backup.sqlite`.

The SQLite backup is simpler and works while gemma is running (WAL mode handles the read-during-write case).

---

## Rollback

If something doesn't work and you want to fall back to the pre-0.3 behaviour wholesale:

```bash
git checkout v0.2.x        # or whatever tag predates the storage migration
uv sync
```

Your Redis state is untouched by `gemma storage migrate` — the migrate command only ever inserts into the destination, never deletes from the source. So a rollback is purely a code switch; no data restore needed.

If the failure is reproducible, please open an issue with:

- `gemma --version`
- The output of `gemma storage info`
- The traceback (`uv run pytest -x` first to confirm whether it's a test-suite-level failure)

— the smaller the migration friction is for the next person, the better.
