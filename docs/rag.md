# RAG implementation deep-dive

gemma-cli ships local retrieval-augmented generation: index your workspace once, then ask questions and have the model answer with file:line citations from your own code/docs. This document explains how the pipeline works, end-to-end, so you can debug it, tune it, or swap pieces out.

```
                                                          ┌──────────────────┐
gemma rag index            ┌────────────────┐             │  gemma rag query │
        │                  │  RAGIndexer    │             │                  │
        ▼                  └───────┬────────┘             ▼
  walk workspace                   │                ┌────────────────┐
  ↓                                ▼                │  RAGRetriever  │
  filter (extensions,        embed (LM Studio /     └────────┬───────┘
  size, denylist)            Ollama; cache hits             │
  ↓                          short-circuit)                  ▼
  Chunk(text, lines)               │                  embed query
  ↓                                ▼                       │
  manifest diff                upsert to                   ▼
  (mtime+size,sha)             VectorStore           cosine top-N
                                   │                       │
                                   ▼                       ▼
                          ┌──────────────────┐         MMR diversify
                          │  VectorStore     │         (k results)
                          │  (sqlite/redis)  │             │
                          └──────────────────┘             ▼
                                                     RetrievalHit[ ]
```

---

## Modules at a glance

| Module | Role |
|---|---|
| [`gemma/rag/namespace.py`](../gemma/rag/namespace.py) | Compute the per-workspace partition key: `sha256(repo_root):branch`. |
| [`gemma/chunking.py`](../gemma/chunking.py) | Split text into overlap-aware token windows. |
| [`gemma/rag/manifest.py`](../gemma/rag/manifest.py) | Walk + filter files; compute mtime+size+sha for incremental indexing. |
| [`gemma/rag/indexer.py`](../gemma/rag/indexer.py) | The pipeline driver. Wires everything; called by `gemma rag index`. |
| [`gemma/rag/store.py`](../gemma/rag/store.py) | Redis-backed vector store (legacy). |
| [`gemma/storage/sqlite_rag.py`](../gemma/storage/sqlite_rag.py) | SQLite-backed vector store (default). |
| [`gemma/rag/retrieval.py`](../gemma/rag/retrieval.py) | Query-time path: embed query → cosine top-N → MMR re-rank. |
| [`gemma/commands/rag.py`](../gemma/commands/rag.py) | The Typer subcommands (`index`, `query`, `status`, `reset`, `cache stats`, `cache clear`). |

---

## Namespace scoping

Every RAG read/write is scoped to a **namespace** so two repos (or two branches of one repo) don't share state. `resolve_namespace(root, branch)` returns:

```
sha256(canonical_repo_root)[:16] : sanitized_branch_name
```

Examples:

| Workspace | Branch | Namespace |
|---|---|---|
| `~/work/api` | `main` | `a1b2c3d4e5f6abcd:main` |
| `~/work/api` | `feature/auth` | `a1b2c3d4e5f6abcd:feature_auth` |
| `~/work/web` | `main` | `9f8e7d6c5b4a3210:main` |

Branch is detected via `git rev-parse --abbrev-ref HEAD`, then sanitised (slashes → underscores) so the slug is safe for storage. Detached HEAD or non-git directories yield `unknown` so indexing still works.

Implication: switching git branches **automatically** gives you a different RAG view. Re-running `gemma rag index` on the new branch only embeds chunks that actually changed (the embed cache picks up identical chunks across branches by content hash).

---

## Indexing pipeline (`gemma rag index`)

`RAGIndexer.index(progress=...)` runs five phases:

### Phase 1 — Walk + filter

`Manifest.walk(root)` yields every file under `root` that:

1. Has an extension in the allowlist (~50 common code/text formats).
2. Is below `max_file_size` (default 1 MB).
3. Is not denied by [`gemma/safety.py`](../gemma/safety.py)'s policy (e.g. `.git/`, `node_modules/`, lockfiles).
4. Is not a symlink to outside the workspace.

Each surviving file becomes a `FileEntry` with `path`, `size`, `mtime`, and a lazy SHA-1.

### Phase 2 — Manifest diff

The store keeps a `manifest_hash` table of `path → blob_sha`. The indexer:

1. Loads the previous manifest.
2. For each current file, compares mtime+size first. If unchanged, the SHA is assumed unchanged (skips the I/O for the SHA). When `--force-hash` is passed the SHA is recomputed regardless.
3. Classifies every file as `added`, `changed`, `unchanged`, or `removed`.

This is the **incremental** part: a re-index after editing one file only touches that file and its dependents. The number of embed calls scales with what changed, not with total workspace size.

### Phase 3 — Chunk

`gemma.chunking.chunk_text` splits each file into windows targeting roughly **600 model tokens** (~2400 chars) with 60-token overlap. Markdown and code-aware splits prefer to break at headings or function boundaries.

Each chunk emits a stable `chunk_id` of the form `{file_sha}:{chunk_index}` so re-chunking is idempotent across runs.

### Phase 4 — Embed (with caching + parallelism)

For each chunk we compute a `content_hash = sha256(embed_input)` where `embed_input` includes the chunk text plus a header so identical text in different contexts gets distinct cache entries.

Then:

1. **Bulk cache lookup** — `mget_embed_cache(model, hashes)` returns vectors for chunks whose hash hits.
2. **Cache misses** are batched and embedded via the active embedder (LM Studio or Ollama).
3. **Cache writeback** — `mset_embed_cache(model, vectors, ttl_seconds)` stores the new vectors.
4. **Bulk upsert** — `store.upsert_chunk` writes each chunk + vector to `rag_chunks` / `rag_embeddings`.

When `cfg.embed_concurrency > 1`, embed-call batches are dispatched across a `ThreadPoolExecutor`. Each worker gets its own embedder via `embedder_factory()` so HTTP keep-alive sessions don't serialise. The cap is 16 to avoid thread storms.

#### Why a content-hash cache matters

The same chunk often shows up across branches. `git checkout dev` after `gemma rag index` on `main` would re-embed everything if not for the cache — instead it's almost entirely cache hits. Telemetry surfaced via `IndexStats.chunks_cache_hit` makes this visible in the post-run summary.

### Phase 5 — Cleanup

For every `removed` file we call `store.delete_file(path)` which cascades to the chunks and their embeddings.

The new manifest is then saved and `set_meta(dim, model)` records the embedding model fingerprint so a future profile switch (e.g. `nomic-embed-text-v1.5` → `mxbai-embed-large`) is detectable.

---

## Vector storage layout

The store is partitioned by namespace and keyed by `chunk_id`. Two storage backends implement the same interface; pick via `Config.storage_backend`.

### SQLite (default)

```sql
CREATE TABLE rag_chunks (
  namespace  TEXT,
  chunk_id   TEXT,
  path       TEXT,
  start_line INTEGER,
  end_line   INTEGER,
  text       TEXT,
  header     TEXT,
  PRIMARY KEY (namespace, chunk_id)
);

CREATE TABLE rag_embeddings (
  namespace TEXT,
  chunk_id  TEXT,
  vector    BLOB,            -- raw float32 bytes
  dim       INTEGER,
  PRIMARY KEY (namespace, chunk_id),
  FOREIGN KEY (namespace, chunk_id)
    REFERENCES rag_chunks(namespace, chunk_id) ON DELETE CASCADE
);
```

Embeddings are stored as raw `np.float32` BLOBs. `np.frombuffer(blob, dtype=np.float32)` round-trips cleanly. ON DELETE CASCADE means deleting a chunk drops its vector with no app-level coordination.

### Redis (legacy)

Per-namespace key set, documented inline in [`gemma/rag/store.py`](../gemma/rag/store.py):

```
gemma:rag:{ns}:index               SET    chunk_ids
gemma:rag:{ns}:chunk:{cid}         HASH   metadata + text
gemma:rag:{ns}:embed:{cid}         STR    raw float32 bytes
gemma:rag:{ns}:manifest            HASH   path → blob_sha
gemma:rag:{ns}:meta                HASH   dim, model, last_indexed_at
gemma:rag:{ns}:file_chunks:{path}  SET    reverse index for delete-by-file
gemma:rag:embed_cache:v1:{model}:{sha} STR  vector cache (TTL via EXPIRE)
```

The embed cache is **namespace-agnostic** so identical chunks across branches share one vector. Versioned with `v1` so we can ship a new `embed_input` format later without orphaning old keys.

---

## Query pipeline (`gemma rag query`)

`RAGRetriever.query(q, k=5, mmr_lambda=0.7)` runs:

### Step 1 — Embed the query

`embedder.embed(q)` returns a 768-dim float32 vector. If embedding fails (model not loaded, server down) we return `[]` so the caller can degrade gracefully.

### Step 2 — Fetch a candidate pool

`store.search_with_embeddings(query_vec, k=fetch_k)` does:

1. `load_all_embeddings()` — one bulk read of every vector in the namespace.
2. Stack into a `(N, D)` matrix.
3. Compute `scores = matrix @ query` (vectors are pre-normalised at write time, so dot product = cosine).
4. `np.argpartition` for the top-`fetch_k` indices.
5. One pipelined HGETALL / SELECT to fetch the chunk metadata for the winners.

`fetch_k = max(k, k * 4)` by default — we pull a pool ~4× larger than the result count so the MMR re-rank in step 3 has room to diversify.

### Step 3 — MMR re-rank

Maximal Marginal Relevance trades **relevance** for **diversity**. Without it, the top-k is often three near-duplicate chunks from the same file. With it, you get one strong hit per code area.

```
mmr_score(c) = λ · sim(q, c) − (1−λ) · max_{c′ ∈ selected} sim(c, c′)
```

- `λ = 1.0` → pure cosine (no diversity).
- `λ = 0.0` → pure novelty (often irrelevant).
- Default: `0.7` — strong relevance bias with enough novelty to spread results across files.

### Step 4 — Return

Each surviving candidate becomes a `RetrievalHit` carrying `chunk_id`, `path`, `start_line`, `end_line`, `text`, `header`, `score`. `gemma rag query` renders a Rich table; programmatic callers consume the list directly.

---

## Operational surface

### Commands

| Command | What it does |
|---|---|
| `gemma rag index [path]` | Walk + chunk + embed the workspace at `path` (default: cwd). Incremental — re-runs only embed changed files. |
| `gemma rag query "<q>"` | Run a query. Prints top-k hits with file:line citations. |
| `gemma rag status` | Show namespace, chunk count, manifest size, embedding model + dim. |
| `gemma rag reset` | Drop every chunk for the current namespace. Use when switching embedding models. |
| `gemma rag cache stats` | Embed cache row count + per-model breakdown + approximate disk footprint. |
| `gemma rag cache clear [--model X]` | Flush the embed cache, optionally for one model only. |

### Tunables (in `Config`)

| Field | Default | What it controls |
|---|---|---|
| `embed_concurrency` | `1` | Parallelism for the embed-call batches. Bump to 2–4 on multi-core hosts; clamped to `[1, 16]`. |
| `embed_cache_enabled` | `True` | Whether to consult the content-hash cache before each embed call. |
| `embed_cache_ttl_days` | `30` | TTL on cache entries. After expiry the next index re-embeds. |

### When to `gemma rag reset`

- Switching embedding models (the dim or model fingerprint changes — surfaces as a warning in `gemma rag status`).
- Workspace metadata is corrupt (rare — usually a manual SQLite/Redis edit gone wrong).
- After a major reorganisation where most files moved (cleaner than a stale manifest).

---

## Performance characteristics

End-to-end on an M-series Mac with `nomic-embed-text-v1.5` MLX:

| Workload | Latency |
|---|---|
| Index 1k files (cold) | ~30 s (embedding-bound) |
| Index 1k files (warm, no changes) | ~200 ms (manifest diff only) |
| Re-index after 1-file edit | ~50 ms |
| Query against a 5k-chunk index | ~10 ms |
| Query against a 50k-chunk index | ~30 ms |

The query path is dominated by the brute-force cosine; the index path is dominated by embedding throughput. If your index ever grows past ~50k chunks and the query latency starts to bite, swap the storage backend to a `sqlite-vec` virtual table or a `usearch` sidecar — both fit behind the existing factory.

---

## Tracking in code

If you're touching the pipeline, the high-leverage files in order are:

1. [`gemma/rag/indexer.py`](../gemma/rag/indexer.py) — `RAGIndexer.index()` is the orchestration entry point.
2. [`gemma/rag/retrieval.py`](../gemma/rag/retrieval.py) — `RAGRetriever.query()` is the query entry point.
3. [`gemma/storage/sqlite_rag.py`](../gemma/storage/sqlite_rag.py) — `SQLiteRAGStore.search_with_embeddings()` is where the cosine math lives.
4. [`gemma/rag/_math.py`](../gemma/rag/_math.py) — `normalise()` and the MMR helpers.

Tests:
- [`tests/test_rag_indexer.py`](../tests/test_rag_indexer.py) — indexer behaviour with stub stores + embedders.
- [`tests/test_storage_sqlite_rag.py`](../tests/test_storage_sqlite_rag.py) — SQLite store contract.
- [`tests/test_retrieval.py`](../tests/test_retrieval.py) — query-path math.
