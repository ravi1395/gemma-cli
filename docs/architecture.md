# gemma-cli — Architecture Overview

> A layered map of how a CLI invocation flows through the codebase. Every box is
> a concrete class or module; every arrow names the data object that crosses it.
> All diagrams are Mermaid — they render natively in GitHub, GitLab, VS Code
> (with the Mermaid extension), and most modern Markdown viewers.

---

## Contents

1. [Big picture — seven layers](#1-big-picture--seven-layers)
2. [Data objects that cross layers](#2-data-objects-that-cross-layers)
3. [Flow: `gemma ask "…"`](#3-flow-gemma-ask-)
4. [Flow: `gemma rag index`](#4-flow-gemma-rag-index)
5. [Flow: `gemma rag query`](#5-flow-gemma-rag-query)
6. [Flow: `gemma tools run`](#6-flow-gemma-tools-run)
7. [Class reference — methods at a glance](#7-class-reference--methods-at-a-glance)

---

## 1. Big picture — seven layers

The codebase is intentionally layered: each arrow points **down** toward
infrastructure. No lower layer imports anything from a higher layer. The
`gemma.commands.*` package is the only place that knows about Typer; the domain
packages (`memory`, `rag`, `tools`) are pure Python and therefore trivially
testable.

```mermaid
flowchart TD
    subgraph L1["L1 — Entry point"]
      M["gemma.main:app\n(Typer, top-level callback)"]
    end

    subgraph L2["L2 — Command handlers (Typer veneers)"]
      CAsk["commands.* (ask/chat/pipe)"]
      CMem["commands.memory\n(remember/forget/pin/context)"]
      CSh["commands.shell\ncommands.explain\ncommands.git"]
      CTools["commands.tools\n(list/run/audit)"]
      CRag["commands.rag\n(index/query/status/reset)"]
      CComp["commands.completion\ncommands.clipboard"]
    end

    subgraph L3["L3 — Cross-cutting services"]
      Cfg["Config"]
      Out["output.render_response\nOutputMode"]
      Cache["ResponseCache"]
      Hist["SessionHistory"]
      Safe["safety.SafetyPolicy\nensure_inside / archive"]
      Red["redaction\n(secrets scrubber)"]
    end

    subgraph L4["L4 — Domain subsystems"]
      Mem["gemma.memory\n(MemoryManager)"]
      Tools["gemma.tools\n(Dispatcher, Registry)"]
      RAG["gemma.rag\n(RAGIndexer, RAGRetriever)"]
      Chunk["chunking.chunk_for_path\n→ List[Chunk]"]
    end

    subgraph L5["L5 — Infra clients"]
      Cli["client.chat"]
      Emb["Embedder"]
      MemStore["MemoryStore"]
      RagStore["RedisVectorStore"]
      Sub["subprocess_runner.run"]
      Audit["tools.audit"]
    end

    subgraph L6["L6 — Off-box services"]
      Oll["Ollama\n(chat + embeddings HTTP)"]
      Red2["Redis\n(memories, rag, cache)"]
      FS["Local filesystem\n(workspace files,\narchive folder)"]
      Net["HTTPS allowlist\n(urllib)"]
    end

    M --> CAsk
    M --> CMem
    M --> CSh
    M --> CTools
    M --> CRag
    M --> CComp

    CAsk --> Cfg
    CAsk --> Out
    CAsk --> Cache
    CAsk --> Hist
    CAsk --> Mem
    CAsk --> Cli

    CRag --> Cfg
    CRag --> RAG
    CRag --> Emb
    CRag --> RagStore

    CTools --> Tools
    CTools --> Audit

    Mem --> MemStore
    Mem --> Cli
    Mem --> Emb
    RAG --> Chunk
    RAG --> Emb
    RAG --> RagStore
    Tools --> Sub
    Tools --> Safe
    Tools --> Audit

    Cli --> Oll
    Emb --> Oll
    MemStore --> Red2
    RagStore --> Red2
    Cache --> Red2
    Sub --> FS
    Safe --> FS
    Tools --> Net
```

**How to read the layers**

| Layer | Owns                               | Key invariant                                     |
|-------|------------------------------------|---------------------------------------------------|
| L1    | Typer `app`, global `--profile`    | Decides *which* handler runs; no business logic.  |
| L2    | One module per command family      | Parses Typer args → calls L3/L4. No Ollama/Redis. |
| L3    | Config, output mode, cache, safety | Pure helpers; safe to import from anywhere in L4. |
| L4    | Memory, tools, RAG subsystems      | All domain logic lives here; hermetic under test. |
| L5    | HTTP + Redis + subprocess clients  | Only layer that performs I/O to external systems. |
| L6    | Ollama, Redis, local FS, HTTPS     | Ground truth — nothing else is authoritative.     |

---

## 2. Data objects that cross layers

These are the value objects that flow between layers. Every one is an
`@dataclass` (or `Enum`) with explicit fields — no stringly-typed dicts crossing
module boundaries.

```mermaid
classDiagram
    class Config {
      +model : str
      +ollama_host : str
      +redis_url : str
      +embedding_model : str
      +memory_enabled : bool
      +cache_enabled : bool
      +temperature : float
      +system_prompt : str
      +load_profile(name) Config
    }

    class ConversationTurn {
      +role : str
      +content : str
      +timestamp : float
    }

    class MemoryRecord {
      +id : str
      +content : str
      +category : MemoryCategory
      +importance : int
      +created_at : float
      +embedding : bytes
    }

    class Chunk {
      +text : str
      +start_line : int
      +end_line : int
      +header : str
    }

    class FileEntry {
      +path : str
      +mtime_ns : int
      +size : int
      +sha1 : str
      +chunk_ids : List[str]
      +from_disk(root, path) FileEntry
      +same_content_as(other) bool
    }

    class ManifestDiff {
      +added : List~FileEntry~
      +changed : List~Tuple~
      +removed : List~FileEntry~
      +unchanged : List~FileEntry~
    }

    class StoredChunk {
      +id : str
      +path : str
      +start_line : int
      +end_line : int
      +text : str
      +header : str
      +score : float
    }

    class RetrievalHit {
      +chunk_id : str
      +path : str
      +start_line : int
      +end_line : int
      +text : str
      +score : float
      +line_range str
      +citation str
    }

    class ToolSpec {
      +name : str
      +description : str
      +capability : Capability
      +schema : dict
    }

    class ToolResult {
      +ok : bool
      +value : Any
      +error : str
    }

    class GateDecision {
      +allow : bool
      +reason : str
    }

    class AuditRecord {
      +ts : str
      +tool : str
      +args_digest : str
      +ok : bool
      +elapsed_ms : int
    }

    ConversationTurn --> MemoryRecord : condensed into
    Chunk --> FileEntry : produced for
    FileEntry --> ManifestDiff : diffed via
    Chunk --> StoredChunk : embedded + persisted as
    StoredChunk --> RetrievalHit : MMR-selected as
    ToolSpec --> GateDecision : gated by
    ToolSpec --> ToolResult : returns
```

---

## 3. Flow: `gemma ask "…"`

The most-travelled path. Memory-augmented, cache-aware, supports streaming and
three scripting output modes (`--json / --only / --code`).

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant CLI as main.app (Typer)
    participant AskH as ask() handler
    participant Cfg as Config
    participant Mgr as MemoryManager
    participant Ctx as ContextAssembler
    participant Retr as MemoryRetriever
    participant Cache as ResponseCache
    participant Cli as client.chat
    participant Oll as Ollama
    participant Out as render_response

    User->>CLI: gemma ask "why did X fail?"
    CLI->>AskH: dispatch with parsed flags
    AskH->>Cfg: _make_config(profile, flags)
    AskH->>Mgr: initialize()
    Mgr-->>AskH: available / degraded

    AskH->>Mgr: record_turn("user", prompt)
    AskH->>Mgr: get_context_messages(prompt)
    Mgr->>Ctx: build_messages(prompt, history)
    Ctx->>Retr: top_k(prompt) → List[MemoryRecord]
    Retr-->>Ctx: relevant memories
    Ctx-->>Mgr: List[messages]
    Mgr-->>AskH: List[messages]

    alt no_stream and cacheable
      AskH->>Cache: get(messages, cfg)
      Cache-->>AskH: cached reply or None
    end

    alt cache miss
      AskH->>Cli: chat(messages, cfg, stream=?)
      Cli->>Oll: POST /api/chat
      Oll-->>Cli: tokens stream
      Cli-->>AskH: Iterator[(event, data)]
    end

    AskH->>Out: render_response(gen, mode, ...)
    Out-->>User: stdout (rich / json / only / code)
    AskH->>Mgr: record_turn("assistant", reply)
    AskH->>Cache: put(messages, cfg, reply)
    Note over Mgr: background task may run<br/>CondensationPipeline when<br/>sliding window overflows
```

**Where things can go quiet:**

- `Mgr.initialize()` returns `degraded=True` if Redis is unreachable — a warning
  prints but the command proceeds stateless.
- `Cache` only engages when `--no-stream` is set *and* temperature is below the
  cache ceiling; streaming responses skip the cache entirely.
- If `--cache-only` is set and there is no hit, the handler exits **1**.

---

## 4. Flow: `gemma rag index`

Incremental indexer. The second run after a single-file edit embeds exactly one
file's chunks — everything else is a no-op diff.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant CLI as main.app
    participant IdxH as index_command
    participant NS as resolve_namespace
    participant Store as RedisVectorStore
    participant Idx as RAGIndexer
    participant Man as Manifest
    participant Chunk as chunking.chunk_for_path
    participant Emb as Embedder
    participant Oll as Ollama embeddings
    participant Redis

    User->>CLI: gemma rag index [path]
    CLI->>IdxH: dispatch
    IdxH->>NS: resolve_namespace(root, branch)
    NS-->>IdxH: "{hash12}:{branch}"
    IdxH->>Store: new RedisVectorStore(namespace, redis_url)
    IdxH->>Idx: RAGIndexer(root, store, embedder).index()

    Idx->>Idx: _walk() — prune _SKIP_DIRS,<br/>filter by extension + size
    loop per file
      Idx->>Man: FileEntry.from_disk(root, path)
    end
    Idx->>Store: load_manifest_hash()
    Store-->>Idx: old manifest blobs
    Idx->>Man: Manifest.from_redis_hash(old).diff(new)
    Man-->>Idx: ManifestDiff(added, changed, removed, unchanged)

    Idx->>Store: delete_file(path) for removed + changed
    Store->>Redis: DEL chunk/embed/file_chunks keys

    loop batch of 32 files
      Idx->>Chunk: chunk_for_path(path, text) → List[Chunk]
      Idx->>Emb: embed_batch([prefixed inputs])
      Emb->>Oll: POST /api/embeddings
      Oll-->>Emb: float32 vectors
      loop per chunk
        Idx->>Store: upsert_chunk(id, path, lines,<br/>text, header, embedding)
        Store->>Redis: HSET chunk + SET embed bytes +<br/>SADD index + file_chunks
      end
    end

    Idx->>Store: save_manifest_hash(blobs)
    Idx->>Store: set_meta(dim, model)
    Idx-->>IdxH: IndexStats(scanned, added, ...)
    IdxH-->>User: Rich-rendered summary line
```

**Invariants enforced here:**

- Write-only to Redis; the indexer never mutates files on disk (honours the
  never-delete rule).
- Embeddings are L2-normalised before `upsert_chunk`, so downstream cosine is a
  plain dot product.
- `save_manifest_hash` does a pipelined DEL + HSET to replace the hash
  atomically (no orphan entries).

---

## 5. Flow: `gemma rag query`

Cosine top-`fetch_k` against the store, then MMR-shrunk to `k`. No Ollama call
happens if the query is whitespace, the store is empty, or the embedder raises.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant CLI as main.app
    participant QH as query_command
    participant Store as RedisVectorStore
    participant Retr as RAGRetriever
    participant Emb as Embedder
    participant Oll as Ollama embeddings
    participant MMR as _mmr()
    participant Out as Rich Console

    User->>CLI: gemma rag query "how does auth work?" --k 5 --mmr 0.5
    CLI->>QH: dispatch
    QH->>Store: chunk_count()
    alt empty store
      QH-->>User: yellow error, exit 1
    end

    QH->>Retr: RAGRetriever(store, embedder).query(q, k, mmr)
    Retr->>Emb: embed(q)
    Emb->>Oll: POST /api/embeddings
    Oll-->>Emb: float32 vector
    Emb-->>Retr: np.ndarray

    Retr->>Store: search(query_vec, k=fetch_k=k*4)
    Store->>Store: load_all_embeddings() (MGET)
    Store->>Store: np.argpartition for top-pool
    Store-->>Retr: List[StoredChunk] (with .score)

    Retr->>Store: load_all_embeddings()  # for pairwise sims
    Store-->>Retr: Dict[id, np.ndarray]
    Retr->>MMR: greedy selection (λ · rel − (1−λ) · max_sim)
    MMR-->>Retr: List[StoredChunk] (trimmed to k)
    Retr-->>QH: List[RetrievalHit]

    loop per hit
      QH->>Out: "#i citation:lines  score=+0.812"
      QH->>Out: text preview (first 15 lines)
    end
    Out-->>User: rendered hits
```

**Graceful-degradation contract:** every early exit in `RAGRetriever.query`
returns `[]` so the CLI can always render "no hits" cleanly — never a traceback
in the user's terminal.

---

## 6. Flow: `gemma tools run`

The tool-use subsystem. Same dispatcher the chat-loop uses when a model emits a
tool call. Every hop is gated and audited.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant CLI as main.app
    participant TR as tools_run_command
    participant Reg as registry.get
    participant Disp as Dispatcher
    participant Gate as capabilities.gate
    participant Schema as _validate_against_schema
    participant Handler as tool handler<br/>(e.g. fs_write)
    participant Safe as safety.ensure_inside
    participant Sub as subprocess_runner.run
    participant Audit as audit.append

    User->>CLI: gemma tools run fs_write --arg path=foo.txt --arg content=…
    CLI->>TR: dispatch
    TR->>Reg: get(name)
    Reg-->>TR: (ToolSpec, handler)
    TR->>Disp: Dispatcher(ctx).dispatch(name, args)

    Disp->>Gate: gate(spec.capability, ctx)
    alt denied
      Gate-->>Disp: GateDecision(allow=False, reason)
      Disp-->>TR: ToolResult(ok=False, error=reason)
    end

    Disp->>Schema: validate(args, spec.schema)
    alt invalid
      Schema-->>Disp: raises
      Disp-->>TR: ToolResult(ok=False, error=...)
    end

    Disp->>Handler: handler(args, ctx)

    opt path-taking tool
      Handler->>Safe: ensure_inside(workspace, path)
      Handler->>Safe: ensure_no_symlink_escape(path)
    end
    opt subprocess tool (lint/tests)
      Handler->>Sub: run(cmd, timeout, env_allowlist)
      Sub-->>Handler: RunResult(stdout, stderr, exit_code)
    end
    opt fs_write / archive
      Handler->>Safe: archive(existing) before overwrite
      Handler->>Handler: atomic os.replace(tmp, path)
    end

    Handler-->>Disp: ToolResult(ok, value)
    Disp->>Audit: append(AuditRecord{ts, tool,<br/>args_digest, ok, elapsed_ms})
    Audit->>Audit: redact() before write
    Disp-->>TR: ToolResult
    TR-->>User: JSON-rendered result
```

**Security posture encoded in this flow:**

- There is no `fs_delete` handler in the registry — the never-delete rule is
  enforced by *omission*, not by a runtime check that could be bypassed.
- Capability gating runs **before** schema validation so a disallowed tool
  never sees its arguments.
- `AuditRecord` paths are hashed (`PathDigest`) so the log never leaks absolute
  filesystem layout.

---

## 7. Class reference — methods at a glance

### L3/L4 classes

```mermaid
classDiagram
    class Config {
      +model, ollama_host, redis_url
      +embedding_model, system_prompt
      +memory_enabled, cache_enabled
      +temperature, ollama_keep_alive
      +load_profile(name) Config
    }

    class ResponseCache {
      +get(messages, cfg) Optional~str~
      +put(messages, cfg, reply) None
    }

    class SessionHistory {
      +show() List~Turn~
      +clear() None
    }

    class SafetyPolicy {
      +workspace_root : Path
      +deny_globs : list
      +ensure_inside(path) None
      +archive(path) Path
    }

    class MemoryManager {
      +initialize() bool
      +available() bool
      +record_turn(role, content) None
      +get_context_messages(prompt, system) list
      +add_memory(content, category, importance) MemoryRecord
      +forget_memory(id) bool
      +pin_memory(id) bool
      +list_memories(limit) list
      +clear_session() None
      +get_stats() dict
    }

    class Dispatcher {
      +mount_specs() list
      +advertised_schemas() list
      +dispatch(name, args) ToolResult
      -_refuse(reason) ToolResult
    }

    class Registry {
      +register(spec, handler) None
      +get(name) tuple
      +all_specs() list
      +as_openai_tools(specs) list
      +mount(ctx) list
    }
```

### RAG classes

```mermaid
classDiagram
    class RAGIndexer {
      -_root : Path
      -_store : RedisVectorStore
      -_embedder : Embedder
      -_extensions : tuple
      -_max_file_size : int
      +index(progress) IndexStats
      -_walk() Iterable~Path~
      -_apply_deletes(diff) int
      -_apply_upserts(diff, stats) None
      -_chunk_file(entry) List~Chunk~
    }

    class RAGRetriever {
      -_store : RedisVectorStore
      -_embedder : Embedder
      +query(q, k, mmr_lambda, fetch_k) List~RetrievalHit~
    }

    class RedisVectorStore {
      +namespace : str
      +upsert_chunk(id, path, lines, text, header, emb) None
      +delete_chunk(id) None
      +delete_file(path) int
      +load_manifest_hash() dict
      +save_manifest_hash(blobs) None
      +set_meta(dim, model) None
      +get_meta() dict
      +chunk_count() int
      +get_chunk(id) Optional~StoredChunk~
      +get_embedding(id) Optional~ndarray~
      +load_all_embeddings() dict
      +search(query_vec, k) List~StoredChunk~
      +clear_namespace() int
    }

    class Manifest {
      +from_redis_hash(blobs) Manifest
      +to_redis_hash() dict
      +diff(new) ManifestDiff
    }

    RAGIndexer --> RedisVectorStore
    RAGIndexer --> Manifest
    RAGRetriever --> RedisVectorStore
    RAGIndexer ..> Embedder
    RAGRetriever ..> Embedder
```

### Tools classes

```mermaid
classDiagram
    class ToolSpec {
      +name : str
      +description : str
      +capability : Capability
      +schema : dict
    }

    class ToolResult {
      +ok : bool
      +value : Any
      +error : str
    }

    class Capability {
      <<enum>>
      READ
      WRITE
      ARCHIVE
      NETWORK
    }

    class GatingContext {
      +workspace_root : Path
      +allowed : set~Capability~
    }

    class GateDecision {
      +allow : bool
      +reason : str
    }

    class AuditRecord {
      +ts : str
      +tool : str
      +args_digest : str
      +ok : bool
      +elapsed_ms : int
    }

    class RunResult {
      +exit_code : int
      +stdout : str
      +stderr : str
      +timed_out : bool
      +truncated : bool
    }

    Dispatcher ..> ToolSpec
    Dispatcher ..> ToolResult
    Dispatcher ..> GateDecision
    Dispatcher ..> AuditRecord
    subprocess_runner ..> RunResult
```

---

## Living document

When a new subsystem lands, add a section 8+ with its own sequence diagram.
Keep every sequence diagram self-contained — a reader should not have to chase
imports across multiple files to understand one flow.

_Last synchronised against code: Phase 6.2 (RAG subsystem, `rag` Typer group)._
