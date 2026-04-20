# gemma-cli — Capabilities Reference

> A local CLI tool for Google's Gemma 4 LLM (via Ollama) with a Redis-backed recursive summarization memory system, agentic tool-use, and local RAG. Designed to give a 4B parameter model the practical memory and context-awareness of a much larger system.

---

## Table of Contents

1. [Core Concept](#core-concept)
2. [Memory System](#memory-system)
3. [Response Cache](#response-cache)
4. [Tool-Use Agent](#tool-use-agent)
5. [Local RAG](#local-rag)
6. [Secret Redaction](#secret-redaction)
7. [Command Reference](#command-reference)
   - [ask](#ask--single-shot-query)
   - [chat](#chat--interactive-repl)
   - [pipe](#pipe--stdin-processing)
   - [Memory Commands](#memory-commands)
   - [Terminal-Assistant Commands](#terminal-assistant-commands)
   - [RAG Commands](#rag-commands)
   - [Tool Commands](#tool-commands)
8. [Configuration & Profiles](#configuration--profiles)
9. [Scripting & Output Modes](#scripting--output-modes)
10. [Graceful Degradation](#graceful-degradation)
11. [Architecture Overview](#architecture-overview)

---

## Core Concept

gemma-cli wraps a locally-running Gemma 4 model in a layered system that solves the key practical limitations of small, local LLMs:

| Problem | Solution |
|---|---|
| No memory between sessions | Dual-tier Redis memory system |
| Token window overflows | Recursive self-summarization into compressed memories |
| Repetitive API calls are slow | SHA256-keyed response cache |
| Can't act on files or the web | Sandboxed tool-use agent loop |
| Can't reason over your codebase | Incremental local RAG indexer |
| Secrets accidentally stored | Automatic redaction before Redis writes |

---

## Memory System

The most distinctive feature of gemma-cli. The model builds and retrieves its own persistent memory across sessions using a two-tier design.

### Tier 1: Sliding Window

The last **8 raw conversation turns** are stored verbatim in Redis and prepended to every prompt. This guarantees the model always has immediate, recent context without re-summarizing it.

```
[turn -8] [turn -7] ... [turn -1]  <-- always injected verbatim
```

### Tier 2: Condensed Memories

When the window overflows (more than 8 turns), Gemma reads its own conversation at low temperature (`0.2`) and extracts structured facts:

```json
{
  "content": "User prefers type annotations in all Python code.",
  "category": "user_preference",
  "importance": 4,
  "source_summary": "Stated during code review discussion."
}
```

These facts are embedded with `nomic-embed-text` (768-dimensional vectors) and stored in Redis. On every subsequent query, the top-K most semantically similar memories are retrieved via cosine similarity and injected before the sliding window.

### Memory Categories

| Category | Description | Example |
|---|---|---|
| `user_preference` | Stated preferences | `"I prefer snake_case over camelCase"` |
| `task_state` | In-progress work | `"Auth rewrite in progress, JWT-based"` |
| `factual_context` | Environment facts | `"OS: macOS Sonoma, stack: FastAPI + Redis"` |
| `instruction` | Explicit instructions | `"Always return JSON in responses"` |
| `correction` | Corrections to prior output | `"Redis TTL is in seconds, not ms"` |
| `relationship` | People, teams, org context | `"Maya owns the payments service"` |
| `tool_usage` | Specific commands, APIs | `"Deploy via: make deploy ENV=prod"` |

### Importance Tiers & TTL

| Importance | Label | TTL | Use When |
|---|---|---|---|
| 5 | Critical | Never expires | Core identity, key permanent preferences |
| 4 | High | 7 days | Ongoing project context |
| 3 | Medium | 72 hours | Incidental but useful facts |
| 2 | Low | 24 hours | Passing mentions |
| 1 | Trivial | 6 hours | Probably won't matter next session |

### Reconsolidation

When total memory count exceeds **200**, Gemma merges the full memory set into a shorter, deduplicated list — the "recursive" step. Older, lower-importance, and redundant memories are merged or dropped.

### Example: Memory in Action

```bash
# Session 1 — Tuesday
$ gemma chat
You: I'm building an auth service. We're using JWT with RS256.
Gemma: Got it. RS256 uses asymmetric keys — your public key for verification...

# [session ends — memory is condensed and stored]

# Session 2 — Thursday (new terminal, no context)
$ gemma chat
You: What format are we using for tokens?
Gemma: Based on our previous discussion, you're using JWT with RS256...
#      ^^^^ retrieved from condensed memory, not re-stated by user
```

---

## Response Cache

Deterministic, low-temperature calls are cached by SHA256 key so identical queries skip the Ollama round-trip entirely.

**Cache key includes:** model name, temperature, system prompt, user prompt, and `keep_alive` value. A single changed character produces a cache miss.

**Default behavior:** Caching is active for `gemma commit`, `gemma sh`, and `gemma diff` (all run at temperature ≤ 0.3). Interactive `chat` is not cached.

```bash
# First call — hits Ollama (~1.8s)
$ gemma ask --json "list three sorting algorithms"
{"content": "...", "elapsed_ms": 1842, "cache_hit": false}

# Identical second call — served from cache (<5ms)
$ gemma ask --json "list three sorting algorithms"
{"content": "...", "elapsed_ms": 3, "cache_hit": true}

# Force a fresh call, bypassing cache
$ gemma ask --no-cache "list three sorting algorithms"

# Assert that a call MUST be cached (useful in CI)
$ gemma ask --cache-only "list three sorting algorithms"
```

Configuration:

```toml
cache_enabled = true
cache_ttl_seconds = 3600        # 1 hour default
cache_temperature_max = 0.3     # won't cache above this temperature
```

---

## Tool-Use Agent

An experimental sandboxed agent loop that lets Gemma call tools on your behalf. The model emits `tool_calls`; the dispatcher runs them; results are fed back in a loop until the model produces a final answer or the turn budget is exhausted.

### Built-in Tools

| Tool | Capability | Description |
|---|---|---|
| `fs_read` | READ | Read files from workspace |
| `fs_write` | WRITE | Write/modify files |
| `fs_archive` | ARCHIVE | Create tar/zip snapshots |
| `lint` | READ | Run linters (flake8, eslint, etc.) |
| `tests` | READ | Run test suites (pytest, jest, etc.) |
| `net_fetch` | READ | HTTPS fetch from allowlisted domains |
| `rag_query` | READ | Query local RAG index |
| `web_search` | READ | DuckDuckGo-backed web search |
| `plan` | READ | Decompose complex tasks (feature-flagged) |

### Capability Gating

- **READ** tools auto-approve (safe, no side effects)
- **WRITE** tools prompt for confirmation in TTY mode
- **ARCHIVE** tools create read-only snapshots

### Example: Agentic Ask

```bash
$ gemma ask "Find all TODO comments in the repo and summarize them by file."
# Gemma calls: fs_read on multiple files, then synthesizes a summary
# All READ calls auto-approved; no prompts needed

$ gemma ask "Add type annotations to auth.py and run the linter."
# Gemma calls: fs_read (auto) → fs_write (prompts you) → lint (auto)
# You confirm the write before it executes
```

```bash
# Disable the agent loop entirely (pure LLM response)
$ gemma ask --no-agent "explain what a JWT is"

# Inspect registered tools
$ gemma tools list

# Run a tool directly (useful for debugging)
$ gemma tools run fs_read --arg path=src/auth.py

# View the tool audit log
$ gemma tools audit
```

### Agent Configuration

```toml
agent_max_turns = 8           # max tool-call rounds per ask
agent_tool_cache = true       # memoize READ tools within a session
agent_tool_concurrency = 4    # parallel dispatch threads
```

---

## Local RAG

Retrieval-Augmented Generation over your local workspace. Files are chunked, embedded, and stored in Redis. Queries retrieve the most semantically relevant chunks and inject them into the prompt.

### How It Works

1. `gemma rag index .` scans your workspace for source files
2. Each file is chunked (with header metadata and line numbers)
3. Chunks are embedded with `nomic-embed-text` (768-dim vectors)
4. Only changed files (different `mtime`, `size`, or `sha1`) are re-embedded — fully incremental
5. Embeddings are content-addressable cached for 30 days

### Example: Indexing and Querying

```bash
# Index the current workspace
$ gemma rag index .
Indexed 312 files, 4,821 chunks. Skipped 289 unchanged.

# Ask a question over your codebase
$ gemma rag query "where is the JWT validation logic?"
> src/auth/middleware.py (lines 42–68) — similarity: 0.87
> src/auth/tokens.py (lines 12–35)     — similarity: 0.81

# Check index status
$ gemma rag status
Namespace: workspace/main | Files: 312 | Chunks: 4,821 | Model: nomic-embed-text

# Rebuild from scratch (e.g., after switching embedding models)
$ gemma rag reset && gemma rag index .

# View embed cache stats
$ gemma rag cache stats
```

### RAG in `ask` (via `rag_query` tool)

When the agent is enabled, Gemma can call `rag_query` autonomously:

```bash
$ gemma ask "How does error handling work in the payments module?"
# Gemma calls rag_query → retrieves relevant chunks → synthesizes answer
```

---

## Secret Redaction

Before any conversation turn is written to Redis, gemma-cli automatically scans and redacts sensitive values. Redacted values are replaced with `[REDACTED:TYPE]` — structural context is preserved, secrets are not.

| Pattern | Trigger | Example Match |
|---|---|---|
| `PRIVATE_KEY` | PEM blocks | `-----BEGIN RSA PRIVATE KEY-----` |
| `BEARER_TOKEN` | Authorization headers | `Bearer eyJhbGci...` |
| `ENV_SECRET` | `.env`-style lines with secret keywords | `DATABASE_PASSWORD=s3cr3t` |
| `AWS_ACCESS_KEY` | AKIA/ASIA prefix, 20 chars | `AKIAIOSFODNN7EXAMPLE` |
| `GITHUB_TOKEN` | `ghp_`, `gho_`, `ghs_`, `ghu_`, `ghr_` prefix | `ghp_xxxxxxxxxxx` |
| `GITLAB_TOKEN` | `glpat-` prefix | `glpat-xxxxxxxxxxxx` |
| `JWT` | Three-segment base64url, `eyJ` header | `eyJhbGciOiJSUzI1NiJ9...` |

```bash
# If you paste a .env file into chat, it's stored safely:
# Input:  "here's my config: DATABASE_PASSWORD=hunter2"
# Stored: "here's my config: DATABASE_PASSWORD=[REDACTED:ENV_SECRET]"
```

---

## Command Reference

### `ask` — Single-Shot Query

```bash
gemma ask "<prompt>" [OPTIONS]
```

| Flag | Default | Description |
|---|---|---|
| `--model, -m` | `gemma4:e4b` | Override Ollama model |
| `--system, -s` | config default | Override system prompt |
| `--think` | off | Enable Gemma 4 extended step-by-step reasoning |
| `--no-stream` | off | Collect full response, render as Markdown |
| `--no-memory` | off | Skip memory retrieval and recording |
| `--keep-alive` | `"30m"` | Model residency (`"-1"` = pin, `"0"` = evict immediately) |
| `--json` | off | Emit structured JSON response |
| `--only <field>` | — | Print single JSON field naked (`content`, `elapsed_ms`, etc.) |
| `--code` | off | Extract and emit only fenced code blocks |
| `--no-cache` | off | Bypass cache, always call model |
| `--cache-only` | off | Error if no cache hit |
| `--agent / --no-agent` | on | Enable/disable tool-use agent loop |

```bash
# Basic query
$ gemma ask "What does the --rebase flag do in git pull?"

# Force step-by-step reasoning
$ gemma ask "If 3x + 7 = 22, what is x?" --think

# Use a specific model for a heavier task
$ gemma ask "Review this code for thread safety" --model gemma4:27b

# Stateless — skip memory entirely
$ gemma ask "What is idempotency?" --no-memory

# Pin model in RAM for a burst of queries
$ gemma ask "explain async/await" --keep-alive -1
```

---

### `chat` — Interactive REPL

```bash
gemma chat [OPTIONS]
```

Full conversational session with memory, history, and optional agent loop. Exit with `exit`, `quit`, `:q`, or `Ctrl-D`.

| Flag | Description |
|---|---|
| `--model, -m` | Override model |
| `--system, -s` | Override system prompt |
| `--fresh` | Clear raw sliding window (keep condensed memories) |
| `--no-memory` | Disable all memory features |
| `--think` | Enable extended thinking for the whole session |
| `--keep-alive` | Model residency (default `"30m"`) |

```bash
# Standard session
$ gemma chat

# Start fresh without prior recent context (but keep long-term memories)
$ gemma chat --fresh

# Long deep-work session — pin model in RAM
$ gemma chat --keep-alive -1

# Focused code review session with a tuned profile
$ gemma chat --profile code

# Stateless session (no memory read or write)
$ gemma chat --no-memory
```

---

### `pipe` — Stdin Processing

```bash
echo "..." | gemma pipe "<instruction>"
cat file.txt | gemma pipe "summarize this"
```

Stateless by design — reads from stdin, runs the instruction, exits. Useful in shell pipelines and scripts.

```bash
# Summarize a log file
$ cat app.log | gemma pipe "identify the most critical errors"

# Explain a diff
$ git diff HEAD~1 | gemma pipe "explain these changes for a non-technical reviewer"

# Generate docs from source
$ cat src/auth.py | gemma pipe "generate docstrings for all public functions" --code

# Chain with other tools
$ kubectl get events | gemma pipe "what is failing and why?" | tee incident-notes.txt
```

---

### Memory Commands

#### `gemma remember`

Manually seed a fact into the memory store.

```bash
$ gemma remember "<fact>" [--category <cat>] [--importance <1-5>]
```

```bash
# Save a preference
$ gemma remember "I prefer type annotations in all Python code." --category pref

# Save a project fact with high importance (7-day TTL)
$ gemma remember "Auth rewrite uses JWT RS256, deadline April 30." \
    --category task --importance 4

# Save a permanent, never-expiring fact
$ gemma remember "Maya is the team lead for the payments service." \
    --category relationship --importance 5
```

Category shortcuts: `pref` (preference), `task`, `feat` (factual_context), `instruction`, `correction`, `relationship`, `tool_usage`

---

#### `gemma forget`

Soft-delete a memory (marked superseded, never retrieved again; audit trail kept).

```bash
# Forget the most recently created memory
$ gemma forget --last

# Forget by semantic match (targets the most similar memory)
$ gemma forget --match "JWT deadline"

# Forget by ID (shown in `history memories`)
$ gemma forget mem_abc123 --force
```

---

#### `gemma pin`

Set a memory's importance to 5 — it will never expire.

```bash
# Pin by semantic match
$ gemma pin --match "team lead payments"

# Pin by ID
$ gemma pin mem_abc123
```

---

#### `gemma context`

Preview what memories and turns would be injected for a given query — useful for debugging retrieval.

```bash
$ gemma context "auth rewrite JWT"
# Prints a rich table: importance | category | similarity | content
# Also shows recent sliding window turns
```

---

#### `gemma history`

```bash
$ gemma history show                    # raw JSON session history
$ gemma history clear                   # wipe raw window (keep condensed memories)
$ gemma history memories --limit 50     # top-50 condensed memories by importance
$ gemma history stats                   # memory system statistics
```

---

### Terminal-Assistant Commands

#### `gemma sh` — Shell Command Generation

Translates a plain-English description into a shell command. Generated at temperature `0.2` for minimal hallucination. A safety blocklist rejects dangerous patterns (`rm -rf /`, `mkfs`, `dd if=`, fork bombs).

```bash
$ gemma sh "<description>" [OPTIONS]
```

```bash
# Generate and auto-execute
$ gemma sh "find all Python files modified in the last 24 hours"
# Generated: find . -name "*.py" -mtime -1
# Execute? [y/N] y

# Safe mode — print only, never execute
$ gemma sh --no-exec "compress the logs/ directory into a tarball"

# Target a specific shell
$ gemma sh "list background jobs" --shell zsh

# Add an explanatory comment above the command
$ gemma sh --explain "count lines in all .go files recursively"

# Copy to clipboard instead of executing
$ gemma sh --copy "start a Python HTTP server on port 9000"
```

---

#### `gemma explain` — Explain Files, Commands, or Errors

Auto-detects input: file argument → `--cmd` → `--error` → stdin (in that priority order).

```bash
# Explain a file (reads up to N lines)
$ gemma explain app.log --lines 100
$ gemma explain src/auth/middleware.py

# Explain a shell command (not executed — purely descriptive)
$ gemma explain --cmd "git rebase -i HEAD~5"
$ gemma explain --cmd "awk 'NR%2==0' file.txt"

# Explain an error message
$ gemma explain --error "ECONNREFUSED 127.0.0.1:6379"
$ gemma explain --error "java.lang.NullPointerException at line 42"

# Explain with memory context (useful for project-specific errors)
$ gemma explain --error "auth token validation failed" --with-memory
```

---

#### `gemma commit` — Conventional Commit Message Generation

Analyzes staged changes (`git diff --cached`) and generates a conventional commit message.

Output format: `<type>(<scope>): <subject>` with optional body.

```bash
$ git add src/auth/tokens.py
$ gemma commit
# fix(auth): handle RS256 key rotation without restart
#
# Reloads public keys from disk on each validation attempt
# instead of caching at startup.

# Auto-commit without confirmation
$ gemma commit --apply

# Force a specific commit type
$ gemma commit --type feat
$ gemma commit --type docs --apply
```

Supported types: `feat`, `fix`, `chore`, `docs`, `refactor`, `test`, `perf`, `ci`, `build`

---

#### `gemma diff` — Human-Readable Diff Summaries

```bash
# Summarize working-tree changes file by file
$ gemma diff

# Summarize staged changes
$ gemma diff --staged

# Summarize changes between commits as one paragraph
$ gemma diff HEAD~3 --overall

# Summarize branch changes
$ gemma diff main..feature/auth-rewrite --overall
```

---

#### `gemma why` — Explain Last Failed Command

Reads the exit code and command from `~/.gemma_last_cmd` (written by a shell hook) and explains what went wrong.

```bash
# First, install the shell hook (one-time setup)
$ gemma install-shell --append ~/.zshrc
# Backs up original .zshrc, appends precmd hook

# Now every failed command is automatically explained
$ git posh   # typo → exits non-zero
$ gemma why
# 'git posh' is not a valid git subcommand. Did you mean 'git push'?

# Manual override
$ gemma why --last-file ~/.my_last_cmd
```

---

### RAG Commands

```bash
# Build or refresh the index (incremental — only changed files re-embedded)
$ gemma rag index .
$ gemma rag index ./src

# Query your codebase semantically
$ gemma rag query "where is rate limiting implemented?"
$ gemma rag query "what handles database connection pooling?"

# Check what's indexed
$ gemma rag status

# Wipe and rebuild (use after switching embedding models)
$ gemma rag reset
$ gemma rag index .

# Embedding cache management
$ gemma rag cache stats
$ gemma rag cache clear
```

---

### Tool Commands

```bash
# List all registered tools with capabilities
$ gemma tools list

# Invoke a tool directly (same path as the model uses)
$ gemma tools run fs_read --arg path=src/main.py
$ gemma tools run web_search --arg query="redis sorted set performance"

# View append-only audit log of all tool calls
$ gemma tools audit
# timestamp | tool | args_digest | exit_code | elapsed_ms | session_id
```

---

## Configuration & Profiles

### Config File

`~/.config/gemma/profiles/<name>.toml`

### Key Configuration Fields

```toml
# Model
model = "gemma4:e4b"                  # default model
system_prompt = "You are a helpful assistant."
temperature = 0.7
ollama_host = "http://localhost:11434"
ollama_keep_alive = "30m"             # "-1" = pin, "0" = evict

# Memory
memory_enabled = true
redis_url = "redis://localhost:6379/0"
embedding_model = "nomic-embed-text"
sliding_window_size = 8               # raw turns kept verbatim
memory_top_k = 5                      # memories retrieved per query
memory_min_similarity = 0.3           # cosine threshold for retrieval
memory_max_count = 200                # triggers reconsolidation

# Cache
cache_enabled = true
cache_ttl_seconds = 3600
cache_temperature_max = 0.3

# Agent
agent_max_turns = 8
agent_tool_concurrency = 4
plan_tool_enabled = false             # feature-flagged decomposition tool

# RAG
embed_cache_enabled = true
embed_cache_ttl_days = 30
```

### Example Profiles

**Code review mode** — `~/.config/gemma/profiles/code.toml`
```toml
model = "gemma4:12b"
system_prompt = "You are a senior code reviewer. Output only code unless explicitly asked for prose."
temperature = 0.2
memory_top_k = 3
```

**Fast mode** — `~/.config/gemma/profiles/fast.toml`
```toml
model = "gemma4:e4b"
ollama_keep_alive = "-1"   # pin model in RAM permanently
memory_top_k = 2
cache_enabled = true
```

**Teaching mode** — `~/.config/gemma/profiles/verbose.toml`
```toml
system_prompt = "Explain everything step by step with examples. Assume a junior developer audience."
temperature = 0.7
memory_top_k = 5
```

### Using Profiles

```bash
$ gemma ask "review this function" --profile code
$ gemma chat --profile verbose
$ gemma ask "what is O(n log n)?" --profile fast
```

CLI flags always override profile fields:
```bash
# Use fast profile but override the model for this one call
$ gemma ask "explain mergesort" --profile fast --model gemma4:12b
```

---

## Scripting & Output Modes

Three mutually exclusive output modes designed for shell pipelines:

### `--json` — Full Structured Object

```bash
$ gemma ask --json "list three sorting algorithms"
{
  "content": "Quicksort, Mergesort, Heapsort...",
  "model": "gemma4:e4b",
  "elapsed_ms": 1842,
  "cache_hit": false
}

# Chain with jq
$ gemma ask --json "suggest a variable name for user count" | jq -r '.content'
```

### `--only <field>` — Single Field, Naked

```bash
# Print only the content, no JSON wrapper
$ gemma ask --only content "write a one-line Python lambda for squaring"
lambda x: x ** 2

# Check elapsed time
$ gemma ask --only elapsed_ms "hello"
1203

# Use in scripts
$ MSG=$(gemma ask --only content "summarize this PR in 10 words")
$ echo "PR summary: $MSG"
```

### `--code` — Code Blocks Only

Extracts and emits only the content inside fenced code blocks. Falls back to full content if no blocks are found.

```bash
# Generate and execute a bash one-liner
$ gemma ask --code "write a bash command to count unique IPs in access.log" | sh

# Generate and save to file
$ gemma ask --code "write a Python script to parse CSV and print row count" > count.py
$ python count.py data.csv

# Generate a config file
$ gemma ask --code "write a minimal nginx config for a reverse proxy to port 3000" \
    > /etc/nginx/sites-available/app
```

---

## Graceful Degradation

gemma-cli degrades gracefully when infrastructure is unavailable — it never crashes due to missing dependencies.

| Condition | Behavior |
|---|---|
| Redis unreachable | Prints warning; falls back to in-memory turn list; no condensation |
| `--no-memory` flag | Skips all memory; pure stateless LLM |
| `nomic-embed-text` unavailable | Skips cosine retrieval; loads memories sorted by importance score |
| Bad JSON from condensation | Silently skips that condensation cycle; conversation continues |
| Redis full (256MB cap) | LRU eviction of least-recently-used keys |
| Tool call failure | Logged to audit trail; agent loop resumes with error context |

---

## Architecture Overview

```
L1: Entry Point
    gemma.main (Typer CLI)

L2: Command Handlers
    ask / chat / pipe
    memory (remember / forget / pin / context)
    shell (sh / why / install-shell / explain)
    git (commit / diff)
    rag (index / query / status / reset)
    tools (list / run / audit)

L3: Cross-Cutting Services
    Config         — tunable parameters, profile loading
    OutputMode     — streaming, Markdown render, JSON, code extraction
    ResponseCache  — SHA256-keyed, Redis-backed
    SessionHistory — JSON fallback when Redis is unavailable
    SafetyPolicy   — capability gating context
    Redaction      — secret scrubbing before Redis writes

L4: Domain Subsystems
    memory/        — MemoryManager, condensation, retrieval, context assembly
    tools/         — registry, dispatcher, audit, planning, builtins
    rag/           — indexer, manifest, vector store, retrieval
    agent/         — session cache, planner

L5: Infrastructure Clients
    client.chat    — Ollama HTTP streaming
    Embedder       — nomic-embed-text via Ollama
    MemoryStore    — Redis CRUD + vector storage
    RedisVectorStore — RAG backend (plain redis:7, no RedisSearch)
    tools.audit    — append-only JSONL log

L6: External Services
    Ollama         — chat completions + embeddings
    Redis          — memories, RAG chunks, response cache
    Filesystem     — workspace files, archive
    HTTPS          — web search (DuckDuckGo), net_fetch (allowlist)
```

### Token Budget Strategy

Context window is split: **75% for input** (system prompt + memories + turns + RAG) and **25% reserved for the model's response**. The context assembly layer trims memories and turns to fit within this budget before every call.

### Warm-Start Optimization

At CLI startup, two daemon threads pre-load both the chat model and the embedding model via 1-token probe calls to Ollama. This eliminates the 1–8 second cold-start tax on the first real request. Disable for scripted/CI invocations:

```bash
$ gemma ask --no-warm "quick one-shot query in a script"
```

---

## Quick-Reference Cheat Sheet

```bash
# ── Conversation ──────────────────────────────────────────────────
gemma ask "your question"                      # single-shot
gemma ask "question" --think                   # force reasoning
gemma chat                                     # interactive REPL
cat file | gemma pipe "analyze this"           # stdin pipeline

# ── Memory ────────────────────────────────────────────────────────
gemma remember "fact" --category pref -i 4    # seed a memory
gemma forget --match "outdated fact"           # soft-delete
gemma pin --match "critical preference"        # never expire
gemma context "auth JWT"                       # preview retrieval
gemma history memories --limit 50             # list memories

# ── Terminal assistant ────────────────────────────────────────────
gemma sh "description of command"             # generate shell cmd
gemma sh --no-exec "description"              # print only, safe
gemma explain src/auth.py                     # explain a file
gemma explain --cmd "awk '{print $2}' f"      # explain a command
gemma explain --error "ECONNREFUSED :6379"    # explain an error
gemma commit                                  # generate commit msg
gemma commit --apply                          # commit automatically
gemma diff HEAD~3 --overall                   # summarize changes
gemma why                                     # explain last failure

# ── RAG ───────────────────────────────────────────────────────────
gemma rag index .                             # index workspace
gemma rag query "where is X implemented?"     # semantic search
gemma rag status                              # index info

# ── Scripting ─────────────────────────────────────────────────────
gemma ask --json "..." | jq '.content'        # structured output
gemma ask --only content "..."                # raw content only
gemma ask --code "write a bash script" | sh  # execute generated code
gemma ask --no-cache "..."                   # bypass cache
gemma ask --profile code "review this"       # use a named profile
```
