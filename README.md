# gemma-cli

A local CLI for Google's Gemma 4 model (running via Ollama) with a **Redis-backed recursive summarization memory system** — giving a 4B parameter model cross-session long-term memory that effectively bypasses its token-window limit.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [The memory problem](#the-memory-problem)
  - [Dual-tier memory design](#dual-tier-memory-design)
  - [Data flow](#data-flow)
  - [Why Redis?](#why-redis)
  - [Redis data model](#redis-data-model)
  - [Project layout](#project-layout)
- [How the memory system improves Gemma](#how-the-memory-system-improves-gemma)
- [Scaling to larger Gemma models](#scaling-to-larger-gemma-models)
- [Installation](#installation)
  - [Quick setup (recommended)](#quick-setup-recommended)
  - [Quick setup — Windows](#quick-setup--windows)
  - [Manual setup](#manual-setup)
- [Running Redis](#running-redis)
  - [Native (macOS)](#native-macos)
  - [Native (Linux)](#native-linux)
  - [Native (Windows)](#native-windows)
  - [Docker (optional)](#docker-optional)
- [CLI Usage](#cli-usage)
  - [ask](#ask)
  - [chat](#chat)
  - [pipe](#pipe)
  - [history](#history)
- [Configuration reference](#configuration-reference)
- [Graceful degradation](#graceful-degradation)
- [Running tests](#running-tests)

---

## Overview

Gemma 4 E4B is fast and capable for a 4B model, but it has a fundamental constraint: it can only reason over what fits in its context window at call time. When you start a new session — or when a conversation grows long enough — all prior context is gone.

gemma-cli solves this with a two-level memory system:

1. **Sliding window** — the most recent N raw turns are always in context.
2. **Condensed memories** — when the window overflows, Gemma *reads its own conversation* and distills it into structured facts. Those facts are stored in Redis, embedded with `nomic-embed-text`, and semantically retrieved on future calls.

The result: Gemma remembers your preferences, ongoing projects, and corrections across sessions — without ever loading stale verbatim history into the prompt.

---

## Architecture

### The memory problem

Every LLM call is stateless. For a CLI tool that wants persistent memory, there are three naive approaches, each with a fatal flaw:

| Approach | Problem |
|---|---|
| Load the entire chat history on every call | Eventually exceeds the token limit; grows unboundedly |
| Summarize the whole history into one block | Cascading information loss — each summary loses detail |
| Store nothing, start fresh each session | No memory at all |

gemma-cli's approach is a **recursive self-summarization pipeline** that avoids all three problems:

- The raw window is fixed-size (default 8 turns), so token usage from raw history is bounded.
- Overflow turns are extracted into *structured*, *categorized* facts — not a single prose summary. Structured extraction preserves detail while compressing bulk.
- Memories are retrieved *selectively* by semantic similarity. Only what's relevant to the current query is loaded, so context stays focused regardless of how many total memories exist.

### Dual-tier memory design

```
┌─────────────────────────────────────────────────────────────┐
│                         CONTEXT WINDOW                       │
│                                                              │
│  ┌──────────────────────────────────┐                        │
│  │  System Prompt                   │  ← always present      │
│  │  + Relevant Memories Block       │  ← top-K by similarity │
│  ├──────────────────────────────────┤                        │
│  │  Turn N-7                        │                        │
│  │  Turn N-6                        │  ← sliding window      │
│  │  ...                             │    (last 8 raw turns)  │
│  │  Turn N (current)                │                        │
│  └──────────────────────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
                                           │ overflow (>8 turns)
                                           ▼
                              ┌─────────────────────┐
                              │  CondensationPipeline│
                              │  (Gemma, temp=0.2)   │
                              │                      │
                              │  Extracts:           │
                              │  [{content,          │
                              │    category,         │
                              │    importance}]      │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │  nomic-embed-text    │
                              │  768-dim vector      │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │  Redis               │
                              │  (hash + sorted set) │
                              └──────────────────────┘
```

**Condensation** runs in a background daemon thread immediately after the user receives a response, so it never blocks the conversation.

**Retrieval** runs at the start of each turn: the current query is embedded, then cosine similarity is computed against all stored memory embeddings (client-side, in NumPy). The top-K results above a similarity threshold are injected into the system prompt before the call.

**Reconsolidation** is triggered automatically when the total memory count exceeds 200. Gemma merges the full memory set into a shorter, deduplicated list — the "recursive" step in recursive summarization.

### Data flow

```
User sends message
        │
        ▼
1. Embed query (nomic-embed-text, 768-dim)
2. Cosine similarity search over all stored embeddings
3. Retrieve top-5 MemoryRecords above min_similarity=0.3
4. Get last 8 raw turns from Redis sliding window
5. Assemble: system_prompt + memory block + recent turns
6. Trim to 75% of context_window (token budget)
        │
        ▼
7. Stream to Gemma 4 E4B via Ollama → print response
        │
        ▼
8. record_turn("user", ...) and record_turn("assistant", ...)
        │
        ▼ (if window overflows, in background thread)
9.  Pop overflow turns from Redis list
10. Find semantically similar existing memories (for dedup context)
11. CondensationPipeline.condense_turns() → JSON extraction via Gemma
12. Embed each new MemoryRecord, save to Redis with TTL
13. Supersede any near-duplicate existing memories (cosine ≥ 0.7)
14. If memory count > 200 → reconsolidate()
```

### Why Redis?

Redis was chosen for three specific reasons that fit this workload:

**1. The data structure match is exact.**

The memory system needs three distinct access patterns simultaneously:
- A **bounded list** for the raw conversation window (`RPUSH` / `LRANGE` / `LTRIM`)
- **Hash maps** for structured memory records with many fields
- A **sorted set** for ranking memories by importance and retrieving the top-N

Redis provides all three natively and atomically. A relational database would require a schema and joins; a key-value store alone couldn't express the sorted index; a document store would be overkill.

**2. TTL-based automatic expiry.**

Low-importance memories (importance=1) should die in 6 hours; critical ones (importance=5) should persist indefinitely. Redis `EXPIRE` implements this at the key level with zero application-side logic — the memories simply cease to exist when they age out, and the LRU eviction policy (`allkeys-lru`) handles the case where the 256MB cap is hit before TTLs fire.

**3. Zero infrastructure overhead on a developer machine.**

The entire Redis instance runs in a 256MB Docker container alongside Gemma (9.6GB) and nomic-embed-text (274MB) on a 16GB Mac. There are no external services, no authentication to configure, no schema migrations. `docker compose up -d` is the entire setup.

**Why not SQLite / a plain JSON file?**

SQLite has no native sorted sets and no per-row TTLs. A JSON file would require loading the entire thing into memory on every turn and has no concurrency safety for the background condensation thread. Redis handles both concerns out of the box.

**Why not use RedisSearch / HNSW for vector similarity?**

RedisSearch requires the `redis-stack` image (a heavier derivative). The full similarity computation — 500 memories × 768 dimensions — fits in ~1.5 MB of RAM and runs in under 1 ms on CPU with NumPy's matrix multiply. Client-side cosine similarity keeps the dependency footprint minimal: plain `redis:7-alpine` is sufficient.

### Redis data model

| Key pattern | Redis type | Contents |
|---|---|---|
| `gemma:session:{sid}:turns` | List | JSON-serialized `ConversationTurn` objects (sliding window) |
| `gemma:session:{sid}:meta` | Hash | `last_active` timestamp |
| `gemma:memory:{mid}` | Hash | All `MemoryRecord` fields: content, category, importance, session_id, turn_range, source_summary, created_at, last_accessed, access_count, superseded_by |
| `gemma:memory:index` | Sorted Set | Active `memory_id` values scored by importance (used for top-K and full scans) |
| `gemma:memory:embed:{mid}` | String | Base64-encoded `float32` numpy array (768 dims for nomic-embed-text) |

**Memory categories** (`MemoryCategory` enum):

| Value | When to use |
|---|---|
| `user_preference` | Stated preferences (language, style, tooling choices) |
| `task_state` | In-progress work, project names, current goals |
| `factual_context` | Facts about the user's environment (OS, stack, team) |
| `instruction` | Explicit instructions given to the model |
| `correction` | A correction to something the model previously stated |
| `relationship` | People, teams, or organizational context |
| `tool_usage` | Specific commands, APIs, or tools the user relies on |

**TTL tiers by importance:**

| Importance | Meaning | TTL |
|---|---|---|
| 5 | Critical (identity, key preferences) | No expiry |
| 4 | High (ongoing project context) | 7 days |
| 3 | Medium (incidental facts) | 72 hours |
| 2 | Low (passing mentions) | 24 hours |
| 1 | Trivial | 6 hours |

**Embedding storage note:** Redis is configured with `decode_responses=True` to make all hash operations return Python `str` instead of raw `bytes`. This would corrupt binary embedding data. Embeddings are therefore base64-encoded before storage and decoded on retrieval — adding ~33% overhead (~4 KB → ~5.5 KB per vector), which is negligible for 768-float vectors.

### Project layout

```
gemma-cli/
├── gemma/
│   ├── main.py            CLI entry point — ask, chat, pipe, history subcommands
│   ├── client.py          Ollama API wrapper (streaming + blocking)
│   ├── history.py         JSON fallback session history (~/.gemma_history.json)
│   ├── config.py          Config dataclass with all tunable parameters
│   ├── embeddings.py      nomic-embed-text wrapper (Embedder class)
│   └── memory/
│       ├── __init__.py    Package exports
│       ├── models.py      ConversationTurn, MemoryRecord, MemoryCategory
│       ├── store.py       Redis backend (CRUD, embeddings, sliding window)
│       ├── condensation.py  Gemma self-summarization + reconsolidation
│       ├── retrieval.py   Cosine similarity semantic search
│       ├── context.py     Prompt assembly + token budget trimming
│       └── manager.py     Orchestrator — single entry point for CLI code
├── tests/
│   ├── conftest.py        Shared fixtures (fakeredis, sample data)
│   ├── test_client.py
│   ├── test_history.py
│   ├── test_memory_models.py
│   ├── test_condensation.py
│   ├── test_store.py
│   ├── test_retrieval.py
│   ├── test_context.py
│   └── test_manager.py
├── docker-compose.yml     Redis 7 Alpine (256MB, LRU, AOF persistence)
└── pyproject.toml
```

---

## How the memory system improves Gemma

Without the memory system, every invocation of `gemma ask` or `gemma chat` begins with a blank slate. The model has strong general reasoning but zero knowledge of you, your projects, or anything you've discussed before. With the memory system, several concrete behaviours change:

**Corrections stick.** When Gemma says something wrong and you correct it, that correction is stored as a `correction`-category memory with importance=5. It will be retrieved and injected any time a semantically related question comes up in future sessions — in any session, not just the current one. Without memory, you correct the same mistake repeatedly.

**Preferences compound.** Stated preferences (`user_preference` memories) accumulate over time. After a few sessions the model knows your preferred language, code style, tool choices, and conventions without you restating them. The more you use it, the more accurate the injected context becomes.

**Task continuity across sessions.** Long projects that span multiple conversations maintain their `task_state` context. The model arrives already knowing the architecture decisions you've made, what phase the project is in, and what was left unresolved last time — without you pasting a summary.

**Context is targeted, not bloated.** Because retrieval is semantic (cosine similarity against your current query), only *relevant* memories are injected. Asking about Python doesn't load memories about your Docker setup. The 75% context budget is spent on facts that actually matter for the current question.

**The model is used twice per overflow cycle.** This is an unusual property of the design: Gemma itself runs the condensation step. When the sliding window overflows, the same model that held the conversation reads those turns at `temperature=0.2` and distills them into structured JSON. The model is essentially processing its own memory — deciding what was important, what corrects what, and what deserves to survive. This means condensation quality is coupled to the chat model's instruction-following ability, which has a direct bearing on the next section.

---

## Scaling to larger Gemma models

Switching models requires a single change — either pass `--model` on the CLI or update `model` in `config.py`. Both chat inference **and** condensation use the same model tag, so one flag changes both roles simultaneously.

```bash
gemma chat --model gemma4:12b
gemma chat --model gemma4:27b
```

The model is used in two structurally different ways in this system, and a larger model affects each differently.

### Role 1: Chat inference

This is the standard use — answering questions with the assembled context. Larger models produce more coherent and nuanced responses, follow the system prompt more reliably, and handle longer injected memory blocks without losing track of the conversation. The improvement here is roughly what you'd expect from any model size comparison.

### Role 2: Condensation (where size matters most)

This is the less obvious impact. The condensation prompt asks the model to read a conversation and output a **bare JSON array** — no markdown fences, no preamble prose — with correct category enum values and calibrated importance scores. For a 4B model this is genuinely difficult:

| Condensation behaviour | Gemma 4 E4B (4B) | Gemma 4 12B | Gemma 4 27B |
|---|---|---|---|
| JSON-only output (no markdown fences) | Inconsistent — fallback parser often triggered | Reliable | Near-perfect |
| Correct category assignment | Can confuse `factual_context` vs `task_state` | Accurate | Accurate |
| Importance calibration (1–5) | Tends to cluster at 3–4; extremes underused | Well-calibrated | Well-calibrated |
| Detecting contradictions of existing memories | Misses subtle conflicts | Catches most | Catches nearly all |
| Reconsolidation quality (merging 200 → 50) | May over-merge or drop detail | Good deduplication | Preserves nuance well |

The fallback JSON parsing in `condensation.py` — bracket-matching, fence-stripping, prose-trimming — exists specifically because E4B is unreliable at structured output. With a 12B or 27B model those paths are rarely reached, which means fewer silently-dropped condensation batches and a higher-quality memory bank over time.

### Hardware requirements

The models share unified memory on Apple Silicon with `nomic-embed-text` (~274 MB) and Redis (~256 MB).

| Model | Approx. size (Q4) | 16 GB Mac | 24 GB Mac | 32 GB+ Mac |
|---|---|---|---|---|
| `gemma4:e4b` | ~3–4 GB | Comfortable | Comfortable | Comfortable |
| `gemma4:12b` | ~8 GB | Tight — leaves ~7 GB for OS + nomic | Comfortable | Comfortable |
| `gemma4:27b` | ~16 GB | Will page to swap — avoid | Marginal | Comfortable |

**On a 16 GB machine:** E4B is the intended default. The 12B model is feasible with Ollama's default Q4 quantisation but leaves little headroom for other applications. The 27B model will cause memory pressure; response latency will degrade noticeably from swap usage.

**On 24 GB+:** The 12B is the best practical upgrade — condensation quality improves substantially and the model still loads entirely in RAM alongside the embedding model. The 27B is usable on 32 GB machines.

### Condensation latency

Because condensation runs in a background thread, slower inference on larger models does not block the user. The condensation job finishes in the background while you continue chatting. The practical effect is that memories from a given overflow batch appear in Redis a few seconds later on a 12B model vs. near-instantly on E4B — which is imperceptible for a CLI tool.

The one case where latency surfaces directly is `reconsolidate()`, which is triggered synchronously when memory count exceeds 200. On a 27B model that pass may take 30–60 seconds; on E4B it is typically under 5 seconds. For most users this threshold is not reached quickly, but it is worth noting if you plan to use the tool heavily with a large model.

---

## Installation

### Quick setup (recommended)

`setup.sh` handles everything automatically — Python, Redis, Ollama, models, and tests. Docker is **not required**.

```bash
git clone <repo-url> gemma-cli
cd gemma-cli
chmod +x setup.sh
./setup.sh
```

**What the script does:**
1. Detects your OS (macOS or Linux)
2. Installs Python ≥ 3.10 if missing (`brew` / `apt-get` / `dnf`)
3. Creates and activates a `.venv` virtual environment
4. Installs gemma-cli with all dependencies (`pip install -e ".[memory,dev]"`)
5. Ensures Redis is running — installs natively (`brew install redis` on macOS, `apt-get`/`dnf` on Linux); falls back to Docker only if Docker is already available and a native install cannot be performed
6. Installs Ollama if missing (`brew install ollama` on macOS, official install script on Linux)
7. Pulls `gemma4:e4b` (~3–4 GB) and `nomic-embed-text` (~274 MB) into Ollama
8. Runs the full test suite to confirm everything is wired up

```bash
# Skip model pulls on slow connections
./setup.sh --skip-models

# Skip the test suite
./setup.sh --skip-tests
```

Re-running is safe — every step is idempotent and skips work already done.

### Quick setup — Windows

`setup.ps1` is the PowerShell equivalent for Windows. It uses `winget` or Chocolatey to install missing dependencies.

```powershell
git clone <repo-url> gemma-cli
cd gemma-cli
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1
```

**What the script does:**
1. Installs Python ≥ 3.10 if missing (via `winget` or `choco`)
2. Creates and activates a `.venv` virtual environment
3. Installs gemma-cli with all dependencies
4. Ensures Redis is running — installs via `winget`/Chocolatey or [Memurai](https://www.memurai.com/) (Redis-compatible for Windows); falls back to Docker if already available
5. Installs Ollama if missing (via `winget` or `choco`)
6. Pulls `gemma4:e4b` and `nomic-embed-text` into Ollama
7. Runs the full test suite

```powershell
# Skip model pulls
.\setup.ps1 -SkipModels

# Skip tests
.\setup.ps1 -SkipTests
```

### Manual setup

**Prerequisites:** Python ≥ 3.10, [Ollama](https://ollama.com) installed and running, Redis (native or Docker).

**macOS / Linux:**
```bash
git clone <repo-url> gemma-cli
cd gemma-cli
python -m venv .venv
source .venv/bin/activate
pip install -e ".[memory]"
ollama pull gemma4:e4b
ollama pull nomic-embed-text
```

**Windows (PowerShell):**
```powershell
git clone <repo-url> gemma-cli
cd gemma-cli
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[memory]"
ollama pull gemma4:e4b
ollama pull nomic-embed-text
```

---

## Running Redis

The setup scripts start Redis automatically. If you are managing it manually, use whichever approach matches your setup.

### Native (macOS)

```bash
brew install redis
brew services start redis

# Stop
brew services stop redis
```

### Native (Linux)

```bash
# Debian/Ubuntu
sudo apt-get install redis-server
sudo systemctl enable --now redis-server

# Fedora/RHEL
sudo dnf install redis
sudo systemctl enable --now redis

# Stop
sudo systemctl stop redis-server   # Debian/Ubuntu
sudo systemctl stop redis          # Fedora/RHEL
```

### Native (Windows)

There is no official Redis server for Windows. Two options:

**Option A — [tporadowski/redis](https://github.com/tporadowski/redis/releases)** (community Windows port):
```powershell
winget install Redis.Redis
# Starts as a Windows service automatically

# Stop
Stop-Service redis
```

**Option B — [Memurai](https://www.memurai.com/)** (Redis-compatible, native Windows):
```powershell
winget install Memurai.MemuraiDeveloper
# Runs as a service on port 6379

# Stop
Stop-Service memurai
```

`setup.ps1` tries both automatically via `winget` or Chocolatey.

### Docker (optional)

A `docker-compose.yml` is included if you prefer containerised Redis (requires Docker Desktop).

```bash
# Start Redis in the background (data persists in a Docker volume)
docker compose up -d

# Verify it is healthy
docker compose ps

# Stop Redis (data is preserved)
docker compose down

# Stop Redis AND wipe all memory data
docker compose down -v
```

In all cases Redis is expected on `localhost:6379`. The Docker image is configured with a 256 MB memory cap, LRU eviction, and AOF persistence so memories survive restarts.

---

## CLI Usage

### ask

Send a single prompt. Relevant memories are retrieved and injected into context unless `--no-memory` is passed. Both the user turn and the model reply are recorded for future context.

```bash
gemma ask "What is the capital of France?"

# Enable extended thinking (Gemma 4 reasons before responding)
gemma ask "Solve this step by step: if 3x + 7 = 22, what is x?" --think

# Override the model for this call
gemma ask "Explain recursion" --model gemma3:8b

# Use a custom system prompt
gemma ask "Summarize this" --system "You are a concise technical writer."

# Disable streaming (prints Markdown-rendered response at the end)
gemma ask "Write a poem" --no-stream

# Skip memory retrieval (still prints a response, just no context injection)
gemma ask "Quick question" --no-memory
```

| Flag | Default | Description |
|---|---|---|
| `--model`, `-m` | `gemma4:e4b` | Override the Ollama model for this call |
| `--system`, `-s` | (config default) | Override the system prompt |
| `--no-stream` | off | Collect the full response, then render as Markdown |
| `--no-memory` | off | Skip memory retrieval and recording |
| `--think` | off | Enable Gemma 4 extended thinking mode |

---

### chat

Interactive REPL with full memory. Each turn retrieves relevant memories, records the exchange, and triggers background condensation when the window overflows.

```bash
gemma chat

# Enable extended thinking for the whole session
gemma chat --think

# Start a fresh sliding window (retains cross-session condensed memories)
gemma chat --fresh

# Disable all memory features for this session
gemma chat --no-memory

# Use a specific model
gemma chat --model gemma3:8b
```

Type `exit`, `quit`, or `:q` (or press `Ctrl-D`) to quit.

The status line shows the current model and memory mode:

```
gemma chat (gemma4:e4b, memory mode) -- type 'exit' or Ctrl-D to quit
```

Memory mode will show `degraded mode` if Redis is unreachable, and the CLI will still work using an in-memory fallback.

| Flag | Default | Description |
|---|---|---|
| `--model`, `-m` | `gemma4:e4b` | Override the Ollama model |
| `--system`, `-s` | (config default) | Override the system prompt |
| `--fresh` | off | Clear the raw sliding window before starting (condensed memories are kept) |
| `--no-memory` | off | Disable all memory features |
| `--think` | off | Enable Gemma 4 extended thinking mode |

---

### pipe

Read from stdin and pass it to Gemma together with an instruction. Stateless — no memory read or write.

```bash
# Summarize a file
cat notes.txt | gemma pipe "Summarize this in three bullet points"

# Analyze code
cat main.py | gemma pipe "Explain what this script does"

# Default instruction is "Analyze this input."
git diff | gemma pipe

# Disable streaming
cat report.md | gemma pipe "Extract action items" --no-stream
```

| Flag | Default | Description |
|---|---|---|
| `--model`, `-m` | `gemma4:e4b` | Override the Ollama model |
| `--no-stream` | off | Collect and render as Markdown instead of streaming |

---

### history

Subcommands for inspecting and managing memory state.

#### `history show`

Print the raw JSON fallback session history (turns stored in `~/.gemma_history.json`).

```bash
gemma history show
```

#### `history clear`

Wipe the raw session history and the Redis sliding window. Condensed memories in Redis are preserved.

```bash
gemma history clear
```

#### `history memories`

List condensed memories currently stored in Redis, sorted by importance (highest first).

```bash
gemma history memories

# Show up to 100 memories (default is 50)
gemma history memories --limit 100
```

Output is a Rich table with columns: **Importance**, **Category**, **Content**.

#### `history stats`

Print statistics about the memory system.

```bash
gemma history stats
```

Example output:

```
session: a3f1c2d9e4b5...
active memories: 14
window turns: 6

by category:
  factual_context: 5
  task_state: 4
  user_preference: 3
  correction: 2

by importance:
  5: 3
  4: 6
  3: 4
  2: 1
```

---

## Configuration reference

All settings live in `gemma/config.py` as a plain dataclass. CLI flags override individual values at runtime; there is no config file — edit the dataclass defaults or subclass `Config` to change persistent behaviour.

| Field | Default | Description |
|---|---|---|
| `model` | `gemma4:e4b` | Ollama model tag for chat and condensation |
| `system_prompt` | `"You are a helpful assistant."` | Default system message |
| `temperature` | `0.7` | Sampling temperature for chat (condensation always uses 0.2) |
| `context_window` | `128000` | Total token budget; 75% is used for input, 25% reserved for response |
| `history_file` | `~/.gemma_history.json` | Path for the JSON fallback session history |
| `ollama_host` | `http://localhost:11434` | Ollama server base URL |
| `memory_enabled` | `True` | Master switch; set `False` to disable all Redis/embedding features |
| `thinking_mode` | `False` | Enable Gemma 4 extended thinking; the model reasons step-by-step before responding |
| `redis_url` | `redis://localhost:6379/0` | Redis connection string |
| `embedding_model` | `nomic-embed-text` | Ollama model used for 768-dim embeddings |
| `sliding_window_size` | `8` | Number of raw turns kept in the Redis list |
| `memory_top_k` | `5` | Number of memories retrieved per turn |
| `memory_min_similarity` | `0.3` | Cosine similarity threshold for retrieval (0–1) |
| `memory_conflict_threshold` | `0.7` | Cosine similarity above which a new memory supersedes an old one |
| `memory_max_count` | `200` | Total active memories before reconsolidation is triggered |
| `condensation_async` | `True` | Run condensation in a background thread (non-blocking) |
| `ttl_map` | See table above | Per-importance TTL seconds; `None` = no expiry |

---

## Graceful degradation

The CLI is designed to work even when infrastructure is missing:

| Situation | Behaviour |
|---|---|
| Redis unreachable | Prints a one-time yellow warning; falls back to in-memory turn list; no condensation |
| `--no-memory` flag | Memory system skipped entirely; works like a stateless chatbot |
| nomic-embed-text unavailable | Retrieval skipped; memories loaded by importance score instead |
| Condensation returns invalid JSON | Silently skipped; the overflow turns are discarded but conversation continues |
| Redis full (256MB cap hit) | LRU eviction removes least-recently-used keys automatically |

You can always start with `gemma chat --no-memory` and add infrastructure later without changing anything else.

---

## Running tests

Tests use `fakeredis` — no live Redis or Ollama instance is required.

```bash
# Install dev dependencies
pip install -e ".[memory,dev]"

# Run the full test suite
pytest tests/ -v

# Run a specific module
pytest tests/test_manager.py -v
pytest tests/test_context.py -v
```

The suite covers: models, store CRUD, retrieval cosine math, context assembly + budget trimming, condensation prompt parsing, and MemoryManager orchestration (sliding window, condensation trigger, degraded mode, stats).
