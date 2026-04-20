# gemma-cli

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

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
- [Reducing Ollama memory footprint](#reducing-ollama-memory-footprint)
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
- [Profiles](#profiles)
- [Scripting mode](#scripting-mode)
- [Memory power-user commands](#memory-power-user-commands)
  - [remember](#remember)
  - [forget](#forget)
  - [pin](#pin)
  - [context](#context)
- [Terminal-assistant commands](#terminal-assistant-commands)
  - [sh](#sh)
  - [explain](#explain)
  - [commit](#commit)
  - [diff](#diff)
  - [why](#why)
  - [install-shell](#install-shell)
- [Configuration reference](#configuration-reference)
- [Secret redaction](#secret-redaction)
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
│   ├── main.py            CLI entry point — all subcommands registered here
│   ├── client.py          Ollama API wrapper (streaming + blocking)
│   ├── config.py          Config dataclass with all tunable parameters
│   ├── embeddings.py      nomic-embed-text wrapper (Embedder class)
│   ├── history.py         JSON fallback session history (~/.gemma_history.json)
│   ├── redaction.py       Secret-pattern redaction (AWS keys, tokens, JWTs, …)
│   ├── commands/
│   │   ├── __init__.py    Package init
│   │   ├── shell.py       sh, why, install-shell commands
│   │   ├── explain.py     explain command (file / stdin / --cmd / --error)
│   │   └── git.py         commit, diff commands
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
│   ├── test_redaction.py
│   ├── test_memory_models.py
│   ├── test_condensation.py
│   ├── test_store.py
│   ├── test_retrieval.py
│   ├── test_context.py
│   ├── test_manager.py
│   ├── test_cmd_shell.py
│   ├── test_cmd_explain.py
│   └── test_cmd_git.py
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
| `gemma3:4b-it-q4_K_M` | ~3–4 GB | Comfortable | Comfortable | Comfortable |
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
7. Pulls `gemma3:4b-it-q4_K_M` (~3–4 GB) and `nomic-embed-text` (~274 MB) into Ollama
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
6. Pulls `gemma3:4b-it-q4_K_M` and `nomic-embed-text` into Ollama
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
ollama pull gemma3:4b-it-q4_K_M
ollama pull nomic-embed-text
```

**Windows (PowerShell):**
```powershell
git clone <repo-url> gemma-cli
cd gemma-cli
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[memory]"
ollama pull gemma3:4b-it-q4_K_M
ollama pull nomic-embed-text
```

---

## Reducing Ollama memory footprint

gemma-cli's defaults already lean memory-conscious (q5 weights, 16k context, 2-minute keep-alive — see [Configuration reference](#configuration-reference)). A few more levers live **on the Ollama server side** and can't be set from the CLI or a profile — they need to be exported in whatever launches `ollama serve` (LaunchAgent on macOS, shell rc, or systemd unit on Linux).

| Env var | Value | Effect |
|---|---|---|
| `OLLAMA_FLASH_ATTENTION` | `1` | Required before `OLLAMA_KV_CACHE_TYPE` is honoured. Enables Flash-Attention kernels. |
| `OLLAMA_KV_CACHE_TYPE` | `q8_0` (or `q4_0`) | Quantizes the KV cache. `q8_0` halves KV memory with no measurable quality loss; `q4_0` quarters it with a small hit on long-context recall. |
| `OLLAMA_NUM_PARALLEL` | `1` | Serialize requests instead of running up to 4 slots in parallel. Each slot duplicates the KV cache. |
| `OLLAMA_MAX_LOADED_MODELS` | `1` | Don't hold the chat and embedding models resident at the same time. Matters on ≤16 GB machines when `warm_start = true`. |

### macOS (menu-bar app)

The Ollama desktop app reads env vars from `launchctl`:

```bash
launchctl setenv OLLAMA_FLASH_ATTENTION 1
launchctl setenv OLLAMA_KV_CACHE_TYPE q8_0
launchctl setenv OLLAMA_NUM_PARALLEL 1
launchctl setenv OLLAMA_MAX_LOADED_MODELS 1
# Quit and relaunch Ollama from the menu bar for the vars to take effect.
```

### macOS / Linux (`ollama serve` manually)

Add to `~/.zshrc` or `~/.bashrc`:

```bash
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
```

Then restart the daemon (`pkill ollama && ollama serve`) to pick them up.

### Linux (systemd)

```bash
sudo systemctl edit ollama
```
Add:
```ini
[Service]
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
```
Then `sudo systemctl restart ollama`.

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

# Keep the model loaded for 2 hours (faster cold start on repeat calls)
gemma ask "What did we decide about the auth layer?" --keep-alive 2h
```

| Flag | Default | Description |
|---|---|---|
| `--model`, `-m` | `gemma3:4b-it-q4_K_M` | Override the Ollama model for this call |
| `--system`, `-s` | (config default) | Override the system prompt |
| `--no-stream` | off | Collect the full response, then render as Markdown |
| `--no-memory` | off | Skip memory retrieval and recording |
| `--think` | off | Enable Gemma 4 extended thinking mode |
| `--keep-alive` | `2m` | How long Ollama keeps the model in RAM after this call (`"2h"`, `"-1"` to pin, `"0"` to evict) |
| `--json` | off | Emit a JSON object (`content`, `model`, `elapsed_ms`, `cache_hit`) |
| `--only <field>` | — | Print a single JSON field naked (mutually exclusive with `--json`/`--code`) |
| `--code` | off | Extract and emit only fenced code blocks (mutually exclusive with `--json`/`--only`) |

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

# Pin the model in RAM for a long session (no eviction between turns)
gemma chat --keep-alive -1
```

Type `exit`, `quit`, or `:q` (or press `Ctrl-D`) to quit.

The status line shows the current model and memory mode:

```
gemma chat (gemma3:4b-it-q4_K_M, memory mode) -- type 'exit' or Ctrl-D to quit
```

Memory mode will show `degraded mode` if Redis is unreachable, and the CLI will still work using an in-memory fallback.

| Flag | Default | Description |
|---|---|---|
| `--model`, `-m` | `gemma3:4b-it-q4_K_M` | Override the Ollama model |
| `--system`, `-s` | (config default) | Override the system prompt |
| `--fresh` | off | Clear the raw sliding window before starting (condensed memories are kept) |
| `--no-memory` | off | Disable all memory features |
| `--think` | off | Enable Gemma 4 extended thinking mode |
| `--keep-alive` | `2m` | How long Ollama keeps the model in RAM between turns (`"2h"`, `"-1"` to pin, `"0"` to evict) |

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
| `--model`, `-m` | `gemma3:4b-it-q4_K_M` | Override the Ollama model |
| `--no-stream` | off | Collect and render as Markdown instead of streaming |
| `--json` | off | Emit a JSON object (`content`, `model`, `elapsed_ms`, `cache_hit`) |
| `--only <field>` | — | Print a single JSON field naked (mutually exclusive with `--json`/`--code`) |
| `--code` | off | Extract and emit only fenced code blocks (mutually exclusive with `--json`/`--only`) |

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

## Profiles

Named profiles let you define reusable combinations of model, system prompt, temperature, and other config fields so you can switch between them without repeating CLI flags.

### Profile files

Profiles live at `~/.config/gemma/profiles/<name>.toml`. Each file is a flat TOML mapping whose keys match fields in the `Config` dataclass. Any field you omit falls back to the dataclass default.

```toml
# ~/.config/gemma/profiles/code.toml
model = "gemma4:12b"
system_prompt = "You are a senior code reviewer. Output only code unless explicitly asked for prose."
temperature = 0.2
memory_top_k = 3
```

### Activating a profile

Pass `--profile <name>` (or `-p <name>`) before any subcommand:

```bash
gemma --profile code ask "refactor this function"
gemma --profile verbose chat
gemma --profile fast pipe "summarize this"
```

### Resolution order

When a profile and a CLI flag both set the same field, the CLI flag wins:

| Priority | Source |
|---|---|
| 1 (highest) | Explicit CLI flag (e.g. `--model gemma4:27b`) |
| 2 | Profile TOML field |
| 3 (lowest) | `Config` dataclass default |

```bash
# Profile sets model=gemma4:12b but --model overrides it to gemma4:27b
gemma --profile code ask "review this" --model gemma4:27b
```

### Example profiles

Four starter profiles are included in [`examples/profiles/`](examples/profiles/):

| File | Purpose |
|---|---|
| `code.toml` | Code review: `gemma4:12b`, low temperature, concise output |
| `verbose.toml` | Teaching mode: detailed step-by-step explanations |
| `fast.toml` | Quick lookups: smallest model, pinned in RAM, minimal memory overhead |
| `low-memory.toml` | 16 GB machines: q4 quant, evict-on-idle, no embed preload, memory off |

Copy them to `~/.config/gemma/profiles/` to use:

```bash
mkdir -p ~/.config/gemma/profiles
cp examples/profiles/*.toml ~/.config/gemma/profiles/
```

Unknown TOML keys emit a `UserWarning` and are ignored — forward-compatible with future Config fields.

---

## Scripting mode

Three mutually exclusive output flags adapt `ask` and `pipe` for scripting workflows. They are opt-in — omitting all three gives the default Rich Markdown output.

### `--json`

Emit the full response as a single JSON object. Ideal for piping to `jq`.

```bash
gemma ask --json "list three sorting algorithms" | jq '.content'
gemma ask --json "what model are you?" | jq '{model, elapsed_ms}'
```

Output shape:

```json
{"content": "...", "model": "gemma3:4b-it-q4_K_M", "elapsed_ms": 1842, "cache_hit": false}
```

### `--only <field>`

Print the value of a single field, naked (no JSON wrapper). Valid fields: `content`, `model`, `elapsed_ms`, `cache_hit`.

```bash
# Capture just the response text
reply=$(gemma ask --only content "write a one-liner to list Python files")

# Measure latency
gemma ask --only elapsed_ms "ping"
```

### `--code`

Strip prose and emit only the fenced code blocks from the response. Use this to pipe generated code directly to a shell or file.

```bash
# Generate and run a bash one-liner
gemma ask --code "write a bash command to find the 10 largest files" | sh

# Save generated Python to a file
gemma ask --code "write a Python hello world script" > hello.py

# Works with pipe too
echo "sort a list in Python" | gemma pipe --code > sort.py
```

If the response contains no fenced blocks the full content is emitted as a fallback.

### Mutual exclusivity

`--json`, `--only`, and `--code` are mutually exclusive — passing more than one exits with an error.

```bash
gemma ask --json --code "…"
# error: --json, --only, and --code are mutually exclusive
```

---

## Memory power-user commands

These commands let you manipulate the memory store directly, without going through the conversation pipeline.

---

### remember

Seed a fact directly into the memory store.  The content is embedded with `nomic-embed-text` and saved as a `MemoryRecord` immediately — no conversation or condensation step required.  The assigned `memory_id` is printed to stdout.

```bash
# Store a plain fact (category: factual_context, importance: 4)
gemma remember "I prefer type annotations in all Python code."

# Store a task-state fact
gemma remember "Auth rewrite is in progress — using JWT, deadline end of month." --category task

# Store a critical instruction that should never expire
gemma remember "Never use rm -rf without explicit confirmation." --category instruction --importance 5

# Store a user preference
gemma remember "Preferred language: Python. Avoid shell scripts where possible." --category pref

# Store a correction at maximum importance
gemma remember "The staging Redis URL is redis://staging.internal:6379/1, not /0." --importance 5
```

The printed `memory_id` can be used with `pin` and `forget` to manage the memory later.

| Flag | Default | Description |
|---|---|---|
| `--category`, `-c` | `feat` | Category shortcut: `feat` (factual_context), `pref` (user_preference), `task` (task_state), `instruction`, `correction`, `relationship`, `tool_usage`. Full enum values also accepted. |
| `--importance`, `-i` | `4` | Importance tier 1 (trivial, 6-hour TTL) to 5 (critical, no expiry). |

---

### forget

Remove a memory from the active index.  The record is **soft-deleted**: the Redis hash is kept for audit purposes but the `superseded_by` field is set so the record is never retrieved or injected into future prompts.

Exactly one of the three target selectors must be provided.

```bash
# Forget by explicit ID (as printed by gemma remember)
gemma forget abc123-...

# Forget the most recently created memory (with confirmation)
gemma forget --last

# Forget the top-1 memory matching a query
gemma forget --match "staging Redis URL"

# Skip the confirmation prompt
gemma forget --last --force
gemma forget --match "old auth token" --force
```

Without `--force` the command prints the memory content and asks `Forget this? [y/N]` before proceeding.

| Flag | Default | Description |
|---|---|---|
| `--last` | off | Target the most recently created active memory |
| `--match QUERY` | — | Target the top-1 memory with the highest cosine similarity to QUERY |
| `--force`, `-f` | off | Skip the confirmation prompt |

---

### pin

Set `importance=5` on an existing memory so it never expires.  The record is re-saved with the new importance, which maps to `None` (no TTL) in the tier table and causes the key to persist indefinitely.

```bash
# Pin by explicit ID
gemma pin abc123-...

# Pin the top-1 memory matching a query
gemma pin --match "JWT deadline"
```

| Flag | Default | Description |
|---|---|---|
| `--match QUERY` | — | Target the top-1 memory with the highest cosine similarity to QUERY |

---

### context

Preview what memory context would be injected for a given query, without calling the model.  Useful for debugging why the model gave an unexpected answer or to verify that a recently seeded fact is retrievable.

```bash
gemma context "auth rewrite JWT"
gemma context "what did we decide about Redis?"
gemma context "preferred Python style"
```

Output is a Rich table with columns **Imp** (importance), **Category**, **Similarity** (cosine score), and **Content**, followed by the most recent turns in the current session's sliding window.

---

## Terminal-assistant commands

These six commands are **stateless by default** (no memory read or write) and designed to fit naturally into shell workflows. They all accept `--model` and `--keep-alive` overrides and behave correctly when stdout is not a TTY.

---

### sh

Translate a natural-language description into a single shell command.

```bash
# Print only — safe for piping or copying
gemma sh --no-exec "find all Python files modified in the last 24 hours"

# Interactive: shows the command in a panel, then asks "Run this? [y/N]"
gemma sh "list the 10 largest files in the current directory"

# Ask for an explanation comment above the command
gemma sh --no-exec --explain "compress all PNG files in ./img"

# Target a specific shell syntax
gemma sh --no-exec --shell zsh "set a local variable and echo it"

# Keep the model warm across repeated calls
gemma sh --no-exec --keep-alive 2h "show listening ports"
```

The command is generated at `temperature=0.2` to minimise hallucination. A small safety blocklist rejects patterns like `rm -rf /`, `mkfs`, `dd if=`, and the fork bomb `:(){:|:&};:` — the command is printed but never executed if it matches.

| Flag | Default | Description |
|---|---|---|
| `--no-exec` | off | Print the command only; never prompt to run |
| `--shell` | `$SHELL` | Target shell syntax: `bash`, `zsh`, `sh` |
| `--explain` | off | Prepend a `#` comment describing what the command does |
| `--model`, `-m` | `gemma3:4b-it-q4_K_M` | Override the Ollama model |
| `--keep-alive` | `2m` | Ollama model-residency duration |

---

### explain

Explain text, a file, a shell command, or an error message in plain English.

Input source is auto-detected (priority: file argument > `--cmd` > `--error` > stdin):

```bash
# Explain a file
gemma explain error.log
gemma explain main.py --lines 40        # first 40 lines only

# Explain a shell command (never executed)
gemma explain --cmd "find . -name '*.py' -mtime -1 -exec grep -l TODO {} +"

# Explain an error message
gemma explain --error "EACCES: permission denied, open '/etc/hosts'"

# Pipe in a stack trace
cat traceback.txt | gemma explain

# Opt into memory retrieval for repo-aware context
gemma explain --cmd "git rebase -i HEAD~5" --with-memory
```

| Flag | Default | Description |
|---|---|---|
| `--cmd` | — | Shell command to explain (not executed) |
| `--error` | — | Error string or exception message to explain |
| `--lines`, `-n` | all (≤20 KB) | For file mode: read only the first N lines |
| `--with-memory` | off | Retrieve relevant Redis context before answering |
| `--no-stream` | off | Collect then render as Markdown |
| `--model`, `-m` | `gemma3:4b-it-q4_K_M` | Override the Ollama model |
| `--keep-alive` | `2m` | Ollama model-residency duration |

---

### commit

Generate a [conventional-commits](https://www.conventionalcommits.org/) message from staged changes.

```bash
# Stage some changes first, then generate a message for review
git add src/auth.py tests/test_auth.py
gemma commit

# Create the commit automatically
gemma commit --apply

# Force the commit type
gemma commit --apply --type feat

# Diffs larger than 20 KB are truncated automatically with a warning
gemma commit --apply
```

The message is generated at `temperature=0.2`. Output format:

```
<type>(<scope>): <subject>

<optional body>
```

Without `--apply` the message is printed for review and you can copy it manually. With `--apply`, `git commit -m <subject> [-m <body>]` is run directly.

| Flag | Default | Description |
|---|---|---|
| `--apply` | off | Create the commit after generating the message |
| `--type` | (model decides) | Force a conventional-commit type: `feat`, `fix`, `chore`, `docs`, … |
| `--model`, `-m` | `gemma3:4b-it-q4_K_M` | Override the Ollama model |
| `--keep-alive` | `2m` | Ollama model-residency duration |

> **Note:** if `commit.gpgsign = true` is set in your git config, `git commit` will invoke your signing key as normal. `gemma commit` does not bypass signing.

---

### diff

Summarize `git diff` output in plain English.

```bash
# Summarize unstaged working-tree changes (per file)
gemma diff

# Summarize staged changes
gemma diff --staged

# Summarize changes against a specific ref
gemma diff HEAD~3
gemma diff main..feature/new-auth

# One-paragraph summary of the whole diff instead of per-file
gemma diff --overall
gemma diff HEAD~1 --overall
```

Per-file output (default) looks like:

```
src/auth.py — extracts JWT validation into a standalone helper function.
tests/test_auth.py — adds three unit tests covering the new JWT helper.
```

| Flag | Default | Description |
|---|---|---|
| `--staged` / `--cached` | off | Diff the staging area instead of the working tree |
| `--overall` | off | One prose paragraph instead of per-file summaries |
| `--model`, `-m` | `gemma3:4b-it-q4_K_M` | Override the Ollama model |
| `--keep-alive` | `2m` | Ollama model-residency duration |

---

### why

Explain why the last shell command failed.  Requires `install-shell` to have
been run and sourced at least once.

```bash
# After a command fails, just run:
gemma why
```

`why` reads `$GEMMA_LAST_FILE` (default `~/.gemma_last_cmd`), which the shell hook
writes after every prompt.  The file contains the exit code and the command that
was run.  Gemma explains the most likely cause and suggests a concrete fix.

```bash
# Override the last-command file path (useful for testing)
gemma why --last-file /tmp/my_last_cmd
```

| Flag | Default | Description |
|---|---|---|
| `--last-file` | `~/.gemma_last_cmd` | Path to the last-command record (also read from `$GEMMA_LAST_FILE`) |
| `--model`, `-m` | `gemma3:4b-it-q4_K_M` | Override the Ollama model |
| `--keep-alive` | `2m` | Ollama model-residency duration |

---

### install-shell

Print (or append) a shell hook that records the last command and its exit code
after every prompt, enabling `gemma why`.

```bash
# Print the bash snippet — source it manually
gemma install-shell --shell bash

# Print the zsh snippet
gemma install-shell --shell zsh

# Append directly to your rc file (backs up the original first)
gemma install-shell --shell bash --append ~/.bashrc
gemma install-shell --shell zsh  --append ~/.zshrc

# After appending, reload your shell
source ~/.bashrc   # or ~/.zshrc
```

The snippet sets `$GEMMA_LAST_FILE` and installs a hook (`PROMPT_COMMAND` for
bash, `add-zsh-hook precmd` for zsh) that writes `<exit_code>\t<command>` to
that file after every command.

A backup of the original rc file is created at `<rc_file>.gemma-backup` before
any modification.

| Flag | Default | Description |
|---|---|---|
| `--shell` | `$SHELL` | Target shell: `bash` or `zsh` |
| `--append` | — | Append the snippet to this rc file path |

---

## Configuration reference

All settings live in `gemma/config.py` as a plain dataclass. CLI flags override individual values at runtime. Named TOML profiles (see [Profiles](#profiles)) can set any field persistently without editing code.

| Field | Default | Description |
|---|---|---|
| `model` | `gemma3:4b-it-q4_K_M` | Ollama model tag for chat and condensation |
| `system_prompt` | `"You are a helpful assistant."` | Default system message |
| `temperature` | `0.7` | Sampling temperature for chat (condensation always uses 0.2) |
| `context_window` | `16384` | Total token budget; 75% is used for input, 25% reserved for response |
| `history_file` | `~/.gemma_history.json` | Path for the JSON fallback session history |
| `ollama_host` | `http://localhost:11434` | Ollama server base URL |
| `memory_enabled` | `True` | Master switch; set `False` to disable all Redis/embedding features |
| `thinking_mode` | `False` | Enable Gemma 4 extended thinking; the model reasons step-by-step before responding |
| `ollama_keep_alive` | `"2m"` | How long Ollama keeps the model loaded in RAM between calls. Accepts duration strings (`"2m"`, `"2h"`) or `"-1"` to pin indefinitely / `"0"` to evict immediately. Overridable per-call with `--keep-alive`. |
| `cache_enabled` | `True` | Master switch for the SHA-keyed response cache |
| `cache_ttl_seconds` | `3600` | Cache entry TTL in seconds; `0` disables all caching |
| `cache_temperature_max` | `0.3` | Skip caching for calls with temperature above this value |
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

## Response cache

gemma-cli caches full LLM responses in Redis for deterministic, low-temperature calls. Repeated invocations with the same inputs (e.g. `gemma commit` on the same staged diff) skip the Ollama round-trip entirely and return in under 20 ms.

### How it works

The cache key is a SHA256 hash of the call inputs:

```text
sha256(model + "\0" + temperature + "\0" + system_prompt + "\0" + user_prompt + "\0" + keep_alive)
```

On a cache miss the response is stored in Redis as:

```text
gemma:cache:<sha256-hex>  →  {"content": "...", "created_at": <unix_ts>}
```

Entries expire after `cache_ttl_seconds` (default 1 hour). Setting `cache_ttl_seconds = 0` disables caching entirely.

### Which commands are cached

| Command | Default temperature | Cached by default? |
|---|---|---|
| `commit` | 0.2 | Yes |
| `sh` | 0.2 | Yes |
| `diff` | 0.3 | Yes |
| `ask --no-stream` | 0.7 | No (above threshold) |
| `pipe --no-stream` | 0.7 | No (above threshold) |
| `explain --no-stream` | 0.7 | No (above threshold) |

Streaming calls (`ask`, `pipe`, `explain` without `--no-stream`) are **never** cached — they are interactive and higher-temperature. To enable caching for `ask`/`pipe`/`explain`, lower the temperature via a profile (e.g. `temperature = 0.2`) and pass `--no-stream`.

### Bypass flags

All cached commands accept two flags:

| Flag | Effect |
|---|---|
| `--no-cache` | Bypass the cache entirely; always call the model and skip storing the result |
| `--cache-only` | Error out if there is no cache hit; never call the model (useful for asserting cached state in scripts) |

```bash
# Force a fresh Ollama call, ignore any cached result
gemma commit --no-cache

# Assert the response is already cached (exit 1 if not)
gemma commit --cache-only

# Bypass caching for a single diff summary
gemma diff --no-cache HEAD~1
```

### Disabling the cache

Set `cache_enabled = false` in a TOML profile, or set `cache_ttl_seconds = 0`:

```toml
# ~/.config/gemma/profiles/nocache.toml
cache_enabled = false
```

```bash
gemma --profile nocache commit
```

---

## Secret redaction

Developers routinely paste configuration snippets, stack traces, and log files into the CLI.  Any of those inputs might contain credentials.  Once a turn is condensed into a Redis `MemoryRecord` it can persist for days — so gemma-cli scrubs secrets out of every turn **before** it reaches the store.

### Redaction mechanism

Redaction runs automatically inside `MemoryManager.record_turn()` on every `ask` and `chat` turn, both the user message and the model reply.  It is transparent: the conversation continues normally, and the model still sees that a value was present (the redaction marker preserves context), just not what the value was.

Terminal-assistant commands (`sh`, `explain`, `commit`, `diff`, `why`) are stateless and never write to Redis, so redaction does not apply to them.

### Replacement format

Each matched secret is replaced with a `[REDACTED:TYPE]` marker.  Structural context that helps with debugging is preserved:

| Input | After redaction |
|---|---|
| `AKIAIOSFODNN7EXAMPLE` | `[REDACTED:AWS_ACCESS_KEY]` |
| `Authorization: Bearer eyJhbGci…` | `Authorization: Bearer [REDACTED:BEARER_TOKEN]` |
| `GITHUB_TOKEN=ghp_abc123…` | `GITHUB_TOKEN=[REDACTED:ENV_SECRET]` |
| `-----BEGIN RSA PRIVATE KEY-----…` | `[REDACTED:PRIVATE_KEY]` |

### Patterns covered

| Type label | What it matches |
|---|---|
| `PRIVATE_KEY` | PEM private-key blocks (`BEGIN RSA/OPENSSH/EC/DSA/PRIVATE KEY … END …`) |
| `BEARER_TOKEN` | `Authorization: Bearer <token>` — any token ≥ 20 chars |
| `ENV_SECRET` | `.env`-style lines whose key name contains `TOKEN`, `SECRET`, `PASSWORD`, `PASSWD`, `PWD`, `APIKEY`, `API_KEY`, `ACCESS_KEY`, or `PRIVATE_KEY` |
| `AWS_ACCESS_KEY` | AWS access key IDs starting with `AKIA` or `ASIA` (20 chars) |
| `GITHUB_TOKEN` | GitHub personal / OAuth / server-to-server tokens (`ghp_`, `gho_`, `ghs_`, `ghu_`, `ghr_`) |
| `GITLAB_TOKEN` | GitLab personal access tokens (`glpat-…`) |
| `JWT` | Three-segment base64url strings matching the `eyJ…eyJ…` JWT header structure |

### Logging

When redaction fires, an `INFO`-level log entry records how many secrets were found and their types (e.g. `Redacted 2 secret(s) from user turn: types={'AWS_ACCESS_KEY', 'JWT'}`).  The raw matched strings are never logged or persisted — only the type labels are recorded.

### Scope and limitations

This is not a comprehensive DLP system.  It targets a small set of high-confidence structural patterns where false positives are rare.  False negatives are possible for credentials that do not match any of the patterns above (e.g. short random API keys with no known prefix, database connection strings).  It is a safety net, not a substitute for not pasting production credentials into a CLI tool.

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

Tests use `fakeredis` and `typer.testing.CliRunner` — no live Redis or Ollama instance is required.  The git command tests (`test_cmd_git.py`) create throwaway repositories in `tmp_path` via subprocess and require `git` to be installed.

```bash
# Install dev dependencies
pip install -e ".[memory,dev]"

# Run the full test suite (181 tests)
pytest tests/ -v

# Run specific modules
pytest tests/test_manager.py -v         # memory orchestration
pytest tests/test_cmd_shell.py -v       # sh / why / install-shell
pytest tests/test_cmd_explain.py -v     # explain
pytest tests/test_cmd_git.py -v         # commit / diff
pytest tests/test_redaction.py -v       # secret redaction
```

The suite covers:

| Area | Module |
|---|---|
| Secret redaction (7 pattern types) | `test_redaction.py` |
| Ollama client + keep-alive propagation | `test_client.py` |
| Redis store CRUD + embeddings | `test_store.py` |
| Cosine similarity retrieval | `test_retrieval.py` |
| Context assembly + token budget | `test_context.py` |
| Condensation prompt parsing | `test_condensation.py` |
| MemoryManager orchestration | `test_manager.py` |
| `gemma sh` / `gemma why` / `gemma install-shell` | `test_cmd_shell.py` |
| `gemma explain` (all input modes) | `test_cmd_explain.py` |
| `gemma commit` / `gemma diff` | `test_cmd_git.py` |
| JSON fallback history | `test_history.py` |

## License

This project is licensed under the [Apache License 2.0](LICENSE).
