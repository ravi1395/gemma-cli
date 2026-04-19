# gemma-cli — Implementation Plan

> A local CLI tool for interacting with Google's Gemma 4 model via Ollama.
> Target machine: Apple M5, 16GB unified memory, macOS

---

## Machine Context

| Spec     | Value                  | Notes                                      |
|----------|------------------------|--------------------------------------------|
| Chip     | Apple M5               | Apple Silicon — Metal GPU acceleration     |
| RAM      | 16GB unified           | CPU + GPU share this pool                  |
| OS       | macOS                  | Homebrew available                         |
| Disk     | ~342GB free            | Ample for model storage                    |
| Python   | 3.10+                  | Already installed                          |

**Chosen model: `gemma4:e4b`** (~9.6GB download, ~6GB active RAM)
Rationale: Fits comfortably in 16GB, leaves headroom for OS + other apps. E4B (4.5B effective params) offers strong quality/speed balance on M5. Upgrade path to 26B MoE if RAM headroom allows.

---

## Project Structure

```
gemma-cli/
├── gemma/
│   ├── __init__.py
│   ├── main.py          # Typer CLI entry point
│   ├── client.py        # Ollama API wrapper (streaming + blocking)
│   ├── history.py       # Chat session persistence (JSON)
│   └── config.py        # Defaults: model, system prompt, temp, etc.
├── tests/
│   ├── test_client.py
│   └── test_history.py
├── pyproject.toml       # Package definition + `gemma` command registration
├── requirements.txt
└── plan.md              # This file
```

---

## Phase 1 — Environment Setup

### 1.1 Install Ollama

```bash
brew install ollama
```

Verify:
```bash
ollama --version
```

### 1.2 Start Ollama Service

Ollama runs a local REST server on `http://localhost:11434`.

```bash
# Start in background (auto-starts on login after brew install)
ollama serve
```

### 1.3 Pull Gemma 4 E4B Model

```bash
ollama pull gemma4:e4b
```

This downloads ~9.6GB to `~/.ollama/models/`. One-time operation.

Smoke test:
```bash
ollama run gemma4:e4b "What is a transformer architecture?"
```

---

## Phase 2 — Python Project Scaffold

### 2.1 Dependencies

```bash
pip install typer ollama rich
```

| Library  | Purpose                                                     |
|----------|-------------------------------------------------------------|
| `typer`  | CLI framework — typed args, auto `--help`, subcommands      |
| `ollama` | Official Python client for Ollama's local REST API          |
| `rich`   | Pretty terminal output: markdown rendering, streaming, color|

### 2.2 `pyproject.toml`

Registers the `gemma` shell command globally after `pip install -e .`

```toml
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "gemma-cli"
version = "0.1.0"
dependencies = ["typer", "ollama", "rich"]

[project.scripts]
gemma = "gemma.main:app"
```

Install as editable (live-reloads on code changes):
```bash
pip install -e .
```

---

## Phase 3 — Core Modules

### 3.1 `config.py` — Central Configuration

```python
from dataclasses import dataclass

@dataclass
class Config:
    model: str = "gemma4:e4b"
    system_prompt: str = "You are a helpful assistant."
    temperature: float = 0.7
    context_window: int = 128000
    history_file: str = "~/.gemma_history.json"
```

### 3.2 `client.py` — Ollama API Wrapper

Key methods:
- `ask(prompt, system, stream=True) -> Generator` — single-shot query with streaming
- `chat(messages, stream=True) -> Generator` — multi-turn chat with history

Streaming is important: Gemma responses can be long; streaming prints tokens as they arrive rather than waiting for the full response.

```python
import ollama
from gemma.config import Config

def ask(prompt: str, config: Config, stream: bool = True):
    response = ollama.chat(
        model=config.model,
        messages=[
            {"role": "system", "content": config.system_prompt},
            {"role": "user",   "content": prompt},
        ],
        stream=stream,
        options={"temperature": config.temperature},
    )
    if stream:
        for chunk in response:
            yield chunk["message"]["content"]
    else:
        yield response["message"]["content"]
```

### 3.3 `history.py` — Chat Session Persistence

Stores conversation turns as JSON at `~/.gemma_history.json`.

```python
[
  {"role": "system",    "content": "You are a helpful assistant."},
  {"role": "user",      "content": "What is RLHF?"},
  {"role": "assistant", "content": "RLHF stands for..."}
]
```

Key methods:
- `load() -> list[dict]` — load history from disk
- `append(role, content)` — add a turn
- `save()` — persist to disk
- `clear()` — wipe session

### 3.4 `main.py` — CLI Entry Point

Four subcommands:

#### `gemma ask` — Single-shot query
```bash
gemma ask "Explain attention mechanisms"
gemma ask --model gemma4:e2b "Quick question"
gemma ask --no-stream "Give me a JSON response"
```

#### `gemma chat` — Interactive session with history
```bash
gemma chat
gemma chat --system "You are a senior Java engineer reviewing code"
gemma chat --fresh          # Ignore previous history, start clean
```

#### `gemma pipe` — Read from stdin (for scripting)
```bash
cat MyService.java | gemma pipe "Review this for SOLID violations"
cat error.log | gemma pipe "What is causing this error?"
```

#### `gemma history` — Manage sessions
```bash
gemma history show     # Print current session
gemma history clear    # Wipe session
```

---

## Phase 4 — CLI Usage Examples (Target UX)

```bash
# One-liners
gemma ask "What is softmax?"
gemma ask "List 5 Java design patterns" --no-stream > patterns.txt

# Piping
cat MyClass.java | gemma pipe "Review for code smells"
cat pom.xml      | gemma pipe "Any outdated dependencies here?"

# Interactive chat
gemma chat
gemma chat --system "You are a Python expert helping me learn LLMs"

# Model switching
gemma ask --model gemma4:e2b "Fast question"

# Session management
gemma history show
gemma history clear
```

---

## Phase 5 — Testing

### 5.1 Unit Tests

- `test_client.py` — mock `ollama.chat`, verify message formatting
- `test_history.py` — verify load/save/clear round-trips

```bash
pip install pytest
pytest tests/
```

### 5.2 Manual Smoke Tests

```bash
# Verify Ollama is running
curl http://localhost:11434/api/tags

# Single ask
gemma ask "Hello, who are you?"

# Pipe test
echo "def fib(n): return fib(n-1)+fib(n-2)" | gemma pipe "What is wrong with this code?"

# Chat session
gemma chat
```

---

## Phase 6 — Optional Enhancements (Future)

| Enhancement | Approach |
|-------------|----------|
| JSON output mode | `--json` flag, wrap output in `json.dumps` |
| Config file (`~/.gemmarc`) | TOML config, override defaults per-user |
| Multiple named sessions | `gemma chat --session work` vs `--session learning` |
| Token usage display | Ollama returns usage stats; surface with `--verbose` |
| Model auto-pull | If model missing, prompt to `ollama pull` automatically |
| Shell completions | `typer --install-completion` built-in |

---

## Implementation Order

1. [ ] Phase 1 — Install Ollama, pull `gemma4:e4b`, smoke test
2. [ ] Phase 2 — Scaffold project, install deps, register `gemma` command
3. [ ] Phase 3.1 — `config.py`
4. [ ] Phase 3.2 — `client.py` with streaming
5. [ ] Phase 3.3 — `history.py`
6. [ ] Phase 3.4 — `main.py` with all 4 subcommands
7. [ ] Phase 5 — Tests
8. [ ] Phase 6 — Optional enhancements

---

## Key Design Decisions

**Why Ollama over llama.cpp directly?**
Ollama wraps llama.cpp and adds model management, a clean REST API, and Apple Silicon Metal acceleration out of the box. Less configuration, same performance.

**Why Typer over Click or argparse?**
Typer uses Python type hints for CLI definition — closer to how a Java developer thinks about method signatures. Auto-generates `--help` docs. Click is the underlying engine.

**Why stream by default?**
Gemma 4 E4B can produce long responses. Streaming prints tokens as generated, giving immediate feedback rather than a blank terminal for 10+ seconds.

**Why JSON history over SQLite?**
Simplicity for a learning project. Easy to inspect, edit, or clear manually. SQLite would be the right call if adding named sessions or search.

---

## Phase 6 — Architectural Expansions

Four independent features, planned as separate documents under `docs/plans/`
with paired ADRs under `docs/adr/`. They share three new cross-cutting
modules (`gemma/platform.py`, `gemma/safety.py`, `gemma/chunking.py`) that
are built first so each feature reuses them instead of duplicating logic.

| # | Feature | Plan | ADR | Effort | Risk |
|---|---------|------|-----|--------|------|
| 6.1 | Sandboxed tool use | [docs/plans/phase-6.1-sandboxed-tools.md](docs/plans/phase-6.1-sandboxed-tools.md) | [docs/adr/0001-tool-sandboxing.md](docs/adr/0001-tool-sandboxing.md) | High | Medium |
| 6.2 | Local-directory RAG | [docs/plans/phase-6.2-local-rag.md](docs/plans/phase-6.2-local-rag.md) | [docs/adr/0002-local-rag.md](docs/adr/0002-local-rag.md) | High | Medium |
| 6.3 | Shell completions | [docs/plans/phase-6.3-completions.md](docs/plans/phase-6.3-completions.md) | [docs/adr/0003-shell-completions.md](docs/adr/0003-shell-completions.md) | Low | Low |
| 6.4 | Clipboard integration | [docs/plans/phase-6.4-clipboard.md](docs/plans/phase-6.4-clipboard.md) | [docs/adr/0004-clipboard.md](docs/adr/0004-clipboard.md) | Low | Low |

### Shared modules (build first)

- **`gemma/platform.py`** — OS + shell + TTY + SSH detection. Consumed by
  6.3 (picks rc-file path) and 6.4 (picks clipboard backend).
- **`gemma/safety.py`** — path allowlist, traversal/symlink-escape guards,
  `archive()` helper that moves a path to `<root>/archive/<ISO-ts>/…` per
  the project's never-delete rule. Consumed by 6.1 and reusable by the
  existing `git` and `shell` commands.
- **`gemma/chunking.py`** — AST chunking for `.py`, heading chunking for
  `.md`, sliding-window fallback for everything else. Consumed by 6.2 and
  reusable by memory condensation in future passes.

### Execution order

1. `platform.py` + `safety.py` + `chunking.py` (shared foundations).
2. **6.4 Clipboard** — smallest, validates the platform layer.
3. **6.3 Completions** — reuses platform layer, idempotent install.
4. **6.2 RAG** — largest reuse win once chunking lands.
5. **6.1 Sandboxed tools** — built on battle-tested `safety.py`.

### Safety invariant (applies across Phase 6)

> Gemma must never delete a file. The tool registry has no `delete`
> capability; the only destructive-looking tool is `archive`, which moves
> a path to `<cwd>/archive/<ISO-ts>/…`. This is enforced in code, not in
> prompts.
