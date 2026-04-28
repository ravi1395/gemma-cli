# Migrating from Ollama to LM Studio

gemma-cli 0.2 makes **LM Studio** the default LLM runtime. Ollama is still supported as an opt-in backend. This document explains what changed, why, and what an existing user has to do.

---

## TL;DR

1. `uv sync` (or `pip install -e ".[memory]"`) — picks up the LM Studio dependency automatically.
2. Open LM Studio and load a chat model (e.g. `mlx-community/gemma-4-E4B-it-4bit` on Apple Silicon).
3. Run `gemma model info` to confirm the resolved backend and model.
4. Existing profiles and Redis-backed memories keep working — the embedding dimension is unchanged.

If you want to keep using Ollama, add `backend = "ollama"` to your profile (or pass `--backend ollama`).

---

## What changed and why

| Before (≤0.1.x) | After (0.2+) |
|---|---|
| Ollama was the only backend; calls went directly through `ollama.Client`. | All chat + embedding calls go through `gemma.backends.LLMBackend`. The default implementation is `LMStudioBackend`; `OllamaBackend` is opt-in. |
| `Config.model = "gemma4:e2b"` (an Ollama tag). | `Config.model = None` by default; resolves to a HuggingFace `owner/repo` matched to your platform. |
| Build system: `setuptools`. | Build system: `hatchling`; environment management: `uv`. |
| No `model` subcommand. | New `gemma model {pull,list,use,info}` for HuggingFace model management. |
| `ollama_keep_alive = "2m"` directly controlled Ollama's eviction timer. | The same field exists; LM Studio's SDK TTL is derived from it (`"30m"` → `ttl=1800`, `"-1"` → `ttl=None`). |
| `--think` flag asked Ollama to expose a separate reasoning stream. | The flag now controls **rendering only**: the LM Studio SDK already classifies fragments as `reasoning` vs `none`, so we route them into the existing `("think", text)` / `("content", text)` tuples. |

The migration was driven by three goals:

1. **Better UX on Apple Silicon.** MLX builds run noticeably faster than llama.cpp on M-series chips; LM Studio ships first-class MLX support.
2. **Less install friction.** `uv sync` + an open LM Studio app is enough to run `gemma ask`. No more `ollama serve`, no GPU runtime to configure.
3. **Bring-your-own-model.** Pull anything from HuggingFace with `gemma model pull <repo>`.

---

## Migration steps

### 1. Update dependencies

```bash
git pull
uv sync                             # or: pip install -e ".[memory]"
```

### 2. Install LM Studio and load a model

Download from <https://lmstudio.ai/> and start the app. Either let gemma-cli's auto-default pick a model (the SDK auto-loads on first call), or pre-load via the LM Studio UI for the fastest first-run experience.

To download a specific model from the CLI:

```bash
# One-time: install the `lms` CLI bundled with LM Studio
npx lmstudio install-cli

# Then:
gemma model pull mlx-community/gemma-4-E4B-it-4bit
```

### 3. Confirm the backend resolution

```bash
gemma model info
```

Expected output on an M-series Mac:

```
backend                   lmstudio
chat model                mlx-community/gemma-4-E4B-it-4bit
embedding model           nomic-ai/nomic-embed-text-v1.5
platform default (chat)   mlx-community/gemma-4-E4B-it-4bit
platform default (embed)  nomic-ai/nomic-embed-text-v1.5
apple silicon             yes
lmstudio host             (SDK default: localhost:1234)
keep_alive / TTL          2m
```

### 4. (Optional) keep using Ollama

Add to your profile, or pass per-call:

```toml
# ~/.config/gemma/profiles/legacy.toml
backend = "ollama"
model = "gemma3:4b"
```

```bash
gemma --profile legacy ask "hello"
# or
gemma --backend ollama ask "hello"
```

The `ollama` package is installed as part of the dev group; for a production install that uses Ollama, install with the `ollama` extra:

```bash
pip install -e ".[memory,ollama]"
```

---

## Behaviour differences worth knowing

### Thinking mode

Ollama exposed `think=True` as an API parameter that returned thinking and content as two channels. The LM Studio SDK does this differently: every streaming fragment carries a `reasoning_type` literal. gemma-cli uses that literal to emit the same `("think", text)` / `("content", text)` tuples as before, so output rendering is unchanged.

The `--think` / `--no-think` flag and `thinking_mode` config field still exist; they now control whether the think stream is *rendered*, not whether the model reasons. Models trained with reasoning will still reason regardless.

### Embedding model identity

The embedding model identifier changed from `nomic-embed-text` (Ollama tag) to `nomic-ai/nomic-embed-text-v1.5` (HuggingFace path). The vector dimension is the same (768), so existing Redis memory entries remain queryable. If you regenerate them, they'll come out at the same dimension and be drop-in compatible.

### Keep-alive vs TTL

| Profile value | Ollama behaviour | LM Studio behaviour |
|---|---|---|
| `"2m"` | keep loaded for 2 minutes after last call | `ttl=120` |
| `"30m"` | 30 minutes | `ttl=1800` |
| `"-1"` | never evict | `ttl=None` |
| `"0"` | evict immediately | `ttl=0` |

The mapping is implemented in `gemma.backends.base.parse_keep_alive_seconds` — you don't have to think about it unless you're writing a backend yourself.

### `setup.sh` / `setup.ps1`

The legacy bootstrap scripts still exist but install Ollama. They're scheduled for replacement with `uv`-based equivalents. For now, prefer `uv sync` directly.

---

## Rollback

If something doesn't work and you want to fall back to the pre-0.2 behaviour wholesale:

```bash
git checkout v0.1.x   # or whatever tag predates the migration
uv sync
```

Open an issue with the failure mode — that's the path that lets us shrink the migration friction for the next person.
