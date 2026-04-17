# Plan: gemma-cli Terminal-Assistant — First Pass

## Context

`gemma-cli` currently has four top-level commands (`ask`, `chat`, `pipe`, `history`) and a Redis-backed memory system. The core is complete, tested (93/93 passing), and documented. The user wants to shift the tool from a general chat REPL into a **terminal-first assistant for working developers** — not vibe-coding whole apps, but addressing the high-frequency friction points of daily shell work.

**This first pass adds the five highest-value commands, a critical safety feature, and a one-line latency win. Everything else from the broader roadmap (memory power-user controls, profiles, scripting output modes, response caching, tool use, RAG) is deferred to follow-up passes.**

Dependency posture: stdlib preferred; a small number of well-maintained third-party packages may be added where they meaningfully reduce code (e.g. `tomli` for profile files in a later pass). This first pass is expected to add zero or one new dependencies.

---

## Design principles

1. **Stdin/stdout first.** Every new command behaves correctly in a pipe.
2. **Fast path for quick queries.** Short asks skip memory retrieval.
3. **No hidden side effects.** Any command that could mutate state (run a shell command, create a commit) confirms first.
4. **Reuse existing plumbing** — `Config`, `MemoryManager`, `client.chat()`, the Typer app.

---

## What this pass delivers

### 1. Secret redaction (`gemma/redaction.py`) — land first
Before any turn is `RPUSH`'d to Redis, a regex suite scans for common leakable patterns and replaces matches with `[REDACTED:TYPE]`.

Patterns covered in v1:
- AWS access keys (`AKIA[0-9A-Z]{16}`, `aws_secret_access_key` lines)
- GitHub / GitLab personal tokens (`ghp_…`, `glpat-…`, `gho_…`, `ghs_…`)
- Generic `Bearer <token>` in Authorization headers
- `-----BEGIN (RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----` blocks
- `.env`-shaped lines where the key name contains `TOKEN` / `SECRET` / `PASSWORD` / `KEY`
- JWT-shaped strings (`eyJ…` with two dots and ≥30 chars per segment)

API: `redact(text: str) -> tuple[str, list[RedactionFinding]]`. Called from `MemoryManager.record_turn()` before turns are persisted — current-session context is untouched.

### 2. Ollama keep-alive (`gemma/config.py`, `gemma/client.py`, `gemma/embeddings.py`)
Add `Config.ollama_keep_alive: str = "30m"`. Pass it as `keep_alive=` on every `ollama.Client.chat()` and `ollama.Client.embed()` call. Keeps both models resident in RAM across CLI invocations; second-invocation TTFT should drop from ~2 s to <300 ms.

Also expose `--keep-alive` as a CLI override (useful for long-lived shells: `--keep-alive 2h`).

### 3. `gemma sh "<prompt>"` — natural language → shell command
New file `gemma/commands/shell.py`. Single-command generator; no prose, no code fences (enforced by a tight system prompt at `temperature=0.2`).

Flow:
1. Build a small, focused prompt: "Output one shell command. No explanation, no fences."
2. Call the model (stateless — no memory read/write).
3. Post-process: strip whitespace, strip stray fences if the model slipped.
4. Print command, ask `Run this? [y/N]`.
5. If `y`, `subprocess.run(cmd, shell=True, executable=<detected shell>)`.

Flags:
- `--no-exec` — print only, never prompt (for piping: `gemma sh --no-exec "..." | pbcopy`)
- `--shell bash|zsh` — target syntax; defaults to `$SHELL`
- `--explain` — print a one-line `#` comment above the command

### 4. `gemma explain`
New file `gemma/commands/explain.py`. Mode auto-detected:
- If stdin is not a TTY → read stdin, explain (errors, stack traces, man pages, logs)
- If a file path is passed → read and explain the file
- `--cmd "..."` → explain a command without running it

Stateless by default. `--with-memory` opts into memory retrieval if the developer wants context applied.

### 5. `gemma commit`
New file `gemma/commands/git.py`. Steps:
1. Run `git diff --cached` via `subprocess.run`. Fail clearly if empty.
2. Feed the diff to a prompt that asks for a conventional-commits style message (one-line subject + optional body).
3. Print for review.
4. If `--apply`, run `git commit -m "<subject>" -m "<body>"`.

Guards:
- Refuses to run outside a git repo (`git rev-parse --git-dir`).
- Truncates diffs > 20 KB with a warning (to stay under token budget).
- Never auto-applies without `--apply`.

### 6. `gemma diff [refspec]`
In the same `gemma/commands/git.py`. Runs `git diff <refspec>` (default: unstaged working tree) and produces a plain-English summary grouped by file. Useful pre-PR sanity check.

### 7. `gemma why` + `gemma install-shell`
Shell-hook mechanism. `gemma install-shell --bash` (or `--zsh`) prints a shell snippet the user `source`s:

```bash
_gemma_record_last() {
  local ec=$?
  # Save last command + exit code + stderr snapshot into $GEMMA_LAST_FILE
  ...
}
trap '_gemma_record_last' DEBUG   # bash
# or precmd for zsh
```

`gemma why` reads `$GEMMA_LAST_FILE`, sends the captured command + exit code + stderr to Gemma with a focused prompt ("This command failed. Explain the likely cause and suggest a fix."), and prints the response. Stateless.

---

## What this pass does NOT include (explicitly deferred)

These are part of the broader roadmap but out of scope here:

- `gemma remember` / `gemma forget` / `gemma pin` / `gemma context`
- Profile switching (`--profile`, `~/.config/gemma/profiles/`)
- `--only` / `--json` / `--code` scripting flags
- SHA-keyed response cache
- Parallel embed+retrieve, lazy embedding load
- Sandboxed tool use, local-directory RAG, completion scripts, clipboard integration

---

## Implementation order

1. **`gemma/redaction.py`** + `tests/test_redaction.py` — land standalone, no wiring yet.
2. Wire `redact()` into `MemoryManager.record_turn()` + `_store.push_turn()` path. Add a test confirming stored turns never contain sample secrets.
3. **Keep-alive:** add `ollama_keep_alive` to `Config`; thread through `client.chat()` and `embeddings.Embedder.embed()`. Add a `--keep-alive` override on `ask` / `chat`.
4. **`gemma/commands/__init__.py`** (empty package init).
5. **`gemma/commands/shell.py`** — `sh`. Register in `main.py`. `--no-exec` path is implementable and tested without shell interaction; `--help` smoke test for the interactive path.
6. **`gemma/commands/explain.py`** — `explain`. Tests cover the three input modes with a mocked client.
7. **`gemma/commands/git.py`** — `commit` and `diff`. Tests use `tmp_path` + `subprocess` to drive a throwaway repo.
8. **`gemma/commands/shell.py` cont.** — `why` and `install-shell`. Unit-test the shell-snippet output strings; manual smoke-test the installed hook.

Each step lands as its own chunk so partial progress is useful even if the full pass pauses.

---

## Files to modify / create

**Modify:**
- `gemma/main.py` — register `sh`, `explain`, `commit`, `diff`, `why`, `install-shell`; thread `--keep-alive`
- `gemma/config.py` — `ollama_keep_alive`
- `gemma/client.py` — pass `keep_alive=` to ollama
- `gemma/embeddings.py` — pass `keep_alive=` to ollama
- `gemma/memory/manager.py` — call `redact()` inside `record_turn()` before persisting
- `gemma/memory/store.py` — no change (redaction happens in the manager)

**New:**
- `gemma/redaction.py`
- `gemma/commands/__init__.py`
- `gemma/commands/shell.py` (`sh`, `why`, `install-shell`)
- `gemma/commands/git.py` (`commit`, `diff`)
- `gemma/commands/explain.py` (`explain`)
- `tests/test_redaction.py`
- `tests/test_cmd_shell.py`
- `tests/test_cmd_git.py`
- `tests/test_cmd_explain.py`

**Probably no new dependencies.** Git and `pbcopy` via `subprocess`; regex via stdlib `re`. A single small dep may be added later if it clearly simplifies a specific command.

---

## Verification

- **Unit tests:** `pytest tests/ -v` — all existing 93 tests continue to pass; new tests cover redaction patterns, each new command's happy path, and at least one error path per command.
- **Redaction integration test:** record a turn containing a sample AWS key / GH token / JWT / private-key header; assert the stored Redis hash never contains the raw secret.
- **Keep-alive benchmark:** a small script (`scripts/bench_ttft.py`, not committed) times 5 consecutive `gemma ask` invocations. Second+ should show TTFT <300 ms.
- **`gemma sh --no-exec "find python files modified today"`** — produces a plausible `find` command.
- **`gemma commit --apply`** on a staged diff in a throwaway repo — creates a commit with a reasonable subject line.
- **Manual `gemma why`** flow: install the hook, run a command that fails, call `gemma why`, confirm a relevant explanation.