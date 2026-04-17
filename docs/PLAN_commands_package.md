# Plan: `gemma/commands/` Package — Terminal Assistant Commands

**Status:** Planned (follow-up to the completed redaction + keep-alive pass).
**Scope:** Add five Tier-1 developer-workflow commands to gemma-cli: `sh`, `explain`, `commit`, `diff`, `why`, plus the one-time `install-shell` helper.
**Out of scope:** memory power-user commands (`remember`/`forget`/`pin`), profiles, scripting output modes, response cache, RAG, tool use. All deferred.

---

## Context

The redaction + keep-alive chunk has landed (121/121 tests passing). The four original commands (`ask`, `chat`, `pipe`, `history`) are stable. What remains from the approved first-pass plan (`~/.claude/plans/lovely-juggling-brook.md`) is a set of developer-friendly subcommands that turn the CLI from a generic REPL into a terminal assistant.

Design principles carried forward from the original plan:

1. **Stdin/stdout first.** Every new command behaves correctly in a pipe.
2. **Fast path for quick queries.** `sh`, `explain`, `why` are stateless by default (no memory read/write) so they return quickly.
3. **No hidden side effects.** Any command that could mutate state (run a shell command, create a commit) confirms first and requires an explicit `--apply`/`y` to proceed.
4. **Reuse existing plumbing** — `Config`, `client.chat()`, the Typer app, `_stream_response()` helper.

---

## Deliverables

### 1. `gemma/commands/__init__.py`
Empty package init. Exists so `gemma.commands.shell`, `gemma.commands.git`, `gemma.commands.explain` are importable. Each submodule exports a small Typer app (or individual command functions) that `main.py` wires onto the top-level `app`.

### 2. `gemma sh "<natural language>"` — `commands/shell.py`
Translate a natural-language description into a single shell command.

**Flow:**
1. Build a tight system prompt: `"You output exactly one shell command. No prose, no markdown, no code fences. If unsafe, output nothing."`
2. Call `client.chat()` with `temperature=0.2`, `stream=False`, no memory.
3. Post-process: strip whitespace, strip accidental triple-backticks, collapse to the first line.
4. Print the command in a bordered panel.
5. If stdout is a TTY and `--no-exec` is NOT set: prompt `Run this? [y/N]` and on `y` `subprocess.run(cmd, shell=True, executable=<detected shell>)`.
6. If stdout is NOT a TTY: print the bare command to stdout, no prompt (pipe-safe).

**Flags:**
- `--no-exec` — print only, never prompt. Enforces pipe-friendly output.
- `--shell bash|zsh|sh` — target syntax; defaults to `$SHELL` basename.
- `--explain` — also print a one-line `#` comment above the command.

**Safety guards (v1):**
- Refuses to run any command containing `rm -rf /`, `mkfs`, `dd if=` (case-insensitive substring check). Prints a warning and falls through to `--no-exec` behavior.
- Never auto-executes without the explicit `y` keystroke.

### 3. `gemma explain` — `commands/explain.py`
Mode auto-detected from arguments and TTY state:

| Input | Mode |
|---|---|
| stdin is not a TTY | Read stdin, explain |
| a path argument passed | Read file, explain |
| `--cmd "..."` | Explain what the command does, without running |
| `--error "..."` | Explain an error string |

Stateless by default. `--with-memory` opts into memory retrieval.

**Prompt skeleton:** `"Explain the following {input_kind} in plain English. Be concise; start with a one-line summary, then bullet points for details."`

**Flags:**
- `--with-memory` — use the full memory-augmented context.
- `--lines N` — for file mode, only explain the first N lines (default: full file up to 20 KB).

### 4. `gemma commit` — `commands/git.py`
Generate a conventional-commits-style message from staged changes.

**Flow:**
1. `git rev-parse --git-dir` — bail if not in a repo.
2. `git diff --cached` — bail if empty (print "nothing staged").
3. If diff > 20 KB, truncate and prepend a warning comment to the prompt.
4. System prompt: `"You write conventional commit messages. Output only: <type>(<scope>): <subject>\n\n<body>. No prose, no fences. Subject ≤72 chars, imperative mood."`
5. `temperature=0.2`, `stream=False`.
6. Parse into `subject` + `body`.
7. Print for review.
8. If `--apply`: `git commit -m "<subject>" -m "<body>"`. Otherwise exit 0 and let the user copy.

**Flags:**
- `--apply` — actually create the commit.
- `--type feat|fix|chore|…` — force the conventional-commit type.

### 5. `gemma diff [refspec]` — `commands/git.py`
Plain-English summary of `git diff <refspec>`, grouped by file.

**Flow:**
1. Default refspec: empty string (unstaged working tree).
2. `git diff <refspec>` — bail on empty.
3. Group hunks by filename. For each file, feed a prompt: `"Summarize what changed in this file in one or two sentences."`
4. Render as a list: `filename — summary`.
5. If `--overall`, feed the full (possibly truncated) diff for a single-paragraph summary.

**Flags:**
- `--overall` — produce one global summary instead of per-file.
- `--staged` — shortcut for `--cached` (also as positional: `gemma diff --staged`).

### 6. `gemma why` — `commands/shell.py`
Explain why the last shell command failed. Requires `install-shell` to have been run.

**Flow:**
1. Read `$GEMMA_LAST_FILE` (default `~/.gemma_last_cmd`). Each record: TSV of `exit_code\tcommand\tstderr_snapshot`.
2. Bail clearly if the file is missing or the last exit code was 0.
3. Prompt: `"This shell command failed. Explain the likely cause and suggest a fix.\n\nCommand: <cmd>\nExit: <ec>\nStderr:\n<stderr>"`.
4. Stream the response.

### 7. `gemma install-shell` — `commands/shell.py`
Print (or optionally append) a shell snippet that captures the last command + exit code + stderr into `$GEMMA_LAST_FILE`.

**Bash snippet (rough):**
```bash
export GEMMA_LAST_FILE="${GEMMA_LAST_FILE:-$HOME/.gemma_last_cmd}"
_gemma_pre() { _gemma_cmd="$BASH_COMMAND"; }
_gemma_post() {
  local ec=$?
  printf '%s\t%s\t' "$ec" "$_gemma_cmd" > "$GEMMA_LAST_FILE"
  # stderr capture requires a wrapper; v1 stores exit code + cmd only.
}
trap '_gemma_pre' DEBUG
PROMPT_COMMAND="_gemma_post${PROMPT_COMMAND:+;$PROMPT_COMMAND}"
```

**Zsh snippet** uses `preexec` + `precmd` instead of `trap DEBUG` + `PROMPT_COMMAND`.

**Flags:**
- `--bash` / `--zsh` — pick the target shell (default: detect from `$SHELL`).
- `--append PATH` — append to the specified rc file (`~/.bashrc`, `~/.zshrc`) instead of printing. Back up first with `.gemma-backup` suffix.
- `--print` (default) — write to stdout only.

**v1 limitation:** stderr capture requires wrapping every command in `2>>somefile`, which is intrusive and fragile. First pass stores only `exit_code` + `command`. The `why` prompt handles the missing stderr gracefully.

---

## Implementation order

Each step is self-contained and testable in isolation.

1. **`gemma/commands/__init__.py`** (empty) + scaffold each new module with a placeholder Typer app.
2. **`commands/shell.py` — `sh`** (happy path + `--no-exec` + safety substring guard). Register in `main.py`.
3. **`commands/explain.py` — `explain`** (three input modes: stdin, file, `--cmd`). Register in `main.py`.
4. **`commands/git.py` — `commit`** (diff → message, `--apply` path). Register.
5. **`commands/git.py` — `diff`** (reuse the git-repo helper from step 4). Register.
6. **`commands/shell.py` — `why` + `install-shell`**. Register.
7. Update `README.md`: add subsections for each new command under a new "Terminal-assistant commands" section; update the command table.
8. Run the full suite. Must hold at ≥121 tests; add ≥2 tests per new command.

---

## Files to create / modify

**New:**
- `gemma/commands/__init__.py`
- `gemma/commands/shell.py` — `sh`, `why`, `install-shell`
- `gemma/commands/explain.py` — `explain`
- `gemma/commands/git.py` — `commit`, `diff`
- `tests/test_cmd_shell.py`
- `tests/test_cmd_explain.py`
- `tests/test_cmd_git.py`

**Modify:**
- `gemma/main.py` — import and register each new command on the top-level `app`.
- `README.md` — new "Terminal-assistant commands" section + command table update.

**No new dependencies.** Git via `subprocess`; regex via stdlib `re`; Typer + Rich already present.

---

## Testing strategy

Use `typer.testing.CliRunner` for command integration tests. Mock `client.chat()` to return canned generators so no Ollama is required.

**Per-command must-haves:**

| Command | Happy path | Error path |
|---|---|---|
| `sh` | `--no-exec` prints a plausible command | missing argument exits non-zero |
| `sh` | safety guard blocks `rm -rf /` | n/a |
| `explain` | stdin mode reads piped input | missing file → clear error |
| `explain` | `--cmd` mode does not execute | n/a |
| `commit` | generates a message from a throwaway repo's staged diff (driven via `tmp_path` + `subprocess`) | exits non-zero outside a git repo |
| `commit` | `--apply` creates a commit with the generated subject | empty staging area → clear error |
| `diff` | summarises a fake diff | empty diff → clear message |
| `why` | reads `$GEMMA_LAST_FILE` and streams a response | missing file → clear error |
| `install-shell` | `--print` emits a syntactically valid snippet | unsupported shell → clear error |

Snapshot the `install-shell` output strings so regressions are caught mechanically.

---

## Verification

- `pytest tests/ -v` — all 121 existing tests still pass; ≥10 new tests added.
- Manual: `gemma sh --no-exec "find python files modified today"` produces a plausible `find` command.
- Manual: in a throwaway git repo with staged changes, `gemma commit --apply` creates a commit with a reasonable subject line.
- Manual: install the shell hook, run `false`, run `gemma why`, confirm a coherent explanation.
- Manual: `gemma diff HEAD~1` on a small repo produces a per-file summary.

---

## Open questions (decide at implementation time)

1. **Safety guard list for `sh`:** should we use a small hand-rolled blocklist (current plan) or shell out to `shellcheck`? Blocklist is zero-dep; shellcheck is more thorough but optional.
2. **`gemma why` stderr capture:** v1 skips it for install simplicity. If the first pass lands well, a v2 hook can wrap commands with a `{ cmd; } 2> >(tee >(tail -c 4096 > $GEMMA_LAST_STDERR) >&2)` trick in zsh; bash is harder. Punt until asked.
3. **`commit --apply` signing:** if `commit.gpgsign=true`, git will open the editor. We do not pass `--no-gpg-sign`. Document this so users expect it.
4. **Memory interaction for `commit`/`diff`:** always stateless in v1. Revisit if users ask for repo-specific context retention.

---

## Definition of done

- [ ] All 6 new commands registered and invocable via `gemma <cmd> --help`.
- [ ] `README.md` updated with usage examples.
- [ ] Test suite ≥131 passing.
- [ ] `setup.sh` unchanged (no new deps).
- [ ] No existing tests modified except to add assertions; no behaviour regressions.
