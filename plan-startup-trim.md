# Plan — Further startup trimming for `gemma-cli`

Status: design / not yet implemented. Lands on top of PRs #5 and #6
(landed in `main`).

---

## Current measurements (after PR #6)

```
import gemma.main
  → 189 ms
  → 694 modules
  → 59.6 MB RSS
  → numpy NOT loaded
```

Heaviest residual modules at startup, from `python -X importtime`:

| Module | Cumulative | Why heavy |
|---|---|---|
| `gemma.tools.builtins.net_fetch` | **~124 ms** | `import trafilatura` at module top |
| `gemma.commands.tools` | ~130 ms | drags `gemma.tools.builtins` (all 9 builtin tools) |
| `gemma.commands.explain` | ~17 ms | `gemma.cache.build_cache` + `gemma.client` chain |

Most short-lived commands (`gemma --help`, `gemma history show`,
`gemma model list`) need none of this. The plan is staged from
cheapest/safest to architecturally invasive.

---

## Stage 1 — Lazy `trafilatura` import (trivial)

**File:** `gemma/tools/builtins/net_fetch.py:31-36`

**Change:** push `import trafilatura` from module top into the one
function that uses it. Replace the module-level `_HAS_TRAFILATURA`
flag with a function-scoped `try: import; except ImportError`.

**Risk:** very low — existing `try`/`except ImportError` semantics are
preserved; the only difference is when the cost is paid.

**Expected win:** **~100 ms off cold-start** for any command that
touches `gemma.commands.tools`. Even bigger when the env doesn't
actually have `trafilatura` (the import is skipped entirely).

**Effort:** ~10 minutes, one file.

**Tests to verify:**
- existing `test_net_fetch_*` tests cover the extraction path
- one targeted assertion: `import gemma.tools.builtins.net_fetch` does
  not put `trafilatura` in `sys.modules`

---

## Stage 2 — Lazy `gemma.tools` package

**Files:**
- `gemma/tools/__init__.py:18-39` — does `from gemma.tools import builtins`
  at the bottom, which eagerly loads all 9 tool modules
- `gemma/tools/builtins/__init__.py:19-29` — explicit imports of every
  tool

**Constraint:** the tool registry is populated by import-time
`@tool(...)` decorators. So unlike memory/rag, we cannot fully defer —
*something* has to trigger registration before a tool is dispatched. But
registration only matters when an agent loop actually runs a tool; for
`gemma --help` / `gemma history show` it's pure waste.

### 2a — Move builtin auto-registration into `gemma.tools.bootstrap()`

Drop `from gemma.tools import builtins` from `gemma/tools/__init__.py`.
Add an explicit `bootstrap()` function that performs the import. Call
it from:
- the agent loop in `gemma/main.py:529` (`_agent_loop`)
- `gemma/commands/tools.py` command handlers (`tools_list_command` etc.)
- any test fixture that needs the registry populated

**Risk:** medium — existing call sites that import `gemma.tools` and
expect builtins to already be registered will break. Audit needed.

**Effort:** half a day. Most of it is auditing.

**Win:** ~130 ms + however many MB the builtin-tool deps take when
they all load together.

### 2b — PEP 562 lazy attribute on `gemma.tools` (no auto-registration change)

Keep `from gemma.tools import builtins` working, but make it a no-op
until something *accesses* a registered name. Replace the eager builtin
import with `__getattr__` that pulls each tool on demand (when looked
up via `gemma.tools.registry.get(name)`).

**Risk:** higher — registry lookup currently assumes the tool was
registered at import. We'd have to plumb name-aware lazy resolution
through `registry.get()`.

**Effort:** 1–2 days.

**Recommendation: do 2a.** Mechanically simple, registration boundary
is explicit, easy to audit.

---

## Stage 3 — Argv-aware lazy command registration in `main.py`

The architectural change. Even after stages 1 & 2, `gemma/main.py`
still does ~10 `from gemma.commands.X import ...` at module top because
Typer's `app.command(name)(fn)` requires the function object at
registration time.

### Why Typer makes this hard

Typer (and Click underneath) builds the command tree at app-instantiation
time. `app.command("ask")(fn)` wires `fn`'s signature into a Click
parser. Help, completion, and arg validation all rely on having every
function resolved up-front. You cannot register a "lazy callable" without
losing help text and completion for that command.

### Proposed approach: argv pre-parser + selective registration

Before constructing the Typer app, peek at `sys.argv` to determine which
subcommand was invoked. Register *only* that subcommand's module. Fall
back to registering everything when help / completion / unknown.

**Architecture sketch:**

```python
# gemma/main.py

# Manifest: name → (module path, attribute, parent app or None)
COMMAND_REGISTRY = {
    "ask":     ("gemma.main", "ask", None),                 # defined inline
    "chat":    ("gemma.main", "chat", None),
    "sh":      ("gemma.commands.shell", "sh_command", None),
    "explain": ("gemma.commands.explain", "explain_command", None),
    "history": _HISTORY_GROUP,                              # subgroup → child manifest
    "rag":     _RAG_GROUP,
    "tools":   _TOOLS_GROUP,
    # ...
}

def _peek_subcommand(argv):
    """Return the first non-flag token after global options.

    Skips '--profile X', '--backend Y', etc. Returns None for --help,
    --version, no-args, or completion contexts where we must register
    everything.
    """
    ...

def _register_subset(app, names):
    """Import & register only the named subcommands."""
    for name in names:
        target = COMMAND_REGISTRY[name]
        mod = importlib.import_module(target[0])
        fn = getattr(mod, target[1])
        app.command(name)(fn)

def main():
    app = typer.Typer(...)
    sub = _peek_subcommand(sys.argv)
    if sub and sub in COMMAND_REGISTRY:
        _register_subset(app, [sub])
    else:
        _register_subset(app, list(COMMAND_REGISTRY))  # full tree
    app()
```

### Edge cases the pre-parser must handle

1. **Global options before the subcommand:**
   `gemma --profile work ask "..."` — skip `--profile work` to find
   `ask`.
2. **Subgroup commands:** `gemma history show` — peek the first two
   tokens; resolve the subgroup then register only its children, or
   simplify by registering the whole subgroup.
3. **`--help` / `-h`:** must register everything for complete help.
4. **No args** (`gemma` alone): same — show full help.
5. **Shell completion:** Typer/Click invoke completion via the
   `_TYPER_COMPLETE_ARGS` env var. Detect and register everything.
6. **Unknown subcommand:** register everything so Typer's "did-you-mean"
   is accurate.
7. **`gemma model use --help`:** for subgroup help; register the
   subgroup.

### Risks

- **Completion fidelity:** must reliably detect the completion path.
  If we miss a completion invocation and register only one command,
  completion silently breaks.
- **Test plumbing:** many tests use `CliRunner` and invoke the Typer
  app directly. They may rely on the full command tree being registered.
  Verify all tests still pass.
- **Decorator vs. function form:** `ask` and `chat` are defined inline
  in `main.py` with `@app.command()`. They'd need to move into the
  registry pattern, or stay inline and let `main.py` keep registering
  them.

### Effort

2–3 days. Most of the time is in:

1. Extracting subgroup `add_typer` calls into the manifest.
2. Writing the argv peeker with full coverage of global flags.
3. Building a test suite covering completion / help / unknown /
   subgroup paths.

### Expected win

Best case: a `gemma history show` invocation only imports
`gemma.commands.history` (and its deps). ~5 modules vs. the current
~50. Probably **another 15–20 MB** off RSS.

Worst case (`--help` / completion): unchanged from today.

---

## Stage 4 — Tidy `gemma.cache` and `gemma.client` from command modules

After stage 3 this matters less (only loaded for commands that need
them), but for completeness:

**Files:**
- `gemma/commands/explain.py:30-31`
- `gemma/commands/git.py`
- `gemma/commands/shell.py`

Each has `from gemma.cache import build_cache` and
`from gemma.client import chat as client_chat` at the top. These can
move inside the function body.

**Effort:** 30 minutes.
**Win:** small (~1–2 ms each); only meaningful before stage 3 lands.

---

## Recommended rollout

| | Stage | Effort | Win | Risk |
|---|---|---|---|---|
| **1.** | Defer `trafilatura` in `net_fetch.py` | 10 min | ~100 ms | very low |
| **2a.** | Bootstrap-style tool registration | half day | ~130 ms + MB | medium |
| **3.** | Argv-aware command registration | 2–3 days | ~15–20 MB | high |
| **4.** | Tidy `cache`/`client` from command modules | 30 min | ~5 ms | very low |

**Suggested PR sequence:**
- **PR #1**: Stages 1 + 4 — both trivial, ship together.
- **PR #2**: Stage 2a — independent, well-contained.
- **PR #3**: Stage 3 — only after #1 and #2 land, biggest blast radius.

If only one stage is feasible: **Stage 1 alone is the best
value/risk ratio** (10 min for ~100 ms cold-start saving).

---

## Verification harness for any stage

After each PR, re-run the same baseline used to verify PR #6:

```python
import time, sys, resource
t0 = time.perf_counter()
import gemma.main
t1 = time.perf_counter()
print(f"main import: {(t1-t0)*1000:.0f} ms")
print(f"numpy loaded: {'numpy' in sys.modules}")
print(f"trafilatura loaded: {'trafilatura' in sys.modules}")
print(f"modules total: {len(sys.modules)}")
print(f"rss: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024:.1f} MB")
```

Targets after each stage land:

| After | RSS | Modules | Notes |
|---|---|---|---|
| PR #6 (today) | 59.6 MB | 694 | numpy out |
| Stage 1 | ~58 MB | ~660 | trafilatura out |
| Stage 2a | ~50 MB | ~550 | tool builtins out |
| Stage 3 | ~30 MB | ~300 | only invoked subcommand loaded |
