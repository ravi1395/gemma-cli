"""Typer CLI entry point for gemma-cli.

Subcommands
-----------
Core:
  ask             Single-shot query (with optional memory)
  chat            Interactive REPL with persistent memory + sliding window
  pipe            Stateless: read input from stdin (no memory)
  history         Inspect session, list condensed memories, show stats

Memory power-user:
  remember        Seed a fact directly into the memory store (bypasses condensation)
  forget          Remove a memory (soft delete, keeps audit trail in Redis)
  pin             Set importance=5 on a memory so it never expires
  context         Preview what memory context would be injected for a query

Terminal-assistant:
  sh              Translate natural language to a shell command
  explain         Explain text, a file, a command, or an error
  commit          Generate a conventional-commit message from staged changes
  diff            Plain-English summary of git diff output
  why             Explain why the last shell command failed
  install-shell   Print/append the shell hook snippet for ``why``

All terminal-assistant commands are stateless by default (no memory read/write)
and behave correctly when stdout is not a TTY (pipe-safe).
"""

from __future__ import annotations

import dataclasses
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from gemma.cache import build_cache
from gemma.client import chat as client_chat
from gemma.commands.clipboard import status_command as clipboard_status_command
from gemma.commands.completion import (
    install_command as completion_install_command,
    print_command as completion_print_command,
    status_command as completion_status_command,
    uninstall_command as completion_uninstall_command,
)
from gemma.commands.explain import explain_command
from gemma.commands.git import commit_command, diff_command
from gemma.commands.memory import (
    context_command,
    forget_command,
    pin_command,
    remember_command,
)
from gemma.commands.shell import (
    install_shell_command,
    sh_command,
    why_command,
)
from gemma.commands.rag import (
    index_command as rag_index_command,
    query_command as rag_query_command,
    reset_command as rag_reset_command,
    status_command as rag_status_command,
)
from gemma.commands.tools import (
    audit_command as tools_audit_command,
    list_command as tools_list_command,
    run_command as tools_run_command,
)
from gemma.completion import profile_completer
from gemma.config import Config
from gemma.history import SessionHistory
from gemma.memory import MemoryManager
from gemma.output import OutputMode, render_response


app = typer.Typer(help="Local CLI for Google's Gemma 4 model via Ollama.")
history_app = typer.Typer(help="Manage history, memories, and stats.")
app.add_typer(history_app, name="history")

# Clipboard subcommands (``gemma clipboard status`` today; room to grow).
clipboard_app = typer.Typer(help="Clipboard diagnostics and helpers.")
clipboard_app.command("status")(clipboard_status_command)
app.add_typer(clipboard_app, name="clipboard")

# Shell completion subcommands — install/print/status/uninstall.
# Kept in their own Typer group so the top-level help stays uncluttered.
completion_app = typer.Typer(help="Shell tab-completion installer and helpers.")
completion_app.command("install")(completion_install_command)
completion_app.command("print")(completion_print_command)
completion_app.command("status")(completion_status_command)
completion_app.command("uninstall")(completion_uninstall_command)
app.add_typer(completion_app, name="completion")

# Tool-use subsystem. ``gemma tools list`` shows every registered tool;
# ``gemma tools run <name>`` invokes one directly (same dispatcher the
# model uses). ``gemma tools audit`` tails the append-only JSONL log.
tools_app = typer.Typer(help="Sandboxed tool registry (experimental).")
tools_app.command("list")(tools_list_command)
tools_app.command("run")(tools_run_command)
tools_app.command("audit")(tools_audit_command)
app.add_typer(tools_app, name="tools")

# Local RAG over the workspace. ``gemma rag index`` builds or refreshes
# the namespace-scoped vector store in Redis; ``query`` asks a question;
# ``status`` prints what is currently indexed; ``reset`` wipes the
# namespace (useful when switching embedding models).
rag_app = typer.Typer(help="Local RAG over the workspace (experimental).")
rag_app.command("index")(rag_index_command)
rag_app.command("query")(rag_query_command)
rag_app.command("status")(rag_status_command)
rag_app.command("reset")(rag_reset_command)
app.add_typer(rag_app, name="rag")

# Active profile set by --profile flag in the top-level callback.
# None means "use Config dataclass defaults".
_active_profile: Optional[Config] = None

# Terminal-assistant commands (stateless, developer-workflow focused)
app.command("sh")(sh_command)
app.command("explain")(explain_command)
app.command("commit")(commit_command)
app.command("diff")(diff_command)
app.command("why")(why_command)
app.command("install-shell")(install_shell_command)

# Memory power-user commands
app.command("remember")(remember_command)
app.command("forget")(forget_command)
app.command("pin")(pin_command)
app.command("context")(context_command)

console = Console()


# -----------------------------------------------------------------------------
# Top-level callback — global flags (e.g. --profile) applied before subcommands
# -----------------------------------------------------------------------------

@app.callback()
def main_callback(
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        "-p",
        help=(
            "Load a named config profile from "
            "~/.config/gemma/profiles/<name>.toml before running a subcommand."
        ),
        # Dynamic completion: offers profile stems from
        # ``~/.config/gemma/profiles/*.toml``. Silent on errors so a
        # broken profiles dir never degrades the shell experience.
        autocompletion=profile_completer,
    ),
) -> None:
    """gemma – local Gemma 4 CLI via Ollama."""
    global _active_profile
    if profile:
        try:
            _active_profile = Config.load_profile(profile)
        except FileNotFoundError as exc:
            console.print(f"[red]error: {exc}[/red]")
            raise typer.Exit(code=1)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _make_config(
    model: Optional[str] = None,
    system: Optional[str] = None,
    memory_enabled: bool = True,
    thinking_mode: bool = False,
    keep_alive: Optional[str] = None,
) -> Config:
    """Build a Config, applying optional profile then explicit CLI overrides.

    Resolution order (highest wins):
      1. Explicit CLI flag (e.g. --model)
      2. Profile TOML field (set via --profile flag)
      3. Config dataclass default
    """
    # Start from a shallow copy of the active profile or bare defaults.
    cfg = dataclasses.replace(_active_profile) if _active_profile is not None else Config()
    if model:
        cfg.model = model
    if system:
        cfg.system_prompt = system
    cfg.memory_enabled = memory_enabled
    cfg.thinking_mode = thinking_mode
    if keep_alive:
        cfg.ollama_keep_alive = keep_alive
    return cfg


def _init_memory(cfg: Config) -> MemoryManager:
    """Initialize a MemoryManager, warning the user if it falls into degraded mode."""
    mgr = MemoryManager(cfg)
    mgr.initialize()
    if mgr.degraded and cfg.memory_enabled:
        console.print(
            "[yellow]memory: Redis unreachable, running without long-term memory[/yellow]"
        )
    return mgr


def _resolve_output_mode(
    json_flag: bool,
    only: Optional[str],
    code: bool,
) -> tuple[OutputMode, Optional[str]]:
    """Validate that at most one scripting output flag is set and return the mode.

    Args:
        json_flag: True if --json was passed.
        only:      Field name if --only was passed, else None.
        code:      True if --code was passed.

    Returns:
        A (OutputMode, field) pair where field is only set for ONLY mode.
    """
    set_count = sum([json_flag, only is not None, code])
    if set_count > 1:
        console.print("[red]error: --json, --only, and --code are mutually exclusive[/red]")
        raise typer.Exit(code=1)
    if json_flag:
        return OutputMode.JSON, None
    if only is not None:
        return OutputMode.ONLY, only
    if code:
        return OutputMode.CODE, None
    return OutputMode.RICH, None


# -----------------------------------------------------------------------------
# ask
# -----------------------------------------------------------------------------

@app.command()
def ask(
    prompt: str = typer.Argument(..., help="The prompt to send to Gemma."),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    system: Optional[str] = typer.Option(None, "--system", "-s"),
    no_stream: bool = typer.Option(False, "--no-stream"),
    no_memory: bool = typer.Option(False, "--no-memory", help="Skip memory retrieval."),
    think: bool = typer.Option(False, "--think", help="Enable Gemma 4 extended thinking mode."),
    keep_alive: Optional[str] = typer.Option(
        None, "--keep-alive",
        help="Ollama model-residency duration (e.g. '30m', '2h', '-1' to pin, '0' to evict).",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit a JSON object (for piping to jq)."),
    only: Optional[str] = typer.Option(None, "--only", help="Print a single JSON field (content, model, elapsed_ms, cache_hit)."),
    code: bool = typer.Option(False, "--code", help="Extract and emit only fenced code blocks."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the response cache."),
    cache_only: bool = typer.Option(
        False, "--cache-only",
        help="Error if no cache hit (requires --no-stream).",
    ),
) -> None:
    """Send a one-off prompt. Retrieves relevant memories unless --no-memory."""
    mode, field = _resolve_output_mode(json_output, only, code)
    cfg = _make_config(
        model=model, system=system, memory_enabled=not no_memory,
        thinking_mode=think, keep_alive=keep_alive,
    )
    memory = _init_memory(cfg) if cfg.memory_enabled else None

    if memory is not None and memory.available:
        memory.record_turn("user", prompt)
        messages = memory.get_context_messages(prompt, system_prompt=cfg.system_prompt)
    else:
        messages = [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": prompt},
        ]

    # Cache path: only for non-streaming, low-temperature calls.
    cache = (
        build_cache(cfg)
        if (no_stream and not no_cache and cfg.cache_enabled
                and cfg.temperature <= cfg.cache_temperature_max)
        else None
    )
    cached: Optional[str] = cache.get(messages, cfg) if cache else None

    if cached is not None:
        reply = render_response(
            iter([("content", cached)]), mode=mode, stream=False, field=field, model=cfg.model
        )
        if memory is not None and memory.available:
            memory.record_turn("assistant", reply)
        return

    if cache_only and no_stream:
        console.print("[red]gemma ask: no cache hit and --cache-only was set.[/red]")
        raise typer.Exit(code=1)

    gen = client_chat(messages, cfg, stream=not no_stream)
    reply = render_response(gen, mode=mode, stream=not no_stream, field=field, model=cfg.model)

    if memory is not None and memory.available:
        memory.record_turn("assistant", reply)

    if cache and reply:
        cache.put(messages, cfg, reply)


# -----------------------------------------------------------------------------
# chat
# -----------------------------------------------------------------------------

@app.command()
def chat(
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    system: Optional[str] = typer.Option(None, "--system", "-s"),
    fresh: bool = typer.Option(False, "--fresh", help="Start with an empty sliding window."),
    no_memory: bool = typer.Option(False, "--no-memory", help="Disable all memory features."),
    think: bool = typer.Option(False, "--think", help="Enable Gemma 4 extended thinking mode."),
    keep_alive: Optional[str] = typer.Option(
        None, "--keep-alive",
        help="Ollama model-residency duration (e.g. '30m', '2h', '-1' to pin, '0' to evict).",
    ),
) -> None:
    """Interactive chat REPL with memory-augmented context."""
    cfg = _make_config(
        model=model, system=system, memory_enabled=not no_memory,
        thinking_mode=think, keep_alive=keep_alive,
    )
    memory = _init_memory(cfg)

    if fresh:
        memory.clear_session()

    mode = "memory" if memory.available else "degraded"
    console.print(
        f"[dim]gemma chat ({cfg.model}, {mode} mode) "
        f"-- type 'exit' or Ctrl-D to quit[/dim]"
    )

    while True:
        try:
            user_input = console.input("[bold cyan]you> [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", ":q"}:
            break

        memory.record_turn("user", user_input)
        messages = memory.get_context_messages(user_input, system_prompt=cfg.system_prompt)

        console.print("[bold green]gemma>[/bold green] ", end="")
        gen = client_chat(messages, cfg, stream=True)
        reply = render_response(gen, mode=OutputMode.RICH, stream=True, model=cfg.model)
        memory.record_turn("assistant", reply)


# -----------------------------------------------------------------------------
# pipe
# -----------------------------------------------------------------------------

@app.command()
def pipe(
    instruction: str = typer.Argument("Analyze this input.", help="What to do with the piped content."),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    no_stream: bool = typer.Option(False, "--no-stream"),
    json_output: bool = typer.Option(False, "--json", help="Emit a JSON object (for piping to jq)."),
    only: Optional[str] = typer.Option(None, "--only", help="Print a single JSON field (content, model, elapsed_ms, cache_hit)."),
    code: bool = typer.Option(False, "--code", help="Extract and emit only fenced code blocks."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the response cache."),
    cache_only: bool = typer.Option(
        False, "--cache-only",
        help="Error if no cache hit (requires --no-stream).",
    ),
) -> None:
    """Read stdin and send it together with an instruction. Stateless (no memory)."""
    mode, field = _resolve_output_mode(json_output, only, code)

    if sys.stdin.isatty():
        console.print("[red]gemma pipe: expected input on stdin[/red]")
        raise typer.Exit(code=2)

    piped = sys.stdin.read()
    cfg = _make_config(model=model, memory_enabled=False)
    combined = f"{instruction}\n\n---\n\n{piped}"
    messages = [
        {"role": "system", "content": cfg.system_prompt},
        {"role": "user", "content": combined},
    ]

    # Cache path: only for non-streaming, low-temperature calls.
    cache = (
        build_cache(cfg)
        if (no_stream and not no_cache and cfg.cache_enabled
                and cfg.temperature <= cfg.cache_temperature_max)
        else None
    )
    cached: Optional[str] = cache.get(messages, cfg) if cache else None

    if cached is not None:
        render_response(
            iter([("content", cached)]), mode=mode, stream=False, field=field, model=cfg.model
        )
        return

    if cache_only and no_stream:
        console.print("[red]gemma pipe: no cache hit and --cache-only was set.[/red]")
        raise typer.Exit(code=1)

    gen = client_chat(messages, cfg, stream=not no_stream)
    result = render_response(gen, mode=mode, stream=not no_stream, field=field, model=cfg.model)

    if cache and result:
        cache.put(messages, cfg, result)


# -----------------------------------------------------------------------------
# history subcommands
# -----------------------------------------------------------------------------

@history_app.command("show")
def history_show() -> None:
    """Print the current session (JSON fallback history, if any)."""
    cfg = Config()
    history = SessionHistory(cfg)
    turns = history.show()
    if not turns:
        console.print("[dim]no history[/dim]")
        return
    for turn in turns:
        role = turn.get("role", "?")
        content = turn.get("content", "")
        color = {"system": "magenta", "user": "cyan", "assistant": "green"}.get(role, "white")
        console.print(f"[bold {color}]{role}>[/bold {color}] {content}")


@history_app.command("clear")
def history_clear() -> None:
    """Wipe both the JSON fallback history and the current-session sliding window."""
    cfg = Config()
    SessionHistory(cfg).clear()
    mgr = MemoryManager(cfg)
    mgr.initialize()
    mgr.clear_session()
    console.print("[yellow]history cleared (condensed memories preserved)[/yellow]")


@history_app.command("memories")
def history_memories(
    limit: int = typer.Option(50, "--limit", "-n", help="Max number of memories to show."),
) -> None:
    """List condensed memories stored in Redis, sorted by importance."""
    cfg = Config()
    mgr = MemoryManager(cfg)
    mgr.initialize()

    if not mgr.available:
        console.print("[yellow]memory system unavailable[/yellow]")
        return

    records = mgr.list_memories(limit=limit)
    if not records:
        console.print("[dim]no condensed memories yet[/dim]")
        return

    table = Table(title=f"Condensed memories ({len(records)})")
    table.add_column("Importance", justify="right", style="bold")
    table.add_column("Category", style="cyan")
    table.add_column("Content")
    for r in records:
        table.add_row(str(r.importance), r.category.value, r.content)
    console.print(table)


@history_app.command("stats")
def history_stats() -> None:
    """Print memory-system statistics."""
    cfg = Config()
    mgr = MemoryManager(cfg)
    mgr.initialize()
    stats = mgr.get_stats()

    if not stats.get("available"):
        console.print("[yellow]memory system unavailable[/yellow]")
        console.print(f"fallback turns: {stats.get('fallback_turns', 0)}")
        return

    console.print(f"[bold]session:[/bold] {stats['session_id']}")
    console.print(f"[bold]active memories:[/bold] {stats['active_memories']}")
    console.print(f"[bold]window turns:[/bold] {stats['window_turns']}")

    if stats["by_category"]:
        console.print("\n[bold]by category:[/bold]")
        for cat, count in sorted(stats["by_category"].items()):
            console.print(f"  {cat}: {count}")

    if stats["by_importance"]:
        console.print("\n[bold]by importance:[/bold]")
        for imp in sorted(stats["by_importance"].keys(), reverse=True):
            console.print(f"  {imp}: {stats['by_importance'][imp]}")


if __name__ == "__main__":
    app()
