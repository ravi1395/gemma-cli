"""Typer CLI entry point for gemma-cli.

Four subcommands:
  - ask       Single-shot query (with optional memory)
  - chat      Interactive REPL with persistent memory + sliding window
  - pipe      Stateless: read input from stdin (no memory)
  - history   Inspect session, list condensed memories, show stats
"""

from __future__ import annotations

import sys
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from gemma.client import chat as client_chat
from gemma.config import Config
from gemma.history import SessionHistory
from gemma.memory import MemoryManager


app = typer.Typer(help="Local CLI for Google's Gemma 4 model via Ollama.")
history_app = typer.Typer(help="Manage history, memories, and stats.")
app.add_typer(history_app, name="history")

console = Console()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _make_config(
    model: Optional[str] = None,
    system: Optional[str] = None,
    memory_enabled: bool = True,
    thinking_mode: bool = False,
) -> Config:
    """Build a Config, applying optional CLI overrides."""
    cfg = Config()
    if model:
        cfg.model = model
    if system:
        cfg.system_prompt = system
    cfg.memory_enabled = memory_enabled
    cfg.thinking_mode = thinking_mode
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


def _stream_response(generator, stream: bool) -> str:
    """Consume a response generator, printing as it arrives. Return the full text.

    The generator yields (chunk_type, text) tuples where chunk_type is either
    "think" (extended reasoning, shown dimmed) or "content" (the response).
    """
    chunks: list[str] = []
    if stream:
        had_thinking = False
        first_content = True
        for chunk_type, text in generator:
            if chunk_type == "think":
                if not had_thinking:
                    console.print("[dim italic]thinking…[/dim italic]")
                    had_thinking = True
                console.print(text, end="", soft_wrap=True, highlight=False, style="dim italic")
            else:
                if had_thinking and first_content:
                    console.print()  # newline after the last thinking chunk
                    first_content = False
                chunks.append(text)
                console.print(text, end="", soft_wrap=True, highlight=False)
        console.print()
    else:
        thinking_parts: list[str] = []
        for chunk_type, text in generator:
            if chunk_type == "think":
                thinking_parts.append(text)
            else:
                chunks.append(text)
        if thinking_parts:
            console.rule("[dim]thinking[/dim]", style="dim")
            console.print("".join(thinking_parts), style="dim italic")
            console.rule(style="dim")
        console.print(Markdown("".join(chunks)))
    return "".join(chunks)


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
) -> None:
    """Send a one-off prompt. Retrieves relevant memories unless --no-memory."""
    cfg = _make_config(model=model, system=system, memory_enabled=not no_memory, thinking_mode=think)
    memory = _init_memory(cfg) if cfg.memory_enabled else None

    if memory is not None and memory.available:
        memory.record_turn("user", prompt)
        messages = memory.get_context_messages(prompt, system_prompt=cfg.system_prompt)
    else:
        messages = [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": prompt},
        ]

    gen = client_chat(messages, cfg, stream=not no_stream)
    reply = _stream_response(gen, stream=not no_stream)

    if memory is not None and memory.available:
        memory.record_turn("assistant", reply)


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
) -> None:
    """Interactive chat REPL with memory-augmented context."""
    cfg = _make_config(model=model, system=system, memory_enabled=not no_memory, thinking_mode=think)
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
        reply = _stream_response(gen, stream=True)
        memory.record_turn("assistant", reply)


# -----------------------------------------------------------------------------
# pipe
# -----------------------------------------------------------------------------

@app.command()
def pipe(
    instruction: str = typer.Argument("Analyze this input.", help="What to do with the piped content."),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    no_stream: bool = typer.Option(False, "--no-stream"),
) -> None:
    """Read stdin and send it together with an instruction. Stateless (no memory)."""
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
    gen = client_chat(messages, cfg, stream=not no_stream)
    _stream_response(gen, stream=not no_stream)


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
