"""Memory power-user commands.

Provides ``remember``, ``forget``, ``pin``, and ``context`` commands for
managing facts in the memory store directly, bypassing the conversation turn
→ condensation pipeline.  Useful for pre-loading preferences, auditing
stored context, and correcting bad memories without running a chat session.
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from gemma.config import Config
from gemma.memory import MemoryManager
from gemma.memory.models import MemoryCategory, MemoryRecord

_console = Console()


# Map CLI shortcut strings to MemoryCategory enum members.
# Full enum values (e.g. "factual_context") are also accepted via the lenient
# MemoryCategory.parse() fallback so both "--category feat" and
# "--category factual_context" work correctly.
_CATEGORY_SHORTCUTS: dict[str, MemoryCategory] = {
    "feat": MemoryCategory.FACTUAL_CONTEXT,
    "pref": MemoryCategory.USER_PREFERENCE,
    "task": MemoryCategory.TASK_STATE,
    "instruction": MemoryCategory.INSTRUCTION,
    "correction": MemoryCategory.CORRECTION,
    "relationship": MemoryCategory.RELATIONSHIP,
    "tool_usage": MemoryCategory.TOOL_USAGE,
}


def remember_command(
    fact: str = typer.Argument(..., help="The fact to store in memory."),
    category: str = typer.Option(
        "feat",
        "--category",
        "-c",
        help=(
            "Category shortcut: feat (factual_context), pref (user_preference), "
            "task (task_state), instruction, correction, relationship, tool_usage. "
            "Full enum values are also accepted."
        ),
    ),
    importance: int = typer.Option(
        4,
        "--importance",
        "-i",
        min=1,
        max=5,
        help="Importance tier: 1 (trivial) to 5 (critical). Default: 4.",
    ),
) -> None:
    """Seed a fact directly into the memory store.

    The fact is embedded with nomic-embed-text and written as a MemoryRecord
    immediately — no conversation history or condensation required.  The
    assigned memory_id is printed to stdout so you can reference it later.

    Examples::

        gemma remember "I prefer type annotations in all Python code."
        gemma remember "Project: auth rewrite — using JWT, deadline end of month." --category task
        gemma remember "Never use rm -rf without confirmation." --category instruction --importance 5
    """
    # Resolve category: shortcut first, then lenient full-value parse.
    mem_cat = _CATEGORY_SHORTCUTS.get(category) or MemoryCategory.parse(category)

    cfg = Config()
    mgr = MemoryManager(cfg)
    ok = mgr.initialize()

    if not ok or not mgr.available:
        typer.echo(
            "error: memory system unavailable (Redis not reachable)", err=True
        )
        raise typer.Exit(code=1)

    try:
        memory_id = mgr.add_memory(fact, category=mem_cat, importance=importance)
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1)

    # Print the memory_id so the user can pin/forget it later.
    typer.echo(memory_id)


def forget_command(
    memory_id: Optional[str] = typer.Argument(
        None, help="Memory ID to forget (as printed by ``gemma remember``)."
    ),
    last: bool = typer.Option(
        False, "--last", help="Forget the most recently created memory."
    ),
    match: Optional[str] = typer.Option(
        None,
        "--match",
        help="Forget the top-1 memory matching this query string.",
        metavar="QUERY",
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Skip the confirmation prompt."
    ),
) -> None:
    """Remove a memory from the active index (soft delete).

    The record is marked inactive in Redis via the ``superseded_by`` field so
    the audit trail is preserved, but it will no longer be retrieved or
    injected into future prompts.

    Exactly one of ``memory_id``, ``--last``, or ``--match`` must be given.

    Examples::

        gemma forget abc123
        gemma forget --last
        gemma forget --match "staging Redis URL" --force
    """
    cfg = Config()
    mgr = MemoryManager(cfg)
    ok = mgr.initialize()

    if not ok or not mgr.available:
        typer.echo(
            "error: memory system unavailable (Redis not reachable)", err=True
        )
        raise typer.Exit(code=1)

    # --- Resolve the target memory -------------------------------------------
    target: Optional[MemoryRecord] = None

    if memory_id is not None:
        target = mgr.get_memory(memory_id)
        if target is None:
            typer.echo(f"error: memory '{memory_id}' not found", err=True)
            raise typer.Exit(code=1)
    elif last:
        target = mgr.get_latest_memory()
        if target is None:
            typer.echo("error: no active memories found", err=True)
            raise typer.Exit(code=1)
    elif match is not None:
        # top_k=1 returns at most one (record, score) pair
        hits = mgr._retriever.find_relevant(match, top_k=1)
        if not hits:
            typer.echo(f"error: no memory found matching '{match}'", err=True)
            raise typer.Exit(code=1)
        target, _ = hits[0]
    else:
        typer.echo(
            "error: provide a memory_id argument, --last, or --match", err=True
        )
        raise typer.Exit(code=1)

    # --- Confirm unless --force ----------------------------------------------
    if not force:
        typer.echo(
            f"Memory  id={target.memory_id}\n"
            f"  category : {target.category.value}\n"
            f"  importance: {target.importance}\n"
            f"  content  : {target.content}"
        )
        if not typer.confirm("Forget this?", default=False):
            typer.echo("Aborted.")
            raise typer.Exit(code=0)

    # --- Perform soft delete -------------------------------------------------
    if mgr.forget_memory(target.memory_id):
        typer.echo(f"Forgotten: {target.memory_id}")
    else:
        typer.echo("error: failed to forget memory", err=True)
        raise typer.Exit(code=1)


def pin_command(
    memory_id: Optional[str] = typer.Argument(
        None, help="Memory ID to pin (as printed by ``gemma remember``)."
    ),
    match: Optional[str] = typer.Option(
        None,
        "--match",
        help="Pin the top-1 memory matching this query string.",
        metavar="QUERY",
    ),
) -> None:
    """Set importance=5 on a memory so it never expires.

    Pinning re-saves the record with ``importance=5``, which maps to ``None``
    (no expiry) in the TTL tier table.  The record is re-indexed so retrieval
    continues to find it.

    Exactly one of ``memory_id`` or ``--match`` must be given.

    Examples::

        gemma pin abc123
        gemma pin --match "JWT deadline"
    """
    cfg = Config()
    mgr = MemoryManager(cfg)
    ok = mgr.initialize()

    if not ok or not mgr.available:
        typer.echo(
            "error: memory system unavailable (Redis not reachable)", err=True
        )
        raise typer.Exit(code=1)

    # --- Resolve the target memory -------------------------------------------
    target: Optional[MemoryRecord] = None

    if memory_id is not None:
        target = mgr.get_memory(memory_id)
        if target is None:
            typer.echo(f"error: memory '{memory_id}' not found", err=True)
            raise typer.Exit(code=1)
    elif match is not None:
        hits = mgr._retriever.find_relevant(match, top_k=1)
        if not hits:
            typer.echo(f"error: no memory found matching '{match}'", err=True)
            raise typer.Exit(code=1)
        target, _ = hits[0]
    else:
        typer.echo(
            "error: provide a memory_id argument or --match", err=True
        )
        raise typer.Exit(code=1)

    # --- Pin -----------------------------------------------------------------
    if mgr.pin_memory(target.memory_id):
        typer.echo(f"Pinned (importance=5, no expiry): {target.memory_id}")
    else:
        typer.echo("error: failed to pin memory", err=True)
        raise typer.Exit(code=1)


def context_command(
    query: str = typer.Argument(..., help="Query to preview injected context for."),
) -> None:
    """Show what memory context would be injected for a given query.

    Uses ``MemoryRetriever.find_relevant`` to rank memories by cosine
    similarity to ``query`` and displays the top-5 results as a Rich table.
    Also shows the most recent turns in the current session's sliding window.

    This command is read-only and does not modify any state.

    Examples::

        gemma context "auth rewrite JWT"
        gemma context "what did we decide about Redis?"
    """
    cfg = Config()
    mgr = MemoryManager(cfg)
    ok = mgr.initialize()

    if not ok or not mgr.available:
        _console.print("[yellow]warning: memory system unavailable[/yellow]")
        return

    # --- Relevant memories ---------------------------------------------------
    hits = mgr._retriever.find_relevant(query, top_k=5)

    table = Table(title=f"Memory context for: {query!r}")
    table.add_column("Imp", justify="right", style="bold")
    table.add_column("Category", style="cyan")
    table.add_column("Similarity", justify="right")
    table.add_column("Content")

    for rec, score in hits:
        table.add_row(
            str(rec.importance),
            rec.category.value,
            f"{score:.3f}",
            rec.content,
        )

    _console.print(table)

    # --- Recent turns in the sliding window ----------------------------------
    recent = mgr._store.get_recent_turns(
        mgr._session_id, mgr._config.sliding_window_size
    )
    if recent:
        _console.print(
            f"\n[bold]Recent turns (last {len(recent)} of sliding window):[/bold]"
        )
        for turn in recent:
            color = {
                "user": "cyan",
                "assistant": "green",
                "system": "magenta",
            }.get(turn.role, "white")
            _console.print(
                f"[bold {color}]{turn.role}>[/bold {color}] {turn.content}"
            )
    else:
        _console.print("\n[dim]No recent turns in current session.[/dim]")
