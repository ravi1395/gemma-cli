"""``gemma rag`` subcommands — index, query, status, reset.

This module is the thin CLI veneer over :mod:`gemma.rag`. Heavy lifting
lives in the package; this file handles:

* wiring up the Redis client, embedder, and store from the active Config
* rendering results via Rich
* mapping exceptions to helpful user-facing messages

Dependency injection points (``_make_store_factory`` etc.) are exposed
as module-level variables so tests can swap in fakes without having to
patch the Ollama or Redis clients.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import typer
from rich.console import Console
from rich.table import Table

from gemma.config import Config
from gemma.embeddings import Embedder
from gemma.rag import (
    RAGIndexer,
    RAGRetriever,
    RedisVectorStore,
    resolve_namespace,
)


console = Console()


# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------
#
# The CLI doesn't own its dependencies directly — it goes through these
# factory hooks so tests can inject stub stores/embedders without
# touching Redis or Ollama. Production code leaves the defaults alone.

def _default_store_factory(namespace: str, redis_url: str) -> RedisVectorStore:
    return RedisVectorStore(namespace=namespace, redis_url=redis_url)


def _default_embedder_factory(cfg: Config) -> Any:
    return Embedder(model=cfg.embedding_model, host=cfg.ollama_host)


#: Hookable factories. Tests override these module-level attributes.
_store_factory: Callable[[str, str], RedisVectorStore] = _default_store_factory
_embedder_factory: Callable[[Config], Any] = _default_embedder_factory


def set_store_factory(factory: Callable[[str, str], RedisVectorStore]) -> None:
    """Override the Redis store factory (for tests)."""
    global _store_factory
    _store_factory = factory


def set_embedder_factory(factory: Callable[[Config], Any]) -> None:
    """Override the embedder factory (for tests)."""
    global _embedder_factory
    _embedder_factory = factory


# ---------------------------------------------------------------------------
# Shared wiring
# ---------------------------------------------------------------------------

def _wire(root: Path, cfg: Optional[Config] = None):
    """Build (cfg, store, embedder) for a given workspace root.

    Used by every subcommand so the Redis namespace is derived in one
    place and only one Ollama client is created per CLI invocation.
    """
    cfg = cfg or Config()
    namespace = resolve_namespace(root)
    store = _store_factory(namespace, cfg.redis_url)
    embedder = _embedder_factory(cfg)
    return cfg, store, embedder


# ---------------------------------------------------------------------------
# `gemma rag index`
# ---------------------------------------------------------------------------

def index_command(
    path: Optional[Path] = typer.Argument(
        None,
        help="Workspace root to index. Defaults to the current working directory.",
    ),
) -> None:
    """Incrementally index the workspace into Redis.

    The first call embeds every matching file; subsequent calls re-embed
    only files whose content changed (by mtime+size+sha1).
    """
    root = (path or Path.cwd()).resolve()
    if not root.is_dir():
        console.print(f"[red]error: {root} is not a directory[/red]")
        raise typer.Exit(code=1)

    cfg, store, embedder = _wire(root)

    console.print(f"indexing [bold]{root}[/bold] → namespace [cyan]{store.namespace}[/cyan]")

    indexer = RAGIndexer(root=root, store=store, embedder=embedder)
    stats = indexer.index(progress=lambda msg: console.print(f"· {msg}", style="dim"))

    console.print()
    console.print(stats.summary())
    if stats.errors:
        console.print("[yellow]errors:[/yellow]")
        for err in stats.errors[:10]:
            console.print(f"  {err}", style="yellow")
        if len(stats.errors) > 10:
            console.print(f"  …and {len(stats.errors) - 10} more", style="yellow")


# ---------------------------------------------------------------------------
# `gemma rag query`
# ---------------------------------------------------------------------------

def query_command(
    question: str = typer.Argument(..., help="Natural-language query."),
    k: int = typer.Option(5, "--k", "-k", help="Number of hits to return."),
    mmr: float = typer.Option(
        0.5, "--mmr", help="MMR lambda ∈ [0, 1]. 1 = pure relevance, 0 = pure diversity.",
    ),
    path: Optional[Path] = typer.Option(
        None, "--root",
        help="Workspace root. Defaults to the current directory.",
    ),
) -> None:
    """Search the indexed workspace for chunks relevant to ``question``."""
    root = (path or Path.cwd()).resolve()
    _cfg, store, embedder = _wire(root)

    if store.chunk_count() == 0:
        console.print(
            "[yellow]no chunks indexed for this workspace. "
            "run `gemma rag index` first.[/yellow]"
        )
        raise typer.Exit(code=1)

    retriever = RAGRetriever(store, embedder)
    hits = retriever.query(question, k=k, mmr_lambda=mmr)

    if not hits:
        console.print("[yellow]no hits.[/yellow]")
        return

    for i, hit in enumerate(hits, start=1):
        header = hit.header or ""
        heading = f"[bold]#{i}[/bold] [cyan]{hit.citation}[/cyan]"
        if header:
            heading += f"  [dim]{header}[/dim]"
        heading += f"  score=[green]{hit.score:+.3f}[/green]"
        console.print(heading)
        # Truncate long chunks for display; the model gets the full text.
        text = hit.text.splitlines()
        preview = "\n".join(text[:15])
        if len(text) > 15:
            preview += f"\n…[{len(text) - 15} more lines]"
        console.print(preview, style="white")
        console.print()


# ---------------------------------------------------------------------------
# `gemma rag status`
# ---------------------------------------------------------------------------

def status_command(
    path: Optional[Path] = typer.Option(
        None, "--root",
        help="Workspace root. Defaults to the current directory.",
    ),
) -> None:
    """Report what's indexed for the current workspace + namespace."""
    root = (path or Path.cwd()).resolve()
    _cfg, store, _embedder = _wire(root)

    meta = store.get_meta()
    manifest_size = len(store.load_manifest_hash())
    chunk_count = store.chunk_count()

    table = Table(title=f"RAG status — {root}", show_header=False, pad_edge=False)
    table.add_column("field", style="cyan")
    table.add_column("value")
    table.add_row("namespace", store.namespace)
    table.add_row("files in manifest", str(manifest_size))
    table.add_row("chunks indexed", str(chunk_count))
    table.add_row("embedding model", meta.get("model", "—"))
    table.add_row("embedding dim", meta.get("dim", "—"))
    table.add_row("last indexed at (unix)", meta.get("last_indexed_at", "—"))
    console.print(table)


# ---------------------------------------------------------------------------
# `gemma rag reset`
# ---------------------------------------------------------------------------

def reset_command(
    path: Optional[Path] = typer.Option(
        None, "--root",
        help="Workspace root. Defaults to the current directory.",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y",
        help="Skip confirmation.",
    ),
) -> None:
    """Delete every RAG key for this workspace's namespace.

    Use when switching embedding models (the namespace's stored
    vectors are only valid for one ``(model, dim)`` pair) or when you
    want to force a full re-index.
    """
    root = (path or Path.cwd()).resolve()
    _cfg, store, _embedder = _wire(root)

    if not yes:
        confirm = typer.confirm(
            f"clear namespace {store.namespace} for {root}?",
            default=False,
        )
        if not confirm:
            console.print("[yellow]aborted.[/yellow]")
            raise typer.Exit(code=1)

    deleted = store.clear_namespace()
    console.print(f"cleared {deleted} keys for namespace [cyan]{store.namespace}[/cyan]")
