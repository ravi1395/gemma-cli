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
from typing import TYPE_CHECKING, Any, Callable, Optional

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

if TYPE_CHECKING:
    from gemma.session import GemmaSession


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

def _wire(
    root: Path,
    cfg: Optional[Config] = None,
    session: Optional["GemmaSession"] = None,
):
    """Build (cfg, store, embedder) for a given workspace root.

    Used by every subcommand so the Redis namespace is derived in one
    place per CLI invocation.

    When a :class:`~gemma.session.GemmaSession` is supplied the store's
    text client is replaced with the session's shared Redis connection so
    all subsystems (memory, cache, RAG) share a single socket. The
    embedder factory is still honoured — tests override it to inject stub
    embedders without needing a real Ollama server.
    """
    cfg = cfg or (session._cfg if session is not None else Config())
    # Memoised branch detection avoids re-forking ``git rev-parse`` on
    # every rag subcommand within the same session (#8).
    branch = session.branch_for(root) if session is not None else None
    namespace = resolve_namespace(root, branch=branch)
    store = _store_factory(namespace, cfg.redis_url)
    # Inject the session's shared client and pool when available
    # (production path). Tests override _store_factory to return a
    # pre-wired fakeredis store, so store._client is already set and we
    # leave it alone.
    if session is not None and store._client is None:
        if session.redis_client is not None:
            store._client = session.redis_client
        if session.redis_pool is not None and store._pool is None:
            store._pool = session.redis_pool
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
    force_hash: bool = typer.Option(
        False,
        "--force-hash",
        help="Always recompute SHA-1 (bypass the mtime+size fast path).",
    ),
) -> None:
    """Incrementally index the workspace into Redis.

    The first call embeds every matching file; subsequent calls re-embed
    only files whose content changed (by mtime+size+sha1).
    """
    from gemma.session import GemmaSession

    root = (path or Path.cwd()).resolve()
    if not root.is_dir():
        console.print(f"[red]error: {root} is not a directory[/red]")
        raise typer.Exit(code=1)

    with GemmaSession(Config()) as session:
        cfg, store, embedder = _wire(root, session=session)

        console.print(f"indexing [bold]{root}[/bold] → namespace [cyan]{store.namespace}[/cyan]")

        # Item #9: each worker gets its own embedder so per-thread
        # HTTP sessions don't serialise on a single keep-alive pool.
        # Item #10: consult the content-hash embed cache before each
        # call; cache_ttl is promoted from days → seconds here.
        ttl_seconds = max(0, int(cfg.embed_cache_ttl_days)) * 24 * 3600
        indexer = RAGIndexer(
            root=root, store=store, embedder=embedder,
            concurrency=cfg.embed_concurrency,
            embedder_factory=lambda: _embedder_factory(cfg),
            cache_enabled=cfg.embed_cache_enabled,
            cache_ttl_seconds=ttl_seconds or None,
        )
        stats = indexer.index(
            progress=lambda msg: console.print(f"· {msg}", style="dim"),
            force_hash=force_hash,
        )

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
    from gemma.session import GemmaSession

    root = (path or Path.cwd()).resolve()
    with GemmaSession(Config()) as session:
        _cfg, store, embedder = _wire(root, session=session)

        retriever = RAGRetriever(store, embedder)
        hits = retriever.query(question, k=k, mmr_lambda=mmr)

    # Empty hit list with no namespace meta means the store has never been
    # indexed — give a more actionable message than "no hits".
    # Avoids the prior SCARD probe that ran before every query (#7).
    if not hits and not store.get_meta():
        console.print(
            "[yellow]no chunks indexed for this workspace. "
            "run `gemma rag index` first.[/yellow]"
        )
        raise typer.Exit(code=1)

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
    from gemma.session import GemmaSession

    root = (path or Path.cwd()).resolve()
    with GemmaSession(Config()) as session:
        _cfg, store, _embedder = _wire(root, session=session)
        # Single pipelined read replaces three sequential round-trips (#11).
        snap = store.snapshot()

    table = Table(title=f"RAG status — {root}", show_header=False, pad_edge=False)
    table.add_column("field", style="cyan")
    table.add_column("value")
    table.add_row("namespace", store.namespace)
    table.add_row("files in manifest", str(snap.manifest_size))
    table.add_row("chunks indexed", str(snap.chunk_count))
    table.add_row("embedding model", snap.meta.get("model", "—"))
    table.add_row("embedding dim", snap.meta.get("dim", "—"))
    table.add_row("last indexed at (unix)", snap.meta.get("last_indexed_at", "—"))
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
    from gemma.session import GemmaSession

    root = (path or Path.cwd()).resolve()
    with GemmaSession(Config()) as session:
        _cfg, store, _embedder = _wire(root, session=session)

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


# ---------------------------------------------------------------------------
# `gemma rag cache stats` / `gemma rag cache clear` (item #10)
# ---------------------------------------------------------------------------
#
# The embed cache is namespace-agnostic (shared across repos and
# branches because keys are content-addressable), so the admin surface
# doesn't need a ``--root``. It does need ``--model`` since the cache
# keys the model tag into every entry: switching encoders should not
# require nuking vectors the old encoder can still use.

def cache_stats_command(
    model: Optional[str] = typer.Option(
        None, "--model",
        help="Restrict stats to one embedding model (default: all).",
    ),
) -> None:
    """Report size and shape of the content-hash embed cache.

    The scan is O(keys); for the sizes we expect (tens of thousands)
    this is still well under 100 ms on local Redis.
    """
    from gemma.session import GemmaSession

    # We only need a Redis-connected store — namespace is irrelevant
    # because the cache lives outside any namespace. Reuse the default
    # factory so tests can still swap it out.
    with GemmaSession(Config()) as session:
        store = _store_factory(namespace="__cache__", redis_url=session._cfg.redis_url)
        if store._client is None and session.redis_client is not None:
            store._client = session.redis_client
        info = store.embed_cache_stats(model=model)

    table = Table(title="embed cache", show_header=False, pad_edge=False)
    table.add_column("field", style="cyan")
    table.add_column("value")
    table.add_row("total keys", str(info["total_keys"]))
    table.add_row("approx bytes", f"{info['approx_bytes']:,}")
    per_model = info.get("per_model", {})
    if per_model:
        table.add_row(
            "by model",
            ", ".join(f"{m}={n}" for m, n in sorted(per_model.items())),
        )
    else:
        table.add_row("by model", "—")
    console.print(table)


def cache_clear_command(
    model: Optional[str] = typer.Option(
        None, "--model",
        help="Clear only keys for this model. Default: clear all cache keys.",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y",
        help="Skip confirmation.",
    ),
) -> None:
    """Drop entries from the content-hash embed cache.

    Use when switching embedding models you no longer plan to revisit,
    or when freeing Redis memory. Does NOT touch indexed chunks —
    those live in the per-namespace store and are removed via
    ``gemma rag reset``.
    """
    from gemma.session import GemmaSession

    target = f"model={model}" if model else "all models"
    if not yes:
        confirm = typer.confirm(
            f"clear embed cache ({target})?",
            default=False,
        )
        if not confirm:
            console.print("[yellow]aborted.[/yellow]")
            raise typer.Exit(code=1)

    with GemmaSession(Config()) as session:
        store = _store_factory(namespace="__cache__", redis_url=session._cfg.redis_url)
        if store._client is None and session.redis_client is not None:
            store._client = session.redis_client
        deleted = store.clear_embed_cache(model=model)
    console.print(f"cleared {deleted} embed cache keys ({target})")
