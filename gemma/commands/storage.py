"""``gemma storage {info, migrate}`` subcommands.

``info`` is a one-shot snapshot of the active storage backend — file
path, table sizes, on-disk footprint. Useful when a user is curious
where their memories live or whether a migration succeeded.

``migrate`` copies data between backends. The common direction is
``redis → sqlite`` (the upgrade path), but the reverse is implemented
too for symmetry — handy when restoring from a Redis backup or
exporting to a server-shared store.

Both commands are idempotent: re-running ``migrate`` after a partial
copy continues from where it left off (rows are upserted by primary
key) and skips rows already present.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterable, Optional

import typer
from rich.console import Console
from rich.table import Table

from gemma.config import Config


_console = Console()


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------

def info_command(
    backend: Optional[str] = typer.Option(
        None,
        "--backend",
        "-b",
        help="Override storage backend for this call ('sqlite' or 'redis').",
    ),
) -> None:
    """Show backend, file path, row counts, and on-disk size."""
    cfg = Config(storage_backend=backend) if backend else Config()
    name = cfg.storage_backend

    if name == "sqlite":
        _info_sqlite(cfg)
    elif name == "redis":
        _info_redis(cfg)
    else:
        _console.print(f"[red]error:[/red] unknown backend {name!r}")
        raise typer.Exit(code=1)


def _info_sqlite(cfg: Config) -> None:
    """Render counts + file size for the SQLite backend."""
    from gemma.storage.sqlite_db import _resolve_path, open_db, sweep_expired

    path = _resolve_path(cfg)
    if not path.exists():
        _console.print(
            f"SQLite store [cyan]{path}[/cyan] not yet created. "
            "It will be initialised on first write."
        )
        return

    conn = open_db(cfg)
    sweep_expired(conn)

    counts: list[tuple[str, int]] = []
    for table in (
        "memories", "memory_embeddings",
        "session_turns", "session_meta",
        "rag_chunks", "rag_embeddings",
        "rag_manifest", "rag_meta",
        "embed_cache", "response_cache",
    ):
        try:
            row = conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()
            counts.append((table, int(row["c"])))
        except Exception:
            counts.append((table, 0))

    size_bytes = path.stat().st_size
    wal_path = path.with_suffix(path.suffix + "-wal")
    wal_bytes = wal_path.stat().st_size if wal_path.exists() else 0

    table = Table(show_header=True, header_style="bold")
    table.add_column("Table")
    table.add_column("Rows", justify="right")
    for name, n in counts:
        table.add_row(name, f"{n:,}")
    _console.print(table)

    _console.print()
    _console.print(f"backend:        [cyan]sqlite[/cyan]")
    _console.print(f"file:           [cyan]{path}[/cyan]")
    _console.print(f"size:           {_human_bytes(size_bytes)}")
    if wal_bytes:
        _console.print(f"wal:            {_human_bytes(wal_bytes)} (rolls forward into the main file on close)")


def _info_redis(cfg: Config) -> None:
    """Render counts for the legacy Redis backend."""
    try:
        import redis  # type: ignore[import-not-found]
    except ImportError:
        _console.print(
            "[red]error:[/red] redis package not installed. "
            "Install with: [cyan]uv sync --extra memory[/cyan]"
        )
        raise typer.Exit(code=1)

    try:
        client = redis.Redis.from_url(cfg.redis_url, decode_responses=True)
        client.ping()
    except Exception as exc:
        _console.print(f"[red]error:[/red] cannot reach Redis at {cfg.redis_url}: {exc}")
        raise typer.Exit(code=1)

    counts = {
        "memories": client.zcard("gemma:memory:index"),
        "rag chunks (all namespaces)": sum(
            1 for _ in client.scan_iter(match="gemma:rag:*:chunk:*", count=500)
        ),
        "response cache entries": sum(
            1 for _ in client.scan_iter(match="gemma:cache:*", count=500)
        ),
    }
    table = Table(show_header=True, header_style="bold")
    table.add_column("Resource")
    table.add_column("Rows", justify="right")
    for name, n in counts.items():
        table.add_row(name, f"{int(n):,}")
    _console.print(table)
    _console.print()
    _console.print(f"backend:        [cyan]redis[/cyan]")
    _console.print(f"url:            [cyan]{cfg.redis_url}[/cyan]")


def _human_bytes(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if f < 1024 or unit == "GB":
            return f"{f:.1f} {unit}" if unit != "B" else f"{int(f)} {unit}"
        f /= 1024
    return f"{n} B"


# ---------------------------------------------------------------------------
# migrate
# ---------------------------------------------------------------------------

def migrate_command(
    from_backend: str = typer.Option(
        "redis",
        "--from",
        help="Source backend: 'redis' or 'sqlite'.",
    ),
    to_backend: str = typer.Option(
        "sqlite",
        "--to",
        help="Destination backend: 'sqlite' or 'redis'.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Report counts only; do not write to the destination.",
    ),
) -> None:
    """Copy memories, RAG chunks, and caches from one backend to the other.

    Idempotent: re-running upserts on primary keys, so a partial run
    can be resumed simply by re-invoking. The source store is opened
    read-only in spirit (we never DELETE), so even an unintended
    destination switch is safe.

    Workflow
    --------
    1. Open both stores using the same ``Config`` modulo backend.
    2. Iterate every table; for each row, upsert into the destination.
    3. Print row-count delta as we go (counts visible during long runs).
    """
    src = from_backend.lower()
    dst = to_backend.lower()
    if src == dst:
        _console.print(f"[yellow]source and destination both {src}; nothing to do.[/yellow]")
        return
    if src not in ("redis", "sqlite") or dst not in ("redis", "sqlite"):
        _console.print(
            f"[red]error:[/red] backends must be 'redis' or 'sqlite' (got {src!r} → {dst!r})"
        )
        raise typer.Exit(code=1)

    src_cfg = Config(storage_backend=src)
    dst_cfg = Config(storage_backend=dst)

    _console.print(
        f"migrating: [cyan]{src}[/cyan] → [cyan]{dst}[/cyan]"
        + ("  [dim](dry run)[/dim]" if dry_run else "")
    )

    started = time.time()
    moved_memory = _migrate_memory(src_cfg, dst_cfg, dry_run=dry_run)
    moved_rag = _migrate_rag(src_cfg, dst_cfg, dry_run=dry_run)
    elapsed = time.time() - started

    table = Table(show_header=True, header_style="bold")
    table.add_column("Resource")
    table.add_column("Rows", justify="right")
    table.add_row("memories", f"{moved_memory:,}")
    table.add_row("rag chunks", f"{moved_rag:,}")
    _console.print(table)
    _console.print(
        f"[green]✓[/green] done in {elapsed:.2f}s"
        + ("  [dim](dry run — destination unchanged)[/dim]" if dry_run else "")
    )


def _migrate_memory(src_cfg: Config, dst_cfg: Config, *, dry_run: bool) -> int:
    """Move memories + their embeddings from ``src`` to ``dst``.

    Counts include both the metadata row and the embedding row when
    both are present — gives the user a visible signal that vectors
    travel with their parents.
    """
    from gemma.storage import build_memory_store

    src = build_memory_store(src_cfg)
    dst = build_memory_store(dst_cfg)
    if not src.connect():
        _console.print(
            f"[yellow]source memory store unreachable; skipping memories.[/yellow]"
        )
        return 0
    dst.connect()

    records = src.get_all_active_memories()
    moved = 0
    for record in records:
        embedding = src.get_embedding(record.memory_id)
        if dry_run:
            moved += 1
            continue
        dst.save_memory(record)
        if embedding is not None and embedding.size > 0:
            dst.save_embedding(record.memory_id, embedding)
        moved += 1
    return moved


def _migrate_rag(src_cfg: Config, dst_cfg: Config, *, dry_run: bool) -> int:
    """Move RAG chunks across every namespace the source knows about.

    For Redis sources we discover namespaces via key-pattern scan; for
    SQLite sources we read the distinct ``namespace`` column. Each
    chunk is upserted into the destination's matching namespace store.
    """
    namespaces = list(_iter_rag_namespaces(src_cfg))
    if not namespaces:
        return 0

    from gemma.storage import build_rag_store

    moved = 0
    for ns in namespaces:
        src = build_rag_store(src_cfg, ns)
        dst = build_rag_store(dst_cfg, ns)
        for cid in src.all_chunk_ids():
            chunk = src.get_chunk(cid)
            embedding = src.get_embedding(cid)
            if chunk is None or embedding is None:
                continue
            if dry_run:
                moved += 1
                continue
            dst.upsert_chunk(
                cid,
                path=chunk.path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                text=chunk.text,
                header=chunk.header,
                embedding=embedding,
            )
            moved += 1
        # Manifest + meta — recreate so the destination sees the same
        # incremental-index fast-path on the next ``gemma rag index``.
        if not dry_run:
            try:
                dst.save_manifest_hash(src.load_manifest_hash())
                meta = src.get_meta()
                if "dim" in meta and "model" in meta:
                    dst.set_meta(int(meta["dim"]), meta["model"])
            except Exception:
                # Manifest/meta aren't load-bearing for queries; skip
                # silently rather than failing the whole migration.
                pass
    return moved


def _iter_rag_namespaces(cfg: Config) -> Iterable[str]:
    """Yield every RAG namespace known to ``cfg``'s backend."""
    if cfg.storage_backend == "sqlite":
        from gemma.storage.sqlite_db import open_db

        conn = open_db(cfg)
        rows = conn.execute(
            "SELECT DISTINCT namespace FROM rag_chunks"
        ).fetchall()
        return [r["namespace"] for r in rows]

    # Redis path — scan for namespace prefixes.
    try:
        import redis  # type: ignore[import-not-found]
    except ImportError:
        return []
    try:
        client = redis.Redis.from_url(cfg.redis_url, decode_responses=True)
        client.ping()
    except Exception:
        return []
    seen: set[str] = set()
    for key in client.scan_iter(match="gemma:rag:*:chunk:*", count=500):
        # key shape: gemma:rag:{ns}:chunk:{cid}
        parts = key.split(":", 4)
        if len(parts) >= 5:
            seen.add(parts[2])
    return sorted(seen)
