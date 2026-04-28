"""Storage backends — Redis (legacy) and SQLite (default).

Three storage concerns flow through this package:

1. **Memory store** — condensed memories + their embeddings + the per-
   session sliding window of recent turns.
2. **RAG store** — workspace document chunks + embeddings + the embed
   cache (content-hash → vector).
3. **Response cache** — SHA-keyed prompt/response cache.

Each concern has a Redis implementation that's been in use since v0.1
and a SQLite implementation introduced in v0.3 as the default. The
SQLite path stores everything in a single file at
``~/.gemma/store.sqlite``; vector search is brute-force cosine in
numpy (no extension required, plenty fast at our scale — see
``docs/storage.md``).

Pick a backend at runtime via ``Config.storage_backend``. The factory
helpers in this module are the only seam call sites should depend on
— a future swap to LanceDB or sqlite-vec is a one-place change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from gemma.config import Config


def build_memory_store(config: "Config", *, client=None, pool=None):
    """Return a memory store matching ``config.storage_backend``.

    Args:
        config: Runtime configuration. Reads ``storage_backend`` and the
            backend-specific connection info (``redis_url`` or
            ``sqlite_path``).
        client: Optional Redis client (legacy DI for tests). Ignored on
            the SQLite path.
        pool:   Optional Redis ``ConnectionPool`` (set by ``GemmaSession``
            to share connections across stores). Ignored on the SQLite
            path.

    Returns:
        An object satisfying the legacy ``MemoryStore`` public surface
        — same method names, same return shapes — so call sites need
        no awareness of which backend is active.
    """
    name = (getattr(config, "storage_backend", None) or "sqlite").lower()
    if name == "sqlite":
        from gemma.storage.sqlite_memory import SQLiteMemoryStore

        return SQLiteMemoryStore(config)
    if name == "redis":
        # Import lives inside the legacy module so a SQLite-only install
        # never pulls the ``redis`` package into the import graph.
        from gemma.memory.store import MemoryStore as RedisMemoryStore

        return RedisMemoryStore(config, client=client, pool=pool)
    raise ValueError(
        f"Unknown storage_backend {name!r}. Use 'sqlite' (default) or 'redis'."
    )


def build_rag_store(config: "Config", namespace: str, *, pool=None):
    """Return a RAG vector store matching ``config.storage_backend``.

    Args:
        config:    Runtime configuration.
        namespace: Per-workspace partition key (e.g. ``sha256(repo):branch``).
            All RAG rows are scoped to this string so multiple
            checkouts share the same store file without colliding.
        pool:      Optional Redis ``ConnectionPool`` (legacy path only).
    """
    name = (getattr(config, "storage_backend", None) or "sqlite").lower()
    if name == "sqlite":
        from gemma.storage.sqlite_rag import SQLiteRAGStore

        return SQLiteRAGStore(config, namespace)
    if name == "redis":
        from gemma.rag.store import RedisVectorStore

        return RedisVectorStore(
            namespace, redis_url=config.redis_url, pool=pool
        )
    raise ValueError(
        f"Unknown storage_backend {name!r}. Use 'sqlite' (default) or 'redis'."
    )


def build_response_cache(config: "Config", *, pool=None) -> Optional[object]:
    """Return a response cache matching ``config.storage_backend``.

    Returns ``None`` when the cache is disabled
    (``cache_enabled=False`` or ``cache_ttl_seconds<=0``) — callers
    treat ``None`` as "skip caching" already.
    """
    if not getattr(config, "cache_enabled", True):
        return None
    if int(getattr(config, "cache_ttl_seconds", 0) or 0) <= 0:
        return None
    name = (getattr(config, "storage_backend", None) or "sqlite").lower()
    if name == "sqlite":
        from gemma.storage.sqlite_cache import SQLiteResponseCache

        return SQLiteResponseCache(config)
    if name == "redis":
        # Defer to the legacy builder which handles pool wiring + redis
        # availability probing.
        from gemma.cache import build_cache

        return build_cache(config, pool=pool)
    raise ValueError(
        f"Unknown storage_backend {name!r}. Use 'sqlite' (default) or 'redis'."
    )


__all__ = ["build_memory_store", "build_rag_store", "build_response_cache"]
