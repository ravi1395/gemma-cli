"""Per-CLI-invocation resource holder for gemma-cli.

GemmaSession owns the Redis client, memory manager, embedder, and response
cache for a single CLI invocation. Every command handler that previously
built these resources by hand now delegates to the session, so the wiring
lives in exactly one place (#13).

All properties are constructed lazily on first access via
``@cached_property`` and the Redis client is released in ``close()`` /
``__exit__``.

Usage::

    with GemmaSession(cfg) as session:
        session.memory.record_turn("user", prompt)
        cache = ResponseCache.eligible(cfg, no_stream=True, no_cache=False,
                                       prebuilt=session.cache)
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from gemma.config import Config

if TYPE_CHECKING:
    from gemma.cache import ResponseCache
    from gemma.embeddings import Embedder
    from gemma.memory import MemoryManager


class GemmaSession:
    """Holds Redis client, memory, embedder, and cache for one CLI invocation.

    Resources are lazy — importing or constructing GemmaSession costs
    nothing until a property is first accessed. The Redis connection is
    opened at most once and shared across the memory manager and the
    response cache for the life of the invocation.
    """

    def __init__(self, cfg: Config) -> None:
        """Create a session bound to a Config.

        Args:
            cfg: Active Config for this invocation. All properties derive
                 their settings (redis_url, embedding_model, etc.) from it.
        """
        self._cfg = cfg
        # Per-invocation memoisation for ``git rev-parse --abbrev-ref HEAD``
        # (#8). Keyed by resolved absolute path so two subcommands run in
        # the same session fork git at most once per workspace root.
        self._branch_cache: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "GemmaSession":
        """Return self so ``with GemmaSession(cfg) as session:`` works."""
        return self

    def __exit__(self, *_: object) -> None:
        """Release resources when leaving the ``with`` block."""
        self.close()

    def close(self) -> None:
        """Explicitly close Redis connections opened by this session.

        Only cleans up resources that were actually built: ``cached_property``
        stores its result in ``self.__dict__`` on first access, so we can
        check the dict directly without triggering lazy construction.
        """
        client = self.__dict__.get("redis_client")
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        # Tear down the shared ConnectionPool if one was built (#3). The
        # client's ``close()`` above releases its own connection; the pool
        # itself owns any idle sockets that were checked back in.
        pool = self.__dict__.get("redis_pool")
        if pool is not None:
            from gemma.redis_pool import disconnect
            disconnect(pool)

    # ------------------------------------------------------------------
    # Lazy resources
    # ------------------------------------------------------------------

    @cached_property
    def redis_pool(self) -> Optional[Any]:
        """Shared :class:`redis.ConnectionPool` for the life of this session (#3).

        Every subsystem that wants to talk to Redis pulls a client from
        this pool so the invocation pays one TCP handshake, not three.
        Returns None when the ``redis`` package is not installed or pool
        construction fails (e.g. malformed URL).
        """
        from gemma.redis_pool import pool_for
        return pool_for(self._cfg)

    @cached_property
    def redis_client(self) -> Optional[Any]:
        """A single decode_responses=True Redis client for this invocation.

        Built from the shared :attr:`redis_pool` so memory, cache, and
        RAG all multiplex over the same TCP connection. Returns None when
        the ``redis`` package is absent or the server is unreachable.
        """
        try:
            import redis as _redis
        except ImportError:
            return None
        try:
            if self.redis_pool is not None:
                client = _redis.Redis(connection_pool=self.redis_pool, decode_responses=True)
            else:
                client = _redis.Redis.from_url(self._cfg.redis_url, decode_responses=True)
            client.ping()
            return client
        except Exception:
            return None

    @cached_property
    def memory(self) -> "MemoryManager":
        """MemoryManager wired to the shared Redis client.

        ``initialize()`` is called here so command handlers do not need to
        know about the two-step construction protocol.

        Returns:
            An initialised MemoryManager. Inspect ``.degraded`` or
            ``.available`` after access to check Redis reachability.
        """
        from gemma.memory import MemoryManager
        from gemma.memory.store import MemoryStore

        # Inject the already-connected client so MemoryStore reuses the
        # same connection. If redis_client is None (Redis unavailable),
        # MemoryStore will try its own connection and enter degraded mode.
        store = MemoryStore(self._cfg, client=self.redis_client, pool=self.redis_pool)
        mgr = MemoryManager(self._cfg, store=store)
        mgr.initialize()
        return mgr

    @cached_property
    def embedder(self) -> "Embedder":
        """Shared Embedder (Ollama embedding model).

        Returns an Embedder configured for ``cfg.embedding_model``. Failures
        (Ollama unreachable, model not pulled) surface at call time via the
        Embedder's own error handling, not here.
        """
        from gemma.embeddings import Embedder

        return Embedder(
            model=self._cfg.embedding_model,
            host=self._cfg.ollama_host,
            keep_alive=self._cfg.ollama_keep_alive,
        )

    @cached_property
    def cache(self) -> Optional["ResponseCache"]:
        """ResponseCache backed by the shared Redis client, or None.

        Returns None when ``cfg.cache_enabled`` is False or Redis is
        unavailable. Per-request eligibility (no_stream, no_cache,
        temperature) must still be evaluated by the caller — typically via
        ``ResponseCache.eligible(..., prebuilt=session.cache)``.
        """
        if not self._cfg.cache_enabled or self.redis_client is None:
            return None
        from gemma.cache import ResponseCache

        return ResponseCache(self.redis_client, self._cfg.cache_ttl_seconds)

    # ------------------------------------------------------------------
    # Cross-invocation helpers
    # ------------------------------------------------------------------

    def branch_for(self, root: Path) -> str:
        """Return the effective branch name for ``root``, memoised per session (#8).

        ``gemma rag`` subcommands used to fork ``git rev-parse
        --abbrev-ref HEAD`` every time they resolved a namespace. In a
        single CLI invocation that's the same branch every call; fork
        once and reuse.

        The returned value is the branch name or the fallback
        (``_default``) — never None. This way callers pass the result
        directly to :func:`resolve_namespace(root, branch=...)` without
        re-triggering branch detection.

        Args:
            root: Workspace root (resolved to an absolute path before
                  caching so ``./foo`` and ``/abs/foo`` share an entry).
        """
        key = str(Path(root).resolve())
        if key in self._branch_cache:
            cached = self._branch_cache[key]
            assert cached is not None  # we only ever cache resolved values
            return cached
        # Import lazily so ``gemma.session`` stays dependency-light.
        from gemma.rag.namespace import _FALLBACK_BRANCH, _detect_branch
        branch = _detect_branch(Path(key)) or _FALLBACK_BRANCH
        self._branch_cache[key] = branch
        return branch
