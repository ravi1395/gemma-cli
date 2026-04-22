"""Per-session Redis connection pool (#3).

Every subsystem (memory, cache, RAG text, RAG binary) used to open its
own TCP socket to Redis. With ``GemmaSession`` (#13) in place we can
build one :class:`redis.ConnectionPool` per CLI invocation and hand
decoded and binary clients out of it, so a single ``gemma ask`` pays
one TCP handshake instead of three-to-four.

Design choices:

* **Not a module-level singleton.** The pool is scoped to a
  :class:`~gemma.session.GemmaSession` and disposed in ``close()``.
  Long-running processes (the ``gemma chat`` REPL) still get pool
  reuse for their whole lifetime; tests never leak sockets across
  runs.
* **Additive.** Every downstream store accepts an optional ``pool``
  kwarg. When absent the old "build my own client" path still works,
  so tests that inject a fakeredis ``client`` directly are unaffected.
"""

from __future__ import annotations

from typing import Any, Optional

from gemma.config import Config


def pool_for(cfg: Config) -> Optional[Any]:
    """Return a fresh :class:`redis.ConnectionPool` for the URL in ``cfg``.

    Returns None when the ``redis`` package is not installed or when
    pool creation fails (e.g. malformed URL). Callers should treat None
    as "no pool — fall back to per-store client construction".
    """
    try:
        import redis as _redis  # type: ignore
    except ImportError:
        return None
    try:
        # decode_responses=True must be set on the pool itself, not just on
        # the per-client Redis() constructor. redis-py 7.x ignores the
        # decode_responses kwarg passed to Redis() when a connection_pool is
        # supplied — it uses the pool's own setting instead. Without this,
        # every text client built from the pool (memory, cache, RAG) would
        # silently receive bytes where it expects str, causing key-construction
        # bugs (e.g. b'chunk_id' interpolated into f-strings) and corrupted
        # hgetall/smembers results.
        return _redis.ConnectionPool.from_url(cfg.redis_url, decode_responses=True)
    except Exception:
        return None


def client_from_pool(pool: Any, *, decode_responses: bool) -> Optional[Any]:
    """Build a :class:`redis.Redis` bound to ``pool``.

    Kept as a tiny helper so the two decode modes (str and bytes) don't
    drift. Returns None when redis-py is unavailable.
    """
    try:
        import redis as _redis  # type: ignore
    except ImportError:
        return None
    try:
        return _redis.Redis(connection_pool=pool, decode_responses=decode_responses)
    except Exception:
        return None


def disconnect(pool: Optional[Any]) -> None:
    """Best-effort pool teardown. Swallows errors so ``close()`` never raises."""
    if pool is None:
        return
    try:
        pool.disconnect()
    except Exception:
        pass
