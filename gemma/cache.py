"""SHA-keyed response cache backed by Redis.

Caches full LLM responses for deterministic, low-temperature calls to
eliminate redundant Ollama round-trips. The cache key is a SHA256 hash of
the prompt inputs: model, temperature, system prompt, user prompt, and
keep_alive. Entries are stored as JSON blobs with a configurable TTL.

Only non-streaming calls whose temperature is at or below
``Config.cache_temperature_max`` are eligible for caching. Streaming calls
and high-temperature outputs are never cached.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from gemma.config import Config

# Redis key prefix for all cache entries.
_K_PREFIX = "gemma:cache:"


class ResponseCache:
    """Redis-backed cache for deterministic LLM responses.

    Cache keys are SHA256 hashes computed from the model name, temperature,
    system prompt, user prompt, and keep-alive value. Entries carry a TTL set
    at construction time; TTL=0 disables all caching.

    Failures (Redis unavailable, serialisation errors) are silently swallowed
    so the cache is always transparent to callers — a broken cache falls back
    to normal model calls.
    """

    def __init__(self, redis_client: Any, ttl_seconds: int) -> None:
        """Initialise the cache.

        Args:
            redis_client: A connected ``redis.Redis`` instance
                          (``decode_responses=True``).
            ttl_seconds:  Entry TTL in seconds. 0 disables all caching.
        """
        self._client = redis_client
        self._ttl = ttl_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, messages: list[dict], config: "Config") -> Optional[str]:
        """Return a cached response, or None on miss / disabled / error.

        Args:
            messages: The message list sent to the model.
            config:   Active Config (model, temperature, keep_alive).

        Returns:
            The cached content string on a hit, or None on a miss.
        """
        if self._ttl == 0 or self._client is None:
            return None
        try:
            key = self._compute_key(messages, config)
            raw = self._client.get(key)
            if raw is None:
                return None
            data = json.loads(raw)
            return data.get("content")
        except Exception:
            return None

    def put(self, messages: list[dict], config: "Config", content: str) -> None:
        """Store a response in the cache.

        Args:
            messages: The message list (used to compute the cache key).
            config:   Active Config (model, temperature, keep_alive).
            content:  The full response text to cache.
        """
        if self._ttl == 0 or self._client is None:
            return
        try:
            key = self._compute_key(messages, config)
            payload = json.dumps(
                {"content": content, "created_at": int(time.time())}
            )
            self._client.set(key, payload, ex=self._ttl)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_key(messages: list[dict], config: "Config") -> str:
        """Compute the SHA256 cache key for a prompt.

        Key shape: ``sha256(model \\0 temperature \\0 system \\0 user \\0 keep_alive)``

        The last ``system`` role entry and the last ``user`` role entry in
        *messages* are used; all intermediate turns are excluded from the key
        per the design spec.

        Args:
            messages: Message list in ``[{"role": ..., "content": ...}]`` form.
            config:   Active Config supplying model / temperature / keep_alive.

        Returns:
            Redis key string: ``"gemma:cache:<sha256-hex>"``.
        """
        system_content = ""
        user_content = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_content = content
            elif role == "user":
                user_content = content  # last user message wins

        raw = "\0".join([
            config.model,
            str(config.temperature),
            system_content,
            user_content,
            str(config.ollama_keep_alive),
        ])
        sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"{_K_PREFIX}{sha}"


def build_cache(config: "Config") -> Optional[ResponseCache]:
    """Create a ResponseCache from config, or None if unavailable.

    Returns None (silently) when:
    - ``config.cache_enabled`` is False
    - the ``redis`` package is not installed
    - Redis is unreachable at ``config.redis_url``

    Args:
        config: Active Config with cache_enabled, cache_ttl_seconds, redis_url.

    Returns:
        A connected ResponseCache, or None.
    """
    if not config.cache_enabled:
        return None
    try:
        import redis as _redis  # type: ignore
    except ImportError:
        return None
    try:
        client = _redis.Redis.from_url(config.redis_url, decode_responses=True)
        client.ping()
        return ResponseCache(client, config.cache_ttl_seconds)
    except Exception:
        return None
