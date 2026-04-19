"""Per-session memoization for agentic tool calls.

Inside a single ``gemma ask`` invocation the model sometimes issues the
same tool call more than once (e.g. two ``rag_query`` calls with identical
arguments on successive turns). This module provides a lightweight in-process
dict that short-circuits duplicate calls so the handler is not re-executed.

Safety constraints
------------------
* Only READ-capability results are cached — WRITE and ARCHIVE side-effects
  must never be skipped.
* Every call (cache hit or miss) still produces an audit record via
  :mod:`gemma.tools.audit`. Cache hits carry ``cached=True`` so a reviewer
  can see the total invocation count even when most calls were served
  from memory.
* The cache lives for the lifetime of one ``_agent_loop`` invocation and
  is discarded afterward — no cross-session bleed.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Dict, Optional, Tuple


# Capability values that are safe to cache.  We keep this as a frozenset of
# strings (rather than importing Capability) so ``gemma.agent.cache`` stays
# import-free of the tools package — the agent module is a leaf, not a hub.
_CACHEABLE_CAPABILITIES: frozenset[str] = frozenset({"read"})


class AgentSessionCache:
    """In-process memoization for READ-capability tool results.

    Each key is a ``(tool_name, canonicalized_args)`` tuple where args are
    serialised to JSON with sorted keys to make the lookup order-independent.
    Values are the ``ToolResult.content`` strings returned by the handler.
    """

    def __init__(self) -> None:
        """Create an empty cache for one agent session."""
        self._store: Dict[Tuple[str, str], str] = {}
        # Guards concurrent get/put from parallel tool dispatch (#20).
        # The dict operations themselves are GIL-protected, but the
        # ``check is_cacheable → put`` sequence issued by _agent_loop
        # and the ``check get → dispatch → put`` sequence across
        # workers are not atomic. The lock makes both idempotent under
        # a concurrent fan-out turn.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, tool: str, args: Dict[str, Any]) -> Optional[str]:
        """Return the cached content string for ``(tool, args)`` or None.

        Args:
            tool: Tool name as registered in the registry.
            args: Argument dict exactly as passed to the handler.

        Returns:
            Previously cached ``ToolResult.content`` string, or ``None``
            if this ``(tool, args)`` combination has not been cached.
        """
        with self._lock:
            return self._store.get(self._key(tool, args))

    def put(self, tool: str, args: Dict[str, Any], content: str) -> None:
        """Store ``content`` for future lookups with the same ``(tool, args)``.

        Args:
            tool:    Tool name.
            args:    Argument dict used in the handler call.
            content: ``ToolResult.content`` string to cache.
        """
        with self._lock:
            self._store[self._key(tool, args)] = content

    @staticmethod
    def is_cacheable(capability: str) -> bool:
        """Return True if results for ``capability`` should be cached.

        Args:
            capability: Capability value string (e.g. ``"read"``, ``"write"``).

        Returns:
            True for READ; False for WRITE, ARCHIVE, NETWORK.
        """
        return capability.lower() in _CACHEABLE_CAPABILITIES

    @property
    def size(self) -> int:
        """Number of entries currently held in the cache."""
        return len(self._store)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _key(tool: str, args: Dict[str, Any]) -> Tuple[str, str]:
        """Stable cache key from tool name + JSON-serialised args (sorted keys)."""
        return (tool, json.dumps(args, sort_keys=True, default=str))
