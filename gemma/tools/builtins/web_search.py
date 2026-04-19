"""``web_search`` — run a search query against a pluggable backend.

This is the agent's way to *find* URLs it then fetches with
``http_get``. Kept separate from ``http_get`` for two reasons:

1. Different capability posture. ``web_search`` hits a single vendor
   endpoint; ``http_get`` hits arbitrary allowlisted hosts. Splitting
   them means a user who disallows ``web_search`` (e.g. for privacy)
   can still let the agent call ``http_get`` against docs sites.
2. Different output shape. Search results are a structured list of
   ``(title, url, snippet)`` triples — not a body of text. The model
   benefits from seeing this as JSON rather than scraping it out of
   an HTML page.

Backend selection is driven by ``cfg.web_search_backend``. The default
is ``"duckduckgo"`` (zero-auth, ships with the ``[agent]`` extra).
Alternative backends (``tavily``, ``brave``) are stubs today — see
``gemma/tools/backends/``.
"""

from __future__ import annotations

import json
from typing import Optional

from gemma.config import Config
from gemma.tools.backends.base import SearchBackend
from gemma.tools.capabilities import Capability
from gemma.tools.registry import ToolResult, ToolSpec, tool


#: Hard upper bound on results per call. The model almost never needs
#: more than a handful; higher numbers waste tokens on low-relevance
#: hits. Users who want a different value should filter in a second
#: call, not ask for 50 up front.
_MAX_RESULTS_CAP = 10


def _get_backend(name: str) -> SearchBackend:
    """Instantiate the backend registered under ``name``.

    New backends are added by importing them here and extending the
    dispatch dict below. Keeping registration explicit (rather than an
    entry-point scan) makes the supported set visible at a glance and
    avoids import-time surprises from third-party packages.

    Args:
        name: Value of ``cfg.web_search_backend`` — case-insensitive.

    Returns:
        A fresh :class:`SearchBackend` instance.

    Raises:
        ValueError: Unknown backend name. Caller should translate to a
            structured ``ToolResult`` so the model sees an actionable
            error string.
    """
    # Lazy imports inside the function so a missing optional extra
    # (e.g. ``ddgs``) only bites when the user actually selects that
    # backend — not on first import of gemma.tools.builtins.
    key = (name or "").lower().strip()

    if key == "duckduckgo":
        from gemma.tools.backends.duckduckgo import DuckDuckGoBackend
        return DuckDuckGoBackend()
    if key == "tavily":
        from gemma.tools.backends.tavily import TavilyBackend
        return TavilyBackend()
    if key == "brave":
        from gemma.tools.backends.brave import BraveBackend
        return BraveBackend()

    raise ValueError(
        f"unknown web_search backend {name!r}. "
        f"Supported: 'duckduckgo' (default), 'tavily', 'brave'."
    )


@tool(ToolSpec(
    name="web_search",
    description=(
        "Search the public web for a query and return a JSON list of hits "
        "with title, URL, and a short snippet. Use this to discover URLs, "
        "then use http_get to fetch the full page content. The default "
        "backend is DuckDuckGo (no API key required)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language search query.",
            },
            "max_results": {
                "type": "integer",
                "description": (
                    f"Upper bound on results returned (1..{_MAX_RESULTS_CAP}). "
                    "Defaults to 5."
                ),
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    capability=Capability.NETWORK,
    timeout_s=15,
    max_output_bytes=32 * 1024,
))
def web_search(
    query: str,
    max_results: int = 5,
    *,
    _cfg: Optional[Config] = None,
) -> ToolResult:
    """Run ``query`` through the configured backend and return JSON hits.

    Args:
        query:       The search query. Empty / whitespace-only queries
                     are rejected so we don't burn a round-trip.
        max_results: How many hits to return. Clamped to ``[1,
                     _MAX_RESULTS_CAP]``.
        _cfg:        Injection point for tests to pick a specific
                     backend without touching user config. Normal
                     callers (the dispatcher, ``gemma tools run``)
                     leave this as ``None`` and a fresh :class:`Config`
                     is used.

    Returns:
        A :class:`ToolResult` whose ``content`` is a JSON string of
        ``[{"title", "url", "snippet"}, ...]`` on success. On failure
        ``ok=False`` and ``content`` is a short human-readable error.
    """
    if not query or not query.strip():
        return ToolResult(
            ok=False, error="invalid_args",
            content="query must be a non-empty string",
        )

    # Clamp on the tool side so the backend never has to care about
    # user-supplied extremes.
    capped = max(1, min(int(max_results), _MAX_RESULTS_CAP))

    cfg = _cfg if _cfg is not None else Config()

    try:
        backend = _get_backend(cfg.web_search_backend)
    except ValueError as exc:
        return ToolResult(
            ok=False, error="unknown_backend", content=str(exc),
        )

    try:
        hits = backend.search(
            query.strip(),
            max_results=capped,
            timeout_s=float(cfg.web_search_timeout_s),
        )
    except RuntimeError as exc:
        # Backend reported a dependency / credential / transient issue.
        # Surface as a structured error so the model can try a
        # different strategy rather than retrying the same failing call.
        return ToolResult(
            ok=False, error="backend_error", content=str(exc),
        )

    # Emit only the model-facing fields (title/url/snippet). The
    # free-form ``extra`` dict stays internal.
    payload = [
        {"title": h.title, "url": h.url, "snippet": h.snippet}
        for h in hits
    ]
    content = json.dumps(payload, ensure_ascii=False)

    return ToolResult(
        ok=True,
        content=content,
        metadata={
            "backend": backend.name,
            "result_count": len(hits),
            "query": query.strip(),
        },
    )


# Re-export the constants so tests can reference the cap without
# reaching into internals. Keeps the test file's imports tidy.
__all__ = ["web_search", "_MAX_RESULTS_CAP", "_get_backend"]
