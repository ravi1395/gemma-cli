"""DuckDuckGo search backend — default, zero-auth.

Uses the third-party ``ddgs`` library (formerly ``duckduckgo_search``).
Installed via the ``[agent]`` optional dependency:

    pip install 'gemma-cli[agent]'

The dep is imported lazily inside :meth:`DuckDuckGoBackend.search` so
the rest of the CLI keeps working when the extra is absent. A friendly
``RuntimeError`` is raised in that case; the tool layer converts it to
a structured ``ToolResult`` so the agent sees an error string, not a
traceback.
"""

from __future__ import annotations

from typing import List

from gemma.tools.backends.base import SearchBackend, SearchHit


class DuckDuckGoBackend(SearchBackend):
    """DuckDuckGo backed by ``ddgs`` — no API key, no cookies on disk.

    Thread-safety: ``ddgs.DDGS`` is used as a context manager per call,
    so one call per thread is safe out of the box. Item #20 (parallel
    dispatch) relies on this property — verified in the ``ddgs``
    source: each ``DDGS()`` instance owns its own ``httpx.Client``.
    """

    name: str = "duckduckgo"

    def search(
        self, query: str, *, max_results: int = 5, timeout_s: float = 10.0
    ) -> List[SearchHit]:
        """Return up to ``max_results`` hits from DuckDuckGo.

        Args:
            query:       Search query passed verbatim to DuckDuckGo.
            max_results: Upper bound on returned hits. Clamped at 25
                         here as an extra defensive cap — the tool layer
                         already clamps, but defense in depth is cheap.
            timeout_s:   Passed through to ``ddgs``'s ``timeout``.

        Returns:
            List of :class:`SearchHit`. Empty list on zero hits.

        Raises:
            RuntimeError: ``ddgs`` is not installed, or the library
                raised while executing the search (rate limit, DNS, etc.).
        """
        try:
            # Lazy import so gemma-cli still imports cleanly when the
            # agent extra is not installed. Keeps the base CLI's
            # dependency surface identical to pre-#16.
            from ddgs import DDGS  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "DuckDuckGo backend requires the 'ddgs' package. "
                "Install with: pip install 'gemma-cli[agent]'"
            ) from exc

        capped = max(1, min(int(max_results), 25))

        # ``DDGS()`` is a context manager; it cleans up the underlying
        # httpx client even on exception, which matters because the
        # agent loop can be interrupted mid-call.
        try:
            with DDGS(timeout=timeout_s) as client:
                raw = list(client.text(query, max_results=capped))
        except Exception as exc:  # ddgs raises its own class hierarchy
            # Re-raise as the contract type so the tool layer handles
            # every backend's failures identically.
            raise RuntimeError(f"DuckDuckGo search failed: {exc}") from exc

        hits: List[SearchHit] = []
        for item in raw:
            # ``ddgs`` returns dicts with keys 'title', 'href', 'body'
            # for the text search. Newer versions also return 'url' —
            # accept either so a library upgrade doesn't silently
            # break us.
            url = item.get("href") or item.get("url") or ""
            if not url:
                # Skip rows with no URL — a hit with no link is useless
                # to the agent and breaks downstream ``http_get``.
                continue
            hits.append(SearchHit(
                title=str(item.get("title", "")).strip(),
                url=str(url).strip(),
                snippet=str(item.get("body") or item.get("snippet") or "").strip(),
            ))

        return hits
