"""Brave search backend — stub.

Brave Search API is a paid vendor with a generous free tier. Requires
a ``BRAVE_SEARCH_API_KEY`` env var and a simple ``httpx`` call against
``https://api.search.brave.com/res/v1/web/search``; no SDK needed.

This stub exists so ``cfg.web_search_backend = "brave"`` fails fast
with a clear message instead of a mysterious ``ImportError``.
"""

from __future__ import annotations

from typing import List

from gemma.tools.backends.base import SearchBackend, SearchHit


class BraveBackend(SearchBackend):
    """Brave Search backend stub — raises until implemented.

    To fill this in:

    1. Read ``BRAVE_SEARCH_API_KEY`` from the env inside ``search``.
    2. Issue a GET to ``https://api.search.brave.com/res/v1/web/search``
       with the ``X-Subscription-Token`` header and ``q``/``count``
       query params.
    3. Normalise ``web.results[*].{title,url,description}`` into
       :class:`SearchHit`. Use ``httpx`` with a ``timeout`` matching
       the ``timeout_s`` parameter.
    """

    name: str = "brave"

    def search(
        self, query: str, *, max_results: int = 5, timeout_s: float = 10.0
    ) -> List[SearchHit]:
        """Not yet implemented. Raises :class:`RuntimeError` with a hint."""
        raise RuntimeError(
            "Brave backend is not implemented yet. "
            "Set cfg.web_search_backend='duckduckgo' (default) or "
            "contribute a Brave client in gemma/tools/backends/brave.py."
        )
