"""Tavily search backend — stub.

Tavily is an LLM-optimised search API that returns pre-summarised
snippets alongside URLs. Integration requires a ``TAVILY_API_KEY`` env
var and the ``tavily-python`` SDK; neither is shipped by default.

This module is a placeholder so ``cfg.web_search_backend = "tavily"``
does not cause an ``ImportError`` deep in the registry — instead, the
tool surfaces a clear "not implemented yet" message.

Leave the class signature unchanged when filling this in: the
``web_search`` builtin relies on :meth:`TavilyBackend.search` matching
:class:`gemma.tools.backends.base.SearchBackend`.
"""

from __future__ import annotations

from typing import List

from gemma.tools.backends.base import SearchBackend, SearchHit


class TavilyBackend(SearchBackend):
    """Tavily backend stub — raises until implemented.

    To fill this in:

    1. ``pip install tavily-python`` (add as an optional extra).
    2. Read ``TAVILY_API_KEY`` from the environment inside ``search``
       (never store it on the instance — backends are short-lived).
    3. Call ``TavilyClient(api_key=...).search(query, max_results=...)``
       and normalise each result into :class:`SearchHit`. Tavily also
       returns a ``score`` field; stash it in ``SearchHit.extra``.
    """

    name: str = "tavily"

    def search(
        self, query: str, *, max_results: int = 5, timeout_s: float = 10.0
    ) -> List[SearchHit]:
        """Not yet implemented. Raises :class:`RuntimeError` with a hint."""
        raise RuntimeError(
            "Tavily backend is not implemented yet. "
            "Set cfg.web_search_backend='duckduckgo' (default) or "
            "contribute a Tavily client in gemma/tools/backends/tavily.py."
        )
