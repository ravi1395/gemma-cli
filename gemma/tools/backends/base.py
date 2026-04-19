"""Backend Protocol and hit dataclass shared by every search backend.

Every concrete backend (DuckDuckGo, Tavily, Brave …) implements the
:class:`SearchBackend` Protocol. The ``web_search`` builtin looks up
the backend by name and calls :meth:`SearchBackend.search`, so adding
a new vendor is a matter of dropping a module in ``gemma/tools/backends/``
and wiring it into :func:`gemma.tools.builtins.web_search._get_backend`.

Design choices
--------------
* **Protocol, not ABC.** Structural typing keeps the backends
  independent — a test fake that merely implements ``search(...)`` is
  a valid backend without any inheritance dance.
* **Plain dataclass for hits.** ``title``/``url``/``snippet`` is the
  LCD across vendors; anything vendor-specific goes in the free-form
  ``extra`` dict so the tool layer can stay vendor-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol


@dataclass(frozen=True)
class SearchHit:
    """One result from a web search, normalised across backends.

    Attributes:
        title: Page title as reported by the search engine. May be
            empty when the backend can't extract a title.
        url: Canonical URL of the result. Always absolute HTTPS —
            backends that return relative URLs must normalise before
            constructing a ``SearchHit``.
        snippet: Short text excerpt the engine shows under the title.
            Empty string is acceptable but preferred over ``None`` so
            downstream JSON serialisation stays boring.
        extra: Vendor-specific fields (e.g. Tavily's ``score``, Brave's
            ``family_friendly`` flag). The tool layer does not surface
            this to the model — keep it opaque.
    """

    title: str
    url: str
    snippet: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


class SearchBackend(Protocol):
    """Search backend Protocol.

    Implementations should be cheap to instantiate — we construct a
    fresh backend per ``web_search`` call so the process can be killed
    at any point without leaking sockets. Long-lived state (cookies,
    rate-limit counters) belongs inside a backend's own module, not on
    the instance.
    """

    #: Stable identifier used in ``Config.web_search_backend``.
    #: Must match the key registered in ``web_search._get_backend``.
    name: str

    def search(
        self, query: str, *, max_results: int = 5, timeout_s: float = 10.0
    ) -> List[SearchHit]:
        """Run ``query`` and return up to ``max_results`` hits.

        Args:
            query:       Natural-language search query.
            max_results: Soft cap on the number of hits to return. The
                         caller clamps this, backends should treat it
                         as a request, not a guarantee (e.g. the
                         engine may return fewer legitimately).
            timeout_s:   Wall-clock budget. Backends should raise or
                         return the partial hits they have when this
                         is exceeded.

        Returns:
            A list of :class:`SearchHit` — possibly empty, never None.

        Raises:
            RuntimeError: Backend dependency not installed, credentials
                missing, or the engine returned a non-retriable error.
                The ``web_search`` tool converts this to a structured
                ``ToolResult`` so the agent loop keeps running.
        """
        ...
