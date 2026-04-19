"""Pluggable search backends for the ``web_search`` builtin.

The agent needs a way to answer "what's on the web about X?" without
hard-coding a single vendor. This package holds:

* :mod:`base` — the :class:`SearchBackend` Protocol and :class:`SearchHit`
  dataclass every backend returns.
* :mod:`duckduckgo` — default backend, zero-auth, uses the ``ddgs``
  library. No API key required; ideal for a local-first CLI.
* :mod:`tavily`, :mod:`brave` — stubs that raise a helpful error
  until an API key is provided and the integration is filled in.

Backend selection lives in ``Config.web_search_backend`` so users (and
profiles) can swap vendors without code changes.
"""

from __future__ import annotations

from gemma.tools.backends.base import SearchBackend, SearchHit

__all__ = ["SearchBackend", "SearchHit"]
