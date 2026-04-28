"""Local RAG over the workspace — Phase 6.2.

Public API
----------
This package gives the rest of gemma-cli a single import surface:

    from gemma.rag import RAGIndexer, RAGRetriever, resolve_namespace

Heavy submodules (``indexer``, ``store``, ``retrieval``) pull in NumPy
and the embedding stack, so the public attributes are exposed via PEP
562 ``__getattr__``. Submodule imports like
``from gemma.rag.namespace import resolve_namespace`` no longer drag
NumPy in via the package init; the cost is paid only when a consumer
reaches for one of the heavy names.

Design
------
* **Redis-only, no RedisSearch.** Embeddings are stored as raw float32
  bytes under ``gemma:rag:{ns}:embed:{id}`` and similarity is computed
  client-side with NumPy. This keeps the user's setup minimal (plain
  ``redis:7`` is enough — no ``redis-stack`` requirement) and lets our
  tests reuse the existing ``fakeredis`` fixture.
* **Incremental re-indexing** via a Redis-backed manifest. Only files
  whose ``(mtime_ns, size, sha1)`` changed are re-chunked and
  re-embedded; everything else is a no-op. A typical "reindex after
  editing one file" is a handful of writes, not a full rebuild.
* **Branch- and workspace-scoped namespaces** so two repos on the same
  machine, or two branches of the same repo, never contaminate each
  other's retrieval.

See :mod:`gemma.rag.manifest`, :mod:`gemma.rag.store`,
:mod:`gemma.rag.indexer`, :mod:`gemma.rag.retrieval`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — type-checker only
    from gemma.rag.indexer import IndexStats, RAGIndexer
    from gemma.rag.manifest import FileEntry, Manifest, ManifestDiff
    from gemma.rag.namespace import resolve_namespace
    from gemma.rag.retrieval import RAGRetriever, RetrievalHit
    from gemma.rag.store import RedisVectorStore


_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "FileEntry": ("gemma.rag.manifest", "FileEntry"),
    "IndexStats": ("gemma.rag.indexer", "IndexStats"),
    "Manifest": ("gemma.rag.manifest", "Manifest"),
    "ManifestDiff": ("gemma.rag.manifest", "ManifestDiff"),
    "RAGIndexer": ("gemma.rag.indexer", "RAGIndexer"),
    "RAGRetriever": ("gemma.rag.retrieval", "RAGRetriever"),
    "RedisVectorStore": ("gemma.rag.store", "RedisVectorStore"),
    "RetrievalHit": ("gemma.rag.retrieval", "RetrievalHit"),
    "resolve_namespace": ("gemma.rag.namespace", "resolve_namespace"),
}


def __getattr__(name: str):
    """Lazy attribute resolver — see module docstring."""
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )
    import importlib

    submodule = importlib.import_module(target[0])
    value = getattr(submodule, target[1])
    globals()[name] = value
    return value


__all__ = sorted(_LAZY_ATTRS)
