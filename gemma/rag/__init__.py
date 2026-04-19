"""Local RAG over the workspace — Phase 6.2.

Public API
----------
This package gives the rest of gemma-cli a single import surface:

    from gemma.rag import RAGIndexer, RAGRetriever, resolve_namespace

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

from gemma.rag.indexer import IndexStats, RAGIndexer
from gemma.rag.manifest import FileEntry, Manifest, ManifestDiff
from gemma.rag.namespace import resolve_namespace
from gemma.rag.retrieval import RAGRetriever, RetrievalHit
from gemma.rag.store import RedisVectorStore

__all__ = [
    "FileEntry",
    "IndexStats",
    "Manifest",
    "ManifestDiff",
    "RAGIndexer",
    "RAGRetriever",
    "RedisVectorStore",
    "RetrievalHit",
    "resolve_namespace",
]
