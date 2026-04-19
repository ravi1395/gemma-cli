"""Benchmark ``rag_query`` when called **as a model-visible tool**.

Backs improvement #17 ("Expose ``rag_query`` as a model-callable
tool"). The plain retriever path is already measured by
``test_bench_retrieval.py``; what this module isolates is the thin
**adapter layer** that sits between the tool-call dispatcher and
the existing ``RAGRetriever.query``:

    dispatcher → ToolSpec (validate args)
              → handler (real gemma.tools.builtins.rag_query)
              → RetrievalHit → .as_dict()
              → ToolResult.ok(payload=...)
              → JSON-serialise into the ``role=tool`` message.

Any regression here is pure serialisation overhead — the target in
the improvement doc is **≤ 5% above the raw retriever cost**.
"""

from __future__ import annotations

import pytest

from gemma.rag.retrieval import RAGRetriever
from gemma.tools.builtins.rag_query import configure_retriever, rag_query
from tests.bench.conftest import CORPUS_SIZES


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_chunks", CORPUS_SIZES)
@pytest.mark.parametrize("k", [5, 20])
def test_rag_tool_e2e(benchmark, store_factory, embedder, n_chunks, k):
    """One full model-side call to ``rag_query``.

    Configures the real handler with a bench retriever, then calls it
    directly — the same code path the dispatcher invokes.  Compare the
    delta against ``test_bench_retrieval.py`` at the same
    ``n_chunks``/``k`` to confirm the adapter stays within the 5%
    ceiling imposed by improvement #17.
    """
    store = store_factory(n_chunks)
    retriever = RAGRetriever(store, embedder)
    configure_retriever(retriever)

    def _run():
        result = rag_query(query="bench query", k=k)
        return result.content

    benchmark(_run)
