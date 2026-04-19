"""``rag_query`` — expose the workspace RAG index as a model-callable tool.

This is a thin adapter over :class:`~gemma.rag.retrieval.RAGRetriever`.
The retriever is injected at runtime via :func:`configure_retriever` so
the tool handler itself remains a plain function with no constructor
dependencies (which is what the ``@tool`` registry expects).

Typical lifecycle
-----------------
1. ``gemma ask`` wires a ``GemmaSession`` → calls ``configure_retriever``.
2. The model issues a ``rag_query`` tool call.
3. The dispatcher calls this handler with ``query`` and optional ``k``.
4. Each ``RetrievalHit`` is serialised to a dict via ``.as_dict()`` and
   the list is JSON-encoded into the ``ToolResult.content`` string.
5. The model receives the JSON string and can cite individual hits.
"""

from __future__ import annotations

import json
from typing import Optional

from gemma.tools.capabilities import Capability
from gemma.tools.registry import ToolResult, ToolSpec, tool


# Module-level retriever injected before any tool call is dispatched.
# None means "not configured" — the handler returns an informative error.
_retriever: Optional[object] = None


def configure_retriever(retriever: object) -> None:
    """Set the RAGRetriever instance used by the ``rag_query`` handler.

    Args:
        retriever: A :class:`~gemma.rag.retrieval.RAGRetriever` instance
                   bound to the current workspace's vector store.
    """
    global _retriever
    _retriever = retriever


@tool(ToolSpec(
    name="rag_query",
    description=(
        "Search the indexed workspace for code or documentation chunks "
        "relevant to a natural-language query. Returns up to ``k`` hits "
        "with file path, line range, relevance score, and text excerpt. "
        "Run ``gemma rag index`` first if the workspace has not been indexed."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language search query.",
            },
            "k": {
                "type": "integer",
                "description": "Maximum number of hits to return (default: 5).",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    capability=Capability.READ,
    timeout_s=10,
    max_output_bytes=64 * 1024,
))
def rag_query(query: str, k: int = 5) -> ToolResult:
    """Query the RAG index and return JSON-encoded hits.

    Args:
        query: Natural-language search string.
        k:     Maximum number of hits (default: 5, clamped to [1, 50]).

    Returns:
        ToolResult whose ``content`` is a JSON object with a ``hits`` list.
    """
    if _retriever is None:
        return ToolResult(
            ok=False,
            error="not_configured",
            content=(
                "rag_query: no retriever configured. "
                "Run `gemma rag index` and use `gemma ask --agent`."
            ),
        )

    k = max(1, min(50, int(k)))

    try:
        hits = _retriever.query(query, k=k)  # type: ignore[union-attr]
    except Exception as exc:
        return ToolResult(ok=False, error="retrieval_error", content=f"retrieval failed: {exc}")

    payload = {"hits": [h.as_dict() for h in hits]}
    return ToolResult(
        ok=True,
        content=json.dumps(payload, separators=(",", ":")),
        metadata={"hit_count": len(hits)},
    )
