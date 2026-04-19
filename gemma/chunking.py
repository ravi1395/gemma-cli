"""Text-chunking strategies for gemma-cli.

Consumed by the Phase 6.2 RAG indexer and designed to be reusable by
future memory-condensation work. Three strategies are exposed:

* :func:`chunk_python`  — AST-based; splits on top-level class / function
  boundaries so chunks carry semantic scope.
* :func:`chunk_markdown` — heading-based; each chunk is a section bounded
  by ``^#`` headings so retrieval results correspond to documentable
  units.
* :func:`chunk_sliding` — fallback; fixed-window with overlap for file
  types we don't have a structural chunker for.

Design principles
-----------------
* **Pure functions.** No I/O, no caching, no logging. Callers own the
  file reads and store.
* **Stable chunk IDs.** A chunk's ``id`` is ``sha1(path + start_line +
  text)[:16]``. Identical inputs always produce identical IDs, which is
  what makes incremental re-embedding correct — see the RAG manifest
  design in ``docs/plans/phase-6.2-local-rag.md``.
* **Graceful degradation.** ``chunk_python`` falls back to
  :func:`chunk_sliding` on :class:`SyntaxError` so we still get *some*
  coverage of a half-finished file.
* **Line-based slicing.** We track ``start_line`` / ``end_line``
  (1-indexed, inclusive) so citation footers can render
  ``path:42-97`` directly.
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Chunk:
    """A slice of a source file ready for embedding.

    Attributes:
        id:         Stable 16-hex-char identifier, derived from
                    ``sha1(path + start_line + text)``. Safe to use as a
                    Redis key segment.
        path:       Relative or absolute path of the source file. The
                    chunker does not canonicalise — callers pass whatever
                    form they want to cite later.
        start_line: First line of the slice, 1-indexed and inclusive.
        end_line:   Last line of the slice, 1-indexed and inclusive.
        text:       The slice content, including any leading section
                    heading but without a trailing newline.
        header:     Optional structural context (class/function name,
                    markdown heading). Used to enrich the embedding
                    input — "the chunk is the function ``foo`` in
                    ``handler.py``" is a better retrieval target than
                    the raw text alone.
    """

    id: str
    path: str
    start_line: int
    end_line: int
    text: str
    header: Optional[str] = None

    @property
    def line_range(self) -> str:
        """Human-friendly ``42-97`` / ``42`` form for citation."""
        if self.start_line == self.end_line:
            return str(self.start_line)
        return f"{self.start_line}-{self.end_line}"


# ---------------------------------------------------------------------------
# Chunker dispatch
# ---------------------------------------------------------------------------

def chunk_for_path(source: str, path: str) -> List[Chunk]:
    """Dispatch to the right chunker based on ``path``'s extension.

    This is the typical entry point for the RAG indexer: one call per
    file, let the dispatcher pick the strategy.
    """
    lower = path.lower()
    if lower.endswith(".py"):
        return chunk_python(source, path)
    if lower.endswith((".md", ".markdown", ".rst")):
        return chunk_markdown(source, path)
    return chunk_sliding(source, path)


# ---------------------------------------------------------------------------
# Python AST chunker
# ---------------------------------------------------------------------------

# Top-level functions and classes above this line count get a further
# safety split so very long defs don't blow the embedder's token budget.
_PY_LONG_DEF_THRESHOLD = 200


def chunk_python(source: str, path: str) -> List[Chunk]:
    """Chunk Python source by top-level ``def`` / ``class`` boundaries.

    Imports and module-level statements at the head of the file are
    grouped into an implicit ``"<module-head>"`` chunk so they are still
    retrievable.

    Long defs (above :data:`_PY_LONG_DEF_THRESHOLD` lines) get a further
    split via :func:`chunk_sliding` with the enclosing def's name
    preserved as ``header``.

    Falls back to :func:`chunk_sliding` on :class:`SyntaxError`.
    """
    if not source:
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Partially-edited files are common in dev flows; don't drop them.
        return chunk_sliding(source, path)

    lines = source.splitlines()
    total = len(lines)
    chunks: List[Chunk] = []

    # Collect top-level nodes that deserve their own chunk.
    top_nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]

    # Pure scripts (no top-level defs) have no meaningful structure for
    # AST chunking — fall through to the empty-chunks guard below, which
    # reroutes to sliding window.
    if not top_nodes:
        return chunk_sliding(source, path)

    # ``first_def_line`` is the starting line of the earliest top-level
    # def/class. Everything above it is module-head.
    first_def_line = top_nodes[0].lineno

    if first_def_line > 1:
        head_text = "\n".join(lines[: first_def_line - 1]).rstrip()
        if head_text:
            chunks.append(
                _make_chunk(
                    path=path,
                    start_line=1,
                    end_line=first_def_line - 1,
                    text=head_text,
                    header="<module-head>",
                )
            )

    for node in top_nodes:
        start = node.lineno
        end = getattr(node, "end_lineno", None) or _estimate_end_lineno(node, total)

        body_text = "\n".join(lines[start - 1 : end])
        header = _py_header(node)
        span = end - start + 1

        if span > _PY_LONG_DEF_THRESHOLD:
            # Further split: reuse sliding window, but preserve the header
            # and offset line numbers so citations remain correct.
            for sub in chunk_sliding(body_text, path, start_offset=start):
                chunks.append(
                    _make_chunk(
                        path=sub.path,
                        start_line=sub.start_line,
                        end_line=sub.end_line,
                        text=sub.text,
                        header=header,
                    )
                )
        else:
            chunks.append(
                _make_chunk(
                    path=path,
                    start_line=start,
                    end_line=end,
                    text=body_text,
                    header=header,
                )
            )

    # If the file had no top-level defs at all (e.g. a pure script), fall
    # back to sliding window so we still produce chunks.
    if not chunks:
        return chunk_sliding(source, path)

    return chunks


def _py_header(node: ast.AST) -> str:
    """Render a short header like ``class Foo`` or ``async def bar``."""
    if isinstance(node, ast.AsyncFunctionDef):
        return f"async def {node.name}"
    if isinstance(node, ast.FunctionDef):
        return f"def {node.name}"
    if isinstance(node, ast.ClassDef):
        return f"class {node.name}"
    return type(node).__name__


def _estimate_end_lineno(node: ast.AST, total: int) -> int:
    """Return a best-effort end line when ``end_lineno`` is missing.

    Python 3.8+ provides ``end_lineno`` on every node, so this branch is
    defensive — covers exotic third-party AST tooling that might strip it.
    """
    last = node.lineno
    for child in ast.walk(node):
        if hasattr(child, "lineno"):
            last = max(last, child.lineno)
    return min(last, total)


# ---------------------------------------------------------------------------
# Markdown heading chunker
# ---------------------------------------------------------------------------

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def chunk_markdown(source: str, path: str) -> List[Chunk]:
    """Chunk markdown (or RST) by ATX-style ``# Heading`` boundaries.

    Each chunk is the heading plus all content up to (but not including)
    the next heading of equal or higher level. Files without any heading
    fall back to :func:`chunk_sliding`.
    """
    if not source:
        return []

    lines = source.splitlines()
    total = len(lines)

    # Collect (line_index_1based, heading_text) pairs.
    heading_positions: List[tuple[int, str]] = []
    for idx, line in enumerate(lines, start=1):
        m = _MD_HEADING.match(line)
        if m:
            heading_positions.append((idx, m.group(2).strip()))

    if not heading_positions:
        return chunk_sliding(source, path)

    chunks: List[Chunk] = []

    # Prologue: content above the first heading, if any.
    first_line = heading_positions[0][0]
    if first_line > 1:
        prologue = "\n".join(lines[: first_line - 1]).rstrip()
        if prologue:
            chunks.append(
                _make_chunk(
                    path=path,
                    start_line=1,
                    end_line=first_line - 1,
                    text=prologue,
                    header="<prologue>",
                )
            )

    for i, (start, heading) in enumerate(heading_positions):
        end = (
            heading_positions[i + 1][0] - 1
            if i + 1 < len(heading_positions)
            else total
        )
        body = "\n".join(lines[start - 1 : end]).rstrip()
        if not body:
            continue
        chunks.append(
            _make_chunk(
                path=path,
                start_line=start,
                end_line=end,
                text=body,
                header=heading,
            )
        )

    return chunks


# ---------------------------------------------------------------------------
# Sliding window fallback
# ---------------------------------------------------------------------------

_DEFAULT_WINDOW = 40
_DEFAULT_OVERLAP = 5


def chunk_sliding(
    source: str,
    path: str,
    *,
    window: int = _DEFAULT_WINDOW,
    overlap: int = _DEFAULT_OVERLAP,
    start_offset: int = 1,
) -> List[Chunk]:
    """Fixed-size line-window chunking with overlap.

    Args:
        source:       Text to chunk.
        path:         File path (used only for the chunk id / citations).
        window:       Lines per chunk. Defaults to 40 — a balance between
                      embedding quality and recall granularity.
        overlap:      Lines of overlap between adjacent chunks. Keeps
                      semantics from being cut in half at the boundary.
        start_offset: 1-indexed line number that the first line of
                      ``source`` corresponds to in the original file.
                      Used by :func:`chunk_python` when sub-chunking a
                      long def: the offset keeps citations accurate.

    Returns:
        A list of :class:`Chunk`. Empty input returns an empty list.

    Raises:
        ValueError: If ``window <= 0`` or ``overlap >= window``.
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if overlap < 0 or overlap >= window:
        raise ValueError(
            f"overlap must satisfy 0 <= overlap < window; got overlap={overlap}, window={window}"
        )

    if not source:
        return []

    lines = source.splitlines()
    total = len(lines)
    step = window - overlap

    chunks: List[Chunk] = []
    i = 0
    while i < total:
        end_idx = min(i + window, total)
        body = "\n".join(lines[i:end_idx]).rstrip()
        if body:
            chunks.append(
                _make_chunk(
                    path=path,
                    start_line=i + start_offset,
                    end_line=end_idx + start_offset - 1,
                    text=body,
                    header=None,
                )
            )
        if end_idx == total:
            break
        i += step

    return chunks


# ---------------------------------------------------------------------------
# Internal: stable chunk-id construction
# ---------------------------------------------------------------------------

def _make_chunk(
    *,
    path: str,
    start_line: int,
    end_line: int,
    text: str,
    header: Optional[str],
) -> Chunk:
    """Build a :class:`Chunk` with a content-stable id.

    The id formula — ``sha1(path + "|" + start_line + "|" + text)[:16]`` —
    is stable across runs and machines. The RAG manifest relies on that
    to decide "same chunk, skip re-embedding" vs "new chunk, embed me".
    """
    h = hashlib.sha1()
    h.update(path.encode("utf-8"))
    h.update(b"|")
    h.update(str(start_line).encode("ascii"))
    h.update(b"|")
    h.update(text.encode("utf-8"))
    chunk_id = h.hexdigest()[:16]

    return Chunk(
        id=chunk_id,
        path=path,
        start_line=start_line,
        end_line=end_line,
        text=text,
        header=header,
    )
