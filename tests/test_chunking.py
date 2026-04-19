"""Tests for gemma.chunking — AST / markdown / sliding-window strategies.

These tests anchor the chunk-ID stability contract, which is what makes
incremental re-embedding in Phase 6.2 correct. Any change to the id
formula is a manifest-schema break and must bump ``chunk_rev``.
"""

from __future__ import annotations

import textwrap

import pytest

from gemma.chunking import (
    Chunk,
    chunk_for_path,
    chunk_markdown,
    chunk_python,
    chunk_sliding,
)


# ---------------------------------------------------------------------------
# chunk_sliding
# ---------------------------------------------------------------------------

def test_sliding_empty_input_returns_empty():
    assert chunk_sliding("", "a.txt") == []


def test_sliding_single_window_for_short_input():
    src = "line 1\nline 2\nline 3"
    chunks = chunk_sliding(src, "a.txt", window=40, overlap=5)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.start_line == 1
    assert c.end_line == 3
    assert c.text == src


def test_sliding_produces_overlap():
    src = "\n".join(f"L{i}" for i in range(1, 101))  # 100 lines
    chunks = chunk_sliding(src, "a.txt", window=40, overlap=5)
    # window=40, step=35 → starts at 1, 36, 71 (and maybe 106 which would be
    # clipped). So we expect exactly 3 chunks covering 1-40, 36-75, 71-100.
    assert [c.start_line for c in chunks] == [1, 36, 71]
    assert [c.end_line for c in chunks] == [40, 75, 100]
    # Overlap between chunks 0 and 1: lines 36-40 appear in both.
    assert "L36" in chunks[0].text and "L36" in chunks[1].text


def test_sliding_rejects_bad_window():
    with pytest.raises(ValueError):
        chunk_sliding("x", "a.txt", window=0)


def test_sliding_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_sliding("x", "a.txt", window=10, overlap=10)


def test_sliding_respects_start_offset():
    src = "a\nb\nc"
    chunks = chunk_sliding(src, "long.py", window=40, overlap=0, start_offset=100)
    assert chunks[0].start_line == 100
    assert chunks[0].end_line == 102


# ---------------------------------------------------------------------------
# chunk_markdown
# ---------------------------------------------------------------------------

def test_markdown_splits_on_headings():
    src = textwrap.dedent(
        """\
        # Intro
        hello

        ## Details
        more

        ## Wrap-up
        end
        """
    )
    chunks = chunk_markdown(src, "README.md")
    headers = [c.header for c in chunks]
    assert headers == ["Intro", "Details", "Wrap-up"]
    # Each chunk includes its heading line.
    assert chunks[0].text.startswith("# Intro")
    assert chunks[1].text.startswith("## Details")


def test_markdown_prologue_captured_when_content_above_first_heading():
    src = "some intro prose\n# First heading\nbody\n"
    chunks = chunk_markdown(src, "doc.md")
    assert chunks[0].header == "<prologue>"
    assert chunks[0].text == "some intro prose"


def test_markdown_with_no_headings_falls_back_to_sliding():
    src = "\n".join(f"line {i}" for i in range(1, 11))
    chunks = chunk_markdown(src, "doc.md")
    assert len(chunks) == 1
    assert chunks[0].header is None   # sliding window has no header


def test_markdown_empty_input_returns_empty():
    assert chunk_markdown("", "doc.md") == []


# ---------------------------------------------------------------------------
# chunk_python
# ---------------------------------------------------------------------------

def test_python_splits_on_top_level_defs():
    src = textwrap.dedent(
        """\
        import os

        CONST = 1

        def foo():
            return 1

        class Bar:
            def method(self):
                return 2

        async def baz():
            pass
        """
    )
    chunks = chunk_python(src, "mod.py")
    headers = [c.header for c in chunks]
    assert headers[0] == "<module-head>"
    assert "def foo" in headers
    assert "class Bar" in headers
    assert "async def baz" in headers


def test_python_pure_script_falls_back_to_sliding():
    src = "print('hi')\nprint('there')\n"
    chunks = chunk_python(src, "script.py")
    assert len(chunks) == 1
    assert chunks[0].header is None


def test_python_syntax_error_falls_back_to_sliding():
    src = "def broken(:\n  pass\n"
    chunks = chunk_python(src, "broken.py")
    # No exception; we still got some coverage.
    assert len(chunks) == 1


def test_python_long_def_is_further_split():
    # Build a function whose body is >200 lines so the safety-split kicks in.
    body = "\n".join(f"    x = {i}" for i in range(1, 250))
    src = "def huge():\n" + body + "\n"
    chunks = chunk_python(src, "big.py")
    # The safety-split should produce more than one chunk for this def,
    # and each sub-chunk should preserve the function's header.
    assert len(chunks) > 1
    assert all(c.header == "def huge" for c in chunks)


def test_python_empty_input_returns_empty():
    assert chunk_python("", "mod.py") == []


# ---------------------------------------------------------------------------
# Dispatch + chunk-id stability
# ---------------------------------------------------------------------------

def test_dispatch_python(tmp_path):
    src = "def f(): pass\n"
    chunks = chunk_for_path(src, "mod.py")
    assert chunks[0].header == "def f"


def test_dispatch_markdown():
    chunks = chunk_for_path("# H\nbody\n", "README.md")
    assert chunks[0].header == "H"


def test_dispatch_unknown_extension_uses_sliding():
    chunks = chunk_for_path("a\nb\nc\n", "log.txt")
    assert len(chunks) == 1
    assert chunks[0].header is None


def test_chunk_id_stable_across_calls():
    src = "# H\nbody\n"
    a = chunk_markdown(src, "x.md")
    b = chunk_markdown(src, "x.md")
    assert [c.id for c in a] == [c.id for c in b]


def test_chunk_id_changes_when_content_changes():
    a = chunk_markdown("# H\nbody\n", "x.md")
    b = chunk_markdown("# H\nbody2\n", "x.md")
    assert a[0].id != b[0].id


def test_chunk_id_changes_when_path_changes():
    a = chunk_markdown("# H\nbody\n", "a.md")
    b = chunk_markdown("# H\nbody\n", "b.md")
    assert a[0].id != b[0].id


def test_chunk_line_range_rendering():
    c = Chunk(id="x", path="a", start_line=5, end_line=5, text="")
    assert c.line_range == "5"
    c2 = Chunk(id="x", path="a", start_line=5, end_line=12, text="")
    assert c2.line_range == "5-12"
