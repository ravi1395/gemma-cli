"""Benchmark HTML → markdown extraction for ``http_get``.

Backs improvement #18 ("HTML→markdown extraction in `http_get`").
The promise there is a 5–15× reduction in tokens per fetched page
with ≤ 20 ms extraction cost on a laptop. This module captures the
extraction-cost half of that claim; the token-reduction half is a
feature metric, not a timing metric, and lives in the improvement
doc itself.

We run against **three fixture-shaped HTML documents** built in
Python — no network I/O — so the benchmark measures parser cost
only:

* ``_blog_post``      — Readability-friendly article markup.
* ``_docs_page``      — long navigation + a content column, which
                        is where naïve HTML-to-text falls over.
* ``_marketing_page`` — short body with many inline scripts and
                        style blocks, the pathological case.

If ``trafilatura`` isn't installed (it sits in an optional extra
the user hasn't installed yet), the module **skips** rather than
fails — this keeps the default ``pytest tests/bench`` run green on
boxes that haven't opted into agentic-mode deps.
"""

from __future__ import annotations

import pytest

trafilatura = pytest.importorskip(
    "trafilatura",
    reason="trafilatura is an optional dep; install with pip install '.[agent]'",
)


# ---------------------------------------------------------------------------
# Fixture HTML bodies
# ---------------------------------------------------------------------------

def _blog_post() -> str:
    body = "<p>" + ("Paragraph prose about algorithms. " * 40) + "</p>"
    return f"""
    <html><head><title>An article about recursion</title></head>
    <body>
      <nav>Home · About · Archive</nav>
      <article>
        <h1>An article about recursion</h1>
        <p>Recursion is the act of defining something in terms of itself.</p>
        {body * 5}
        <h2>Worked examples</h2>
        {body * 3}
      </article>
      <footer>© 2026</footer>
    </body></html>
    """


def _docs_page() -> str:
    sidebar = "".join(f"<li><a href='#s{i}'>Section {i}</a></li>" for i in range(80))
    body = "".join(
        f"<h2 id='s{i}'>Section {i}</h2><p>Canonical explanation of topic {i}. "
        f"{'Detail ' * 30}</p>"
        for i in range(20)
    )
    return f"""
    <html><head><title>The docs</title></head>
    <body>
      <aside><ul>{sidebar}</ul></aside>
      <main>{body}</main>
    </body></html>
    """


def _marketing_page() -> str:
    noise = "<script>var x = {};</script>" * 30 + "<style>.a{color:red}</style>" * 20
    return f"""
    <html><head><title>Buy the thing</title>{noise}</head>
    <body>
      {noise}
      <h1>Buy the thing</h1>
      <p>It is the best thing. Three short bullets follow.</p>
      <ul><li>Fast.</li><li>Cheap.</li><li>Good.</li></ul>
      {noise}
    </body></html>
    """


PAGES = {
    "blog": _blog_post(),
    "docs": _docs_page(),
    "marketing": _marketing_page(),
}


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", list(PAGES.keys()))
def test_html_to_md(benchmark, kind):
    """Extract the main body of a typical HTML page to markdown.

    The assertion is intentionally loose — we care about timing,
    not parser quality here. A correctness test for extraction
    quality belongs in ``tests/`` proper, not in the bench suite.
    """
    html = PAGES[kind]

    def _run():
        out = trafilatura.extract(html, include_comments=False, output_format="markdown")
        assert out and len(out) > 0, f"empty extraction for {kind}"

    benchmark(_run)
