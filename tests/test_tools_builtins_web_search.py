"""Unit tests for the ``web_search`` builtin (item #16).

These tests never hit a real search engine — they inject stub backends
that deterministically return a fixed list of hits (or raise). That
keeps the suite hermetic, fast, and safe for CI.

What we cover:
* Happy path — backend returns hits, tool wraps them as JSON.
* ``max_results`` clamp on both ends (1 floor, cap ceiling).
* Empty/whitespace query → ``invalid_args`` error result.
* Unknown backend name → ``unknown_backend`` error result.
* Backend raises ``RuntimeError`` → surfaced as ``backend_error``.
* Tool is registered under ``NETWORK`` capability and advertised by
  :func:`gemma.tools.registry.mount` with the default context.
"""

from __future__ import annotations

import json
from typing import List

import pytest

from gemma.config import Config
from gemma.tools import registry as _registry
from gemma.tools.backends.base import SearchBackend, SearchHit
from gemma.tools.builtins.web_search import (
    _MAX_RESULTS_CAP,
    _get_backend,
    web_search,
)
from gemma.tools.capabilities import Capability, GatingContext


# ---------------------------------------------------------------------------
# Stub backends
# ---------------------------------------------------------------------------

class _FixedBackend(SearchBackend):
    """Returns a pre-canned list of hits, regardless of ``query``."""

    name = "_fixed"

    def __init__(self, hits: List[SearchHit]) -> None:
        self._hits = hits

    def search(self, query: str, *, max_results: int = 5, timeout_s: float = 10.0):
        # Honour the cap so tests can assert the clamp propagated.
        return self._hits[:max_results]


class _RaisingBackend(SearchBackend):
    """Always raises — models the 'ddgs not installed / rate-limited' path."""

    name = "_raising"

    def search(self, query: str, *, max_results: int = 5, timeout_s: float = 10.0):
        raise RuntimeError("boom: no internet")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg_with_backend(name: str) -> Config:
    cfg = Config()
    cfg.web_search_backend = name
    return cfg


def _patch_backend_resolver(monkeypatch, fake_backend: SearchBackend) -> None:
    """Swap ``_get_backend`` so the tool sees ``fake_backend`` instead of
    constructing the real one. Scoped via monkeypatch so the real
    resolver is restored after each test."""
    import gemma.tools.builtins.web_search as ws_mod
    monkeypatch.setattr(ws_mod, "_get_backend", lambda _name: fake_backend)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_web_search_returns_json_hits(monkeypatch):
    """Backend returns 2 hits → content is a JSON array of 2 objects."""
    hits = [
        SearchHit(title="Py asyncio", url="https://example.com/a", snippet="async guide"),
        SearchHit(title="Threading",  url="https://example.com/b", snippet="threading guide"),
    ]
    _patch_backend_resolver(monkeypatch, _FixedBackend(hits))

    result = web_search(
        query="asyncio vs threading", max_results=5,
        _cfg=_cfg_with_backend("_fixed"),
    )

    assert result.ok is True
    payload = json.loads(result.content)
    assert payload == [
        {"title": "Py asyncio", "url": "https://example.com/a", "snippet": "async guide"},
        {"title": "Threading",  "url": "https://example.com/b", "snippet": "threading guide"},
    ]
    assert result.metadata == {
        "backend": "_fixed",
        "result_count": 2,
        "query": "asyncio vs threading",
    }


def test_web_search_trims_query_whitespace(monkeypatch):
    """Leading/trailing spaces are stripped before passing to the backend."""
    hits: List[SearchHit] = []
    captured: dict = {}

    class _Capture(SearchBackend):
        name = "_capture"
        def search(self, query: str, *, max_results=5, timeout_s=10.0):
            captured["query"] = query
            return hits

    _patch_backend_resolver(monkeypatch, _Capture())
    res = web_search(query="  hello world  ", _cfg=_cfg_with_backend("_capture"))
    assert res.ok is True
    assert captured["query"] == "hello world"


# ---------------------------------------------------------------------------
# Clamping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "requested, expected_cap",
    [
        (0, 1),                    # floor at 1 — never a zero-request
        (1, 1),                    # identity
        (5, 5),                    # default
        (_MAX_RESULTS_CAP, _MAX_RESULTS_CAP),
        (999, _MAX_RESULTS_CAP),   # cap at top
        (-4, 1),                   # negatives also clamp to 1
    ],
)
def test_web_search_clamps_max_results(monkeypatch, requested, expected_cap):
    """``max_results`` is forced into [1, _MAX_RESULTS_CAP]."""
    observed: dict = {}

    class _Recording(SearchBackend):
        name = "_recording"
        def search(self, query: str, *, max_results=5, timeout_s=10.0):
            observed["max_results"] = max_results
            return []

    _patch_backend_resolver(monkeypatch, _Recording())
    res = web_search(query="x", max_results=requested, _cfg=_cfg_with_backend("_recording"))
    assert res.ok is True
    assert observed["max_results"] == expected_cap


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_web_search_rejects_empty_query():
    """Empty / whitespace-only queries are refused with ``invalid_args``."""
    for bad in ("", "   ", "\t\n"):
        result = web_search(query=bad)
        assert result.ok is False
        assert result.error == "invalid_args"


def test_web_search_unknown_backend_surface():
    """An unknown ``web_search_backend`` value produces a structured error."""
    result = web_search(query="hi", _cfg=_cfg_with_backend("totally_fake_backend"))
    assert result.ok is False
    assert result.error == "unknown_backend"
    assert "totally_fake_backend" in result.content


def test_web_search_backend_raises_becomes_tool_error(monkeypatch):
    """RuntimeError from the backend is converted to ok=False/backend_error."""
    _patch_backend_resolver(monkeypatch, _RaisingBackend())

    result = web_search(query="hi", _cfg=_cfg_with_backend("_raising"))
    assert result.ok is False
    assert result.error == "backend_error"
    assert "boom" in result.content


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------

def test_web_search_is_registered_as_network_tool():
    """``@tool`` decorator populated the registry with capability=NETWORK."""
    # The builtins module was already imported via the test file's own
    # import chain; no need to import again here.
    spec, _handler = _registry.get("web_search")
    assert spec.capability is Capability.NETWORK
    assert "query" in spec.parameters["properties"]
    assert spec.parameters["required"] == ["query"]


def test_web_search_mounted_when_network_allowed():
    """Under the default GatingContext (allow_network=True), web_search
    shows up in :func:`mount`'s advertised list."""
    ctx = GatingContext(allow_network=True, allow_writes=False)
    names = {spec.name for spec in _registry.mount(ctx)}
    assert "web_search" in names


def test_web_search_hidden_when_network_disabled():
    """With ``allow_network=False``, mount filters out NETWORK tools."""
    ctx = GatingContext(allow_network=False, allow_writes=False)
    names = {spec.name for spec in _registry.mount(ctx)}
    assert "web_search" not in names


# ---------------------------------------------------------------------------
# _get_backend resolver
# ---------------------------------------------------------------------------

def test_get_backend_returns_duckduckgo_instance_when_ddgs_installed():
    """The default backend resolves to DuckDuckGoBackend. This does NOT
    hit the network — construction is cheap."""
    backend = _get_backend("duckduckgo")
    assert backend.name == "duckduckgo"
    # Case-insensitive match — profiles often use any casing.
    assert _get_backend("DuckDuckGo").name == "duckduckgo"


def test_get_backend_tavily_and_brave_stubs_raise_on_search():
    """Stub backends instantiate fine but refuse to run a search."""
    for name in ("tavily", "brave"):
        backend = _get_backend(name)
        with pytest.raises(RuntimeError, match="not implemented"):
            backend.search("x")


def test_get_backend_unknown_raises_valueerror():
    """Surfaces a clear error rather than an ImportError or KeyError."""
    with pytest.raises(ValueError, match="unknown web_search backend"):
        _get_backend("google")
