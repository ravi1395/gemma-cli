"""Tests for the ``http_get`` built-in.

No real network calls are made — :func:`urllib.request.urlopen` is
monkey-patched so the tests run deterministically in any environment.
The allowlist loader is exercised both at its default and with a
user-supplied TOML.
"""

from __future__ import annotations

import io
import urllib.error
from pathlib import Path

import pytest

from gemma.tools.builtins import net_fetch as _nf
from gemma.tools.registry import get as _get


def _http_get():
    _spec, handler = _get("http_get")
    return handler


class _FakeResp:
    """Minimal stand-in for the object returned by ``urlopen``."""

    def __init__(self, body: bytes, status: int = 200, content_type: str = "text/plain"):
        self._buf = io.BytesIO(body)
        self.status = status
        self.headers = {"Content-Type": content_type}

    def read(self, n=-1):
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_http_get_allowlisted_host_returns_body(monkeypatch):
    def fake_urlopen(req, timeout, context):
        # Sanity: we did construct an HTTPS Request with a User-Agent.
        assert req.full_url.startswith("https://")
        assert "gemma-cli" in req.headers.get("User-agent", "")
        return _FakeResp(b"hello world")

    monkeypatch.setattr(_nf.urllib.request, "urlopen", fake_urlopen)

    result = _http_get()(url="https://docs.python.org/3/library/index.html")
    assert result.ok is True
    assert "hello world" in result.content
    assert result.metadata["status"] == 200
    assert result.metadata["truncated"] is False


# ---------------------------------------------------------------------------
# Rejections (policy)
# ---------------------------------------------------------------------------

def test_http_get_rejects_non_https():
    result = _http_get()(url="http://docs.python.org/")
    assert result.ok is False
    assert result.error == "bad_scheme"


def test_http_get_rejects_host_not_in_allowlist():
    result = _http_get()(url="https://evil.example.com/path")
    assert result.ok is False
    assert result.error == "host_blocked"


def test_http_get_no_host_component():
    result = _http_get()(url="https:///no-host")
    assert result.ok is False
    assert result.error == "no_host"


# ---------------------------------------------------------------------------
# Failure modes of the network call
# ---------------------------------------------------------------------------

def test_http_get_surfaces_http_error(monkeypatch):
    def boom(req, timeout, context):
        raise urllib.error.HTTPError(
            req.full_url, 404, "Not Found", hdrs={}, fp=None,
        )

    monkeypatch.setattr(_nf.urllib.request, "urlopen", boom)

    result = _http_get()(url="https://api.github.com/missing")
    assert result.ok is False
    assert result.error == "http_error"
    assert result.metadata["status"] == 404


def test_http_get_surfaces_url_error(monkeypatch):
    def boom(req, timeout, context):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(_nf.urllib.request, "urlopen", boom)

    result = _http_get()(url="https://api.github.com/anywhere")
    assert result.ok is False
    assert result.error == "fetch_failed"


# ---------------------------------------------------------------------------
# Caps
# ---------------------------------------------------------------------------

def test_http_get_truncates_over_cap(monkeypatch):
    # 256 KiB + 100 bytes; tool caps at 256 KiB and appends a marker.
    big = b"A" * (256 * 1024 + 100)

    def fake_urlopen(req, timeout, context):
        return _FakeResp(big)

    monkeypatch.setattr(_nf.urllib.request, "urlopen", fake_urlopen)

    result = _http_get()(url="https://docs.python.org/huge")
    assert result.ok is True
    assert result.metadata["truncated"] is True
    assert "truncated" in result.content


# ---------------------------------------------------------------------------
# Allowlist loader
# ---------------------------------------------------------------------------

def test_load_allowlist_defaults_when_no_config(monkeypatch, tmp_path):
    # Point HOME at an empty tmp dir so no ~/.config/gemma file exists.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(_nf.Path, "home", classmethod(lambda cls: Path(str(tmp_path))))
    hosts = _nf._load_allowlist()
    # Defaults must be present.
    for h in _nf._DEFAULT_ALLOWLIST:
        assert h in hosts


def test_load_allowlist_extends_with_user_toml(monkeypatch, tmp_path):
    cfg_dir = tmp_path / ".config" / "gemma"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "tool-allowlist.toml").write_text(
        '[http_get]\nhosts = ["internal.example.com"]\n'
    )

    monkeypatch.setattr(_nf.Path, "home", classmethod(lambda cls: Path(str(tmp_path))))
    hosts = _nf._load_allowlist()
    # Defaults preserved, extras appended.
    assert "internal.example.com" in hosts
    for h in _nf._DEFAULT_ALLOWLIST:
        assert h in hosts


def test_load_allowlist_falls_back_on_parse_error(monkeypatch, tmp_path):
    """A broken TOML must not unlock extra hosts — defaults only."""
    cfg_dir = tmp_path / ".config" / "gemma"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "tool-allowlist.toml").write_text("this is [not valid")
    monkeypatch.setattr(_nf.Path, "home", classmethod(lambda cls: Path(str(tmp_path))))
    hosts = _nf._load_allowlist()
    assert list(hosts) == list(_nf._DEFAULT_ALLOWLIST)


# ---------------------------------------------------------------------------
# Host matcher
# ---------------------------------------------------------------------------

def test_host_allowed_is_case_insensitive():
    allow = ["docs.python.org"]
    assert _nf._host_allowed("DOCS.PYTHON.ORG", allow) is True
    assert _nf._host_allowed("docs.python.org", allow) is True


def test_host_allowed_rejects_suffix_matching():
    """No wildcards — ``sub.docs.python.org`` must NOT match ``docs.python.org``."""
    allow = ["docs.python.org"]
    assert _nf._host_allowed("sub.docs.python.org", allow) is False
    assert _nf._host_allowed("evil-docs.python.org", allow) is False
