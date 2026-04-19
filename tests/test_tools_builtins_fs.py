"""Tests for the filesystem built-in tools.

Covers the three READ tools (``read_file``, ``list_dir``, ``stat``), the
single WRITE tool (``write_file``) and the ARCHIVE tool (``archive_path``).

Each handler resolves paths against ``Path.cwd()``, so every test
chdirs into a ``tmp_path`` via ``monkeypatch.chdir`` to keep the
workspace root isolated from the repository's real tree. Import side
effects populate the registry, so we call the handler functions
directly via ``gemma.tools.registry.get``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from gemma.tools.registry import get as _get


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Chdir into an empty tmp workspace so safety resolves against it."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _handler(name: str):
    """Fetch a registered handler by name (raises if missing)."""
    _spec, handler = _get(name)
    return handler


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

def test_read_file_happy_path(workspace):
    (workspace / "hello.txt").write_text("hi there")
    result = _handler("read_file")(path="hello.txt")
    assert result.ok is True
    assert "hi there" in result.content
    # sha256 metadata shape.
    touched = result.metadata["paths_touched"][0]
    assert touched["path"].endswith("hello.txt")
    assert len(touched["sha256"]) == 16


def test_read_file_missing_returns_not_found(workspace):
    result = _handler("read_file")(path="does_not_exist.txt")
    assert result.ok is False
    assert result.error == "not_found"


def test_read_file_rejects_escape(workspace):
    """A ../ traversal to /etc must be refused by the safety layer."""
    result = _handler("read_file")(path="../../../../etc/passwd")
    assert result.ok is False
    assert result.error == "safety"


def test_read_file_refuses_directory(workspace):
    (workspace / "sub").mkdir()
    result = _handler("read_file")(path="sub")
    assert result.ok is False
    assert result.error == "not_file"


def test_read_file_truncates_beyond_cap(workspace):
    """Files over the 64 KiB cap must come back with a truncation marker."""
    big = workspace / "big.txt"
    big.write_bytes(b"A" * (70 * 1024))
    result = _handler("read_file")(path="big.txt")
    assert result.ok is True
    assert result.metadata["truncated"] is True
    assert "truncated" in result.content


# ---------------------------------------------------------------------------
# list_dir
# ---------------------------------------------------------------------------

def test_list_dir_happy_path(workspace):
    (workspace / "a.py").write_text("")
    (workspace / "b.txt").write_text("")
    (workspace / "sub").mkdir()
    result = _handler("list_dir")(path=".")
    assert result.ok is True
    # Directory entries have a trailing slash.
    assert "sub/" in result.content
    assert "a.py" in result.content
    assert "b.txt" in result.content


def test_list_dir_glob_filters(workspace):
    (workspace / "a.py").write_text("")
    (workspace / "b.txt").write_text("")
    result = _handler("list_dir")(path=".", glob="*.py")
    assert result.ok is True
    assert "a.py" in result.content
    assert "b.txt" not in result.content


def test_list_dir_rejects_non_directory(workspace):
    (workspace / "file").write_text("x")
    result = _handler("list_dir")(path="file")
    assert result.ok is False
    assert result.error == "not_dir"


def test_list_dir_rejects_escape(workspace):
    result = _handler("list_dir")(path="../../..")
    assert result.ok is False
    assert result.error == "safety"


# ---------------------------------------------------------------------------
# stat
# ---------------------------------------------------------------------------

def test_stat_happy_path(workspace):
    p = workspace / "file.txt"
    p.write_text("abc")
    result = _handler("stat")(path="file.txt")
    assert result.ok is True
    assert result.metadata["is_file"] is True
    assert result.metadata["is_dir"] is False
    assert result.metadata["size"] == 3


def test_stat_missing_returns_not_found(workspace):
    result = _handler("stat")(path="ghost")
    assert result.ok is False
    assert result.error == "not_found"


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------

def test_write_file_creates_new_file(workspace):
    result = _handler("write_file")(path="new.txt", content="hello")
    assert result.ok is True
    assert (workspace / "new.txt").read_text() == "hello"
    # Fresh creates don't trigger archiving.
    assert "archived_prior_to" not in result.metadata


def test_write_file_archives_prior_version(workspace):
    target = workspace / "doc.md"
    target.write_text("v1")
    result = _handler("write_file")(path="doc.md", content="v2")
    assert result.ok is True
    assert target.read_text() == "v2"
    # Prior bytes survive in the archive — never-delete invariant.
    archived = Path(result.metadata["archived_prior_to"])
    assert archived.exists()
    assert archived.read_text() == "v1"


def test_write_file_refuses_directory_target(workspace):
    (workspace / "sub").mkdir()
    result = _handler("write_file")(path="sub", content="no")
    assert result.ok is False
    assert result.error == "is_dir"


def test_write_file_rejects_escape(workspace):
    result = _handler("write_file")(path="../escape.txt", content="nope")
    assert result.ok is False
    assert result.error == "safety"


def test_write_file_rejects_over_cap(workspace):
    """Write > 1 MiB is capped at the tool layer, not the disk."""
    # 1 MiB + 1 byte of UTF-8-safe ASCII
    payload = "x" * (1 * 1024 * 1024 + 1)
    result = _handler("write_file")(path="big.txt", content=payload)
    assert result.ok is False
    assert result.error == "too_large"
    assert not (workspace / "big.txt").exists()


def test_write_file_no_tmp_left_behind_on_success(workspace):
    """The `.gemma-tmp` sibling must be renamed, not left around."""
    result = _handler("write_file")(path="f.txt", content="x")
    assert result.ok is True
    leftovers = list(workspace.glob("*.gemma-tmp"))
    assert leftovers == []


# ---------------------------------------------------------------------------
# archive_path
# ---------------------------------------------------------------------------

def test_archive_path_moves_file_into_archive(workspace):
    target = workspace / "old.py"
    target.write_text("legacy")
    result = _handler("archive_path")(path="old.py")
    assert result.ok is True
    # Original is gone from its old location…
    assert not target.exists()
    # …but the bytes live on at the archived location.
    dest = Path(result.metadata["archived_to"])
    assert dest.exists()
    assert dest.read_text() == "legacy"


def test_archive_path_rejects_missing(workspace):
    result = _handler("archive_path")(path="ghost.py")
    assert result.ok is False
    assert result.error == "not_found"


def test_archive_path_rejects_escape(workspace):
    result = _handler("archive_path")(path="../outside.txt")
    assert result.ok is False
    assert result.error == "safety"


# ---------------------------------------------------------------------------
# Fuzz: path safety across every READ/WRITE/ARCHIVE tool
# ---------------------------------------------------------------------------

# Paths a model could plausibly forge to exfiltrate or tamper outside
# the workspace. Every entry must be refused by the safety layer
# regardless of which tool receives it.
_ESCAPING_PATHS = [
    "../../etc/passwd",
    "/etc/passwd",
    "/root/.ssh/id_rsa",
    "../../../home",
]


@pytest.mark.parametrize("bad", _ESCAPING_PATHS)
@pytest.mark.parametrize("tool_name", ["read_file", "list_dir", "stat", "archive_path"])
def test_fuzz_path_args_rejected_everywhere(workspace, bad, tool_name):
    """Every tool that takes a path must refuse escapes uniformly."""
    result = _handler(tool_name)(path=bad)
    assert result.ok is False
    # "safety" for unambiguous denials; "not_found" is also acceptable
    # for paths that happen to be inside root but absent.
    assert result.error in {"safety", "not_found"}


def test_fuzz_write_file_rejects_escape_paths(workspace):
    """write_file takes two args; parametrise separately to keep signatures clean."""
    for bad in _ESCAPING_PATHS:
        result = _handler("write_file")(path=bad, content="x")
        assert result.ok is False
        assert result.error == "safety"
