"""Tests for gemma.safety — path guards and the never-delete archive helper.

These tests are the most important in the project: they are the last line
of defence against the nightmare scenario of gemma deleting something it
should not. Every rejection path must have a test; every archive path
must prove the source content survives.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from gemma.safety import (
    SafetyError,
    SafetyPolicy,
    archive,
    default_policy,
    ensure_allowed,
    ensure_inside,
    ensure_no_symlink_escape,
    is_denylisted,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def root(tmp_path: Path) -> Path:
    """Real directory to use as the policy root. Returns a resolved path."""
    return tmp_path.resolve()


@pytest.fixture
def policy(root: Path) -> SafetyPolicy:
    return default_policy(root)


# ---------------------------------------------------------------------------
# ensure_inside
# ---------------------------------------------------------------------------

def test_ensure_inside_accepts_direct_child(policy: SafetyPolicy, root: Path):
    target = root / "src" / "a.py"
    target.parent.mkdir(parents=True)
    target.write_text("x = 1")
    resolved = ensure_inside(target, policy)
    assert resolved == target.resolve()


def test_ensure_inside_rejects_absolute_outside(policy: SafetyPolicy, tmp_path):
    # A completely unrelated tmp path — guaranteed outside the policy root.
    outside = tmp_path.parent / "not-the-root-at-all"
    with pytest.raises(SafetyError, match="escapes declared root"):
        ensure_inside(outside, policy)


def test_ensure_inside_rejects_dot_dot_escape(policy: SafetyPolicy, root: Path):
    # ``<root>/../evil`` resolves out of the root.
    bad = root / ".." / "evil"
    with pytest.raises(SafetyError, match="escapes declared root"):
        ensure_inside(bad, policy)


def test_ensure_inside_rejects_sibling_with_shared_prefix(tmp_path: Path):
    """/tmp/foo should NOT be treated as inside /tmp/foobar."""
    root = (tmp_path / "foobar").resolve()
    root.mkdir()
    sibling = (tmp_path / "foo" / "x").resolve()
    sibling.parent.mkdir()
    sibling.write_text("x")
    pol = default_policy(root)
    with pytest.raises(SafetyError):
        ensure_inside(sibling, pol)


def test_ensure_inside_accepts_nonexistent_path_under_root(policy: SafetyPolicy, root: Path):
    """Writes to not-yet-existing paths must still be checked."""
    not_yet = root / "new-dir" / "new-file.txt"
    resolved = ensure_inside(not_yet, policy)
    assert resolved == not_yet.resolve()


# ---------------------------------------------------------------------------
# Symlink escape
# ---------------------------------------------------------------------------

@pytest.mark.skipif(os.name == "nt", reason="symlink creation often needs admin on Windows")
def test_symlink_whose_target_escapes_is_rejected(policy: SafetyPolicy, root: Path, tmp_path):
    outside = tmp_path.parent / "escape-target"
    outside.mkdir()

    link = root / "link-out"
    link.symlink_to(outside)
    # ensure_inside follows resolution (so it catches this), but we also
    # want the parent-walk check to fire.
    with pytest.raises(SafetyError):
        ensure_allowed(link / "file.txt", policy)


@pytest.mark.skipif(os.name == "nt", reason="symlink creation often needs admin on Windows")
def test_symlink_whose_target_stays_inside_is_ok(policy: SafetyPolicy, root: Path):
    inside = root / "real"
    inside.mkdir()
    (inside / "x.txt").write_text("hi")

    link = root / "alias"
    link.symlink_to(inside)

    ensure_no_symlink_escape(link / "x.txt", policy)  # must not raise
    resolved = ensure_allowed(link / "x.txt", policy)
    assert resolved == (inside / "x.txt").resolve()


# ---------------------------------------------------------------------------
# Denylist
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "denied",
    [
        ".git/config",
        ".env",
        ".env.production",
        ".ssh/id_rsa",
        ".aws/credentials",
        ".config/gemma/profiles/dev.toml",
        ".gemma/rag-manifest.json",
    ],
)
def test_denylisted_paths_rejected(policy: SafetyPolicy, root: Path, denied: str):
    target = root / denied
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("secret")
    assert is_denylisted(target, policy) is True
    with pytest.raises(SafetyError, match="denylist"):
        ensure_allowed(target, policy)


def test_nondenylisted_path_allowed(policy: SafetyPolicy, root: Path):
    ok = root / "src" / "handler.py"
    ok.parent.mkdir(parents=True)
    ok.write_text("def f(): pass")
    assert is_denylisted(ok, policy) is False
    ensure_allowed(ok, policy)  # must not raise


# ---------------------------------------------------------------------------
# archive() — the never-delete contract
# ---------------------------------------------------------------------------

def test_archive_moves_file_and_preserves_content(policy: SafetyPolicy, root: Path):
    src = root / "src" / "handler.py"
    src.parent.mkdir(parents=True)
    src.write_text("payload")
    original_bytes = src.read_bytes()

    dest = archive(src, policy)

    # Original location freed.
    assert not src.exists()
    # Content preserved at the archived location.
    assert dest.exists()
    assert dest.read_bytes() == original_bytes
    # Destination is under <root>/archive/<ts>/...
    assert (policy.root / "archive") in dest.parents


def test_archive_preserves_relative_path(policy: SafetyPolicy, root: Path):
    src = root / "deep" / "nested" / "a.py"
    src.parent.mkdir(parents=True)
    src.write_text("x")

    dest = archive(src, policy)

    rel = dest.relative_to(policy.root / "archive")
    # rel == <ts>/deep/nested/a.py
    assert rel.parts[1:] == ("deep", "nested", "a.py")


def test_archive_rejects_denylisted_source(policy: SafetyPolicy, root: Path):
    src = root / ".env"
    src.write_text("SECRET=1")
    with pytest.raises(SafetyError):
        archive(src, policy)
    # Denylisted file must still be present after a failed archive.
    assert src.exists()


def test_archive_rejects_path_outside_root(policy: SafetyPolicy, tmp_path):
    outsider = tmp_path.parent / "outside-file.txt"
    outsider.write_text("x")
    with pytest.raises(SafetyError):
        archive(outsider, policy)
    assert outsider.exists()


def test_archive_rejects_path_inside_archive(policy: SafetyPolicy, root: Path):
    """We must never archive-the-archiver; otherwise we'd loop forever."""
    # Seed an archive by archiving a throwaway file.
    probe = root / "probe.txt"
    probe.write_text("x")
    archived = archive(probe, policy)

    with pytest.raises(SafetyError, match="archive directory"):
        archive(archived, policy)


def test_archive_missing_source_raises_filenotfound(policy: SafetyPolicy, root: Path):
    ghost = root / "does-not-exist.txt"
    with pytest.raises(FileNotFoundError):
        archive(ghost, policy)


def test_archive_disambiguates_on_collision(policy: SafetyPolicy, root: Path, monkeypatch):
    """Two archive() calls on the same relpath in the same second must
    produce two distinct destination paths, and both sources must end up
    preserved on disk."""
    # Pin the timestamp so the two archives collide.
    from gemma import safety

    class _FixedDatetime:
        @staticmethod
        def now(tz=None):
            import datetime as _dt
            return _dt.datetime(2026, 4, 18, 12, 0, 0, tzinfo=_dt.timezone.utc)

    monkeypatch.setattr(safety, "datetime", _FixedDatetime)

    src1 = root / "a.py"
    src1.write_text("first")
    dest1 = archive(src1, policy)

    src2 = root / "a.py"
    src2.write_text("second")
    dest2 = archive(src2, policy)

    assert dest1 != dest2
    assert dest1.read_text() == "first"
    assert dest2.read_text() == "second"
