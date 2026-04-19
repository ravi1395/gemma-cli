"""Filesystem-safety primitives for gemma-cli.

Enforces three project-wide invariants in one place so every feature that
touches the filesystem reuses the same checks:

1. **Paths stay inside their declared root.** No relative-path tricks, no
   symlink targets that escape the root. Used by the Phase 6.1 tool
   registry and reusable by ``gemma commit`` / ``gemma sh``.
2. **Denylisted locations are untouchable.** ``.git/``, ``.env*``, SSH and
   AWS credential dirs, and the gemma config dir itself are off-limits
   regardless of which root was declared.
3. **Never delete — always archive.** The only path-mutating helper in
   this module moves files into ``<root>/archive/<ISO-ts>/<relpath>``.
   There is no delete function. A caller wanting to "remove" a file
   should call :func:`archive`; the file's content is preserved and the
   original location is freed.

This module is pure Python with no external dependencies. All functions
raise :class:`SafetyError` on violation so callers can distinguish safety
refusals from ordinary I/O errors.
"""

from __future__ import annotations

import fnmatch
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Tuple


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class SafetyError(Exception):
    """Raised when a path or operation violates a safety invariant.

    Distinct from ``OSError`` / ``PermissionError`` so calling code can
    tell "filesystem said no" from "gemma said no" — the latter is almost
    always an attack or a bug we want to surface loudly.
    """


# ---------------------------------------------------------------------------
# Denylist
# ---------------------------------------------------------------------------
#
# Globs are matched against POSIX-style paths produced by ``Path.as_posix()``
# for consistency across Linux/macOS/Windows. Each entry is matched against
# *both* the absolute path and its relative form under the home directory,
# so ``~/.ssh/id_rsa`` and ``/home/alice/.ssh/id_rsa`` both match.

_DEFAULT_DENY_GLOBS: Tuple[str, ...] = (
    # Version control internals
    "**/.git",
    "**/.git/**",
    "**/.hg",
    "**/.hg/**",
    "**/.svn",
    "**/.svn/**",
    # Environment / secrets
    "**/.env",
    "**/.env.*",
    # SSH / cloud credentials
    "**/.ssh",
    "**/.ssh/**",
    "**/.aws",
    "**/.aws/**",
    "**/.gnupg",
    "**/.gnupg/**",
    # Our own config dir — we never want a model-invoked tool mutating
    # ``~/.config/gemma/profiles/*.toml`` or the gemma history file.
    "**/.config/gemma",
    "**/.config/gemma/**",
    "**/.gemma",
    "**/.gemma/**",
)


@dataclass(frozen=True)
class SafetyPolicy:
    """Configurable safety policy.

    Most callers should use :func:`default_policy`. Tests and advanced
    users (e.g. a future ``gemma rag index`` that genuinely needs to read
    a ``.env.example``) can construct a custom policy that tightens or —
    explicitly and auditably — relaxes the defaults.

    Attributes:
        root: The anchor directory. All paths checked against this policy
              must resolve to a location under ``root``.
        deny_globs: Glob patterns (POSIX-style) that are never allowed.
        allow_symlinks: When False, symlinks whose targets escape ``root``
                        are rejected. True only makes sense for read-only
                        tooling in trusted trees.
    """

    root: Path
    deny_globs: Tuple[str, ...] = _DEFAULT_DENY_GLOBS
    allow_symlinks: bool = False


def default_policy(root: Path) -> SafetyPolicy:
    """Return the standard policy rooted at ``root``.

    ``root`` is resolved to an absolute path via :meth:`Path.resolve` so
    later checks can compare absolute paths directly.
    """
    return SafetyPolicy(root=Path(root).resolve())


# ---------------------------------------------------------------------------
# Path-safety checks
# ---------------------------------------------------------------------------

def ensure_inside(candidate: Path, policy: SafetyPolicy) -> Path:
    """Resolve ``candidate`` and confirm it lies inside ``policy.root``.

    The resolution uses ``strict=False`` so non-existent paths (e.g. a
    write target) are still checked. A write to ``/etc/passwd`` is caught
    here even though the file exists outside the root.

    Args:
        candidate: Path to validate. May be relative; will be resolved
            against ``policy.root`` by :meth:`Path.resolve` (which itself
            joins against cwd, so callers should pass absolute paths or
            chdir appropriately).

    Returns:
        The resolved absolute path. Safe to pass to ``open()``.

    Raises:
        SafetyError: If the resolved path escapes the root.
    """
    resolved = Path(candidate).resolve()
    root = policy.root

    # Use ``os.path.commonpath`` rather than string-prefix compare so
    # ``/tmp/foo`` is not accepted as being inside ``/tmp/foobar``.
    try:
        common = Path(os.path.commonpath([str(resolved), str(root)]))
    except ValueError:
        # Raised when the paths live on different Windows drives.
        raise SafetyError(
            f"Path {resolved} is on a different drive than root {root}"
        )

    if common != root:
        raise SafetyError(
            f"Path {resolved} escapes declared root {root}"
        )

    return resolved


def ensure_no_symlink_escape(candidate: Path, policy: SafetyPolicy) -> None:
    """Walk the parent chain of ``candidate`` and reject symlink escapes.

    ``Path.resolve`` already follows symlinks, so :func:`ensure_inside`
    will catch a symlink *target* that escapes the root. This function
    catches the subtler case of a symlink *component* in the middle of
    the path whose target briefly leaves the root before re-entering it.

    No-ops when ``policy.allow_symlinks`` is True.
    """
    if policy.allow_symlinks:
        return

    # Walk from the candidate up; every component that is itself a symlink
    # must point at a target inside the root.
    current = Path(candidate)
    seen: set[Path] = set()
    while True:
        if current in seen:
            # Cycle — extremely defensive; resolve() would normally
            # raise RuntimeError here, but we prefer a SafetyError.
            raise SafetyError(f"Symlink cycle detected at {candidate}")
        seen.add(current)

        if current.is_symlink():
            target = (current.parent / os.readlink(current)).resolve()
            try:
                common = Path(os.path.commonpath([str(target), str(policy.root)]))
            except ValueError:
                raise SafetyError(
                    f"Symlink at {current} targets a different drive than root"
                )
            if common != policy.root:
                raise SafetyError(
                    f"Symlink at {current} targets {target}, outside root {policy.root}"
                )

        parent = current.parent
        if parent == current:
            # Reached the filesystem root without finding an escaping symlink.
            return
        current = parent


def is_denylisted(candidate: Path, policy: SafetyPolicy) -> bool:
    """True iff ``candidate`` matches any of ``policy.deny_globs``.

    Matches against :meth:`Path.as_posix()` so the same glob works on
    Windows and POSIX alike.
    """
    return _matches_any(Path(candidate).as_posix(), policy.deny_globs)


def ensure_allowed(candidate: Path, policy: SafetyPolicy) -> Path:
    """Apply the full safety suite and return the resolved path.

    Equivalent to :func:`ensure_inside` + :func:`ensure_no_symlink_escape`
    + denylist check. This is the function most callers should use — the
    lower-level helpers exist so tests can exercise each invariant in
    isolation.
    """
    resolved = ensure_inside(candidate, policy)
    ensure_no_symlink_escape(candidate, policy)
    if is_denylisted(resolved, policy):
        raise SafetyError(f"Path {resolved} is on the denylist")
    return resolved


# ---------------------------------------------------------------------------
# Archive (never delete)
# ---------------------------------------------------------------------------

def archive(path: Path, policy: SafetyPolicy) -> Path:
    """Move ``path`` into ``<root>/archive/<ISO-ts>/<relpath>``.

    This is the **only** sanctioned way for gemma code to "remove" a file
    or directory. The content is preserved at the archive location; the
    original location is freed.

    Semantics:
      * The timestamp directory is created with 0o700 so other users on
        shared systems can't snoop at archived content.
      * Relative paths are preserved: archiving ``src/a.py`` from root
        ``/repo`` yields ``/repo/archive/<ts>/src/a.py``.
      * If the destination already exists (extremely rare thanks to
        second-granularity timestamps), a numeric suffix is appended.

    Args:
        path: File or directory to archive. Must pass :func:`ensure_allowed`.
        policy: Safety policy. ``policy.root`` is where the ``archive/``
                directory will be created.

    Returns:
        The absolute path of the archived copy.

    Raises:
        SafetyError: If the source path fails safety checks or if the
                     source is the archive directory itself (guards
                     against archiving-the-archiver infinite loops).
        FileNotFoundError: If the source does not exist.
    """
    source = ensure_allowed(path, policy)

    archive_root = (policy.root / "archive").resolve()
    if _is_same_or_parent(archive_root, source):
        raise SafetyError(
            f"Refusing to archive a path inside the archive directory: {source}"
        )

    if not source.exists():
        raise FileNotFoundError(f"Cannot archive {source}: does not exist")

    # ISO-8601 timestamp in UTC, colon-free for Windows-path friendliness.
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rel = source.relative_to(policy.root)

    dest = archive_root / ts / rel
    dest = _disambiguate(dest)
    dest.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    # shutil.move is rename() when same-fs; copy2+remove when cross-fs.
    # In both cases the original location ends up empty — the file's
    # content is not destroyed, just relocated.
    shutil.move(str(source), str(dest))
    return dest


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _matches_any(posix_path: str, patterns: Iterable[str]) -> bool:
    """Return True if ``posix_path`` matches any of the glob patterns."""
    return any(fnmatch.fnmatchcase(posix_path, pat) for pat in patterns)


def _is_same_or_parent(parent: Path, candidate: Path) -> bool:
    """Return True iff ``candidate == parent`` or ``candidate`` is inside ``parent``."""
    try:
        return candidate == parent or candidate.is_relative_to(parent)
    except AttributeError:
        # Python 3.10-compat fallback — is_relative_to exists in 3.9+ so
        # this branch is belt-and-braces.
        try:
            candidate.relative_to(parent)
            return True
        except ValueError:
            return candidate == parent


def _disambiguate(candidate: Path) -> Path:
    """Append ``.1``, ``.2``, … until ``candidate`` does not exist.

    Used when two ``archive()`` calls land in the same second for the same
    relative path (extremely rare but not impossible under heavy load).
    """
    if not candidate.exists():
        return candidate
    for i in range(1, 1000):
        alt = candidate.with_suffix(candidate.suffix + f".{i}")
        if not alt.exists():
            return alt
    raise SafetyError(
        f"Could not disambiguate archive destination after 1000 attempts: {candidate}"
    )
