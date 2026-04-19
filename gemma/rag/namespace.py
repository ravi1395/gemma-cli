"""Redis-key namespace resolution for RAG.

Two invariants drive this module:

1. **Different workspaces must not share keys.** Two repos cloned on
   the same machine — e.g. ``~/src/foo`` and ``~/src/bar`` — should be
   indexable independently. We derive a short, deterministic prefix
   from the absolute workspace path via SHA-1.
2. **Different branches of the same repo get different namespaces.**
   Retrieval on ``feature/auth-v2`` should not surface chunks from
   ``main``. When git metadata is absent (shallow clone, non-repo
   directory) we fall back to a single ``_default`` branch name.

The namespace is the string interpolated into every Redis key:
``gemma:rag:{namespace}:<kind>:<id>``. Keep this module small and
dependency-light so importing :mod:`gemma.rag` stays cheap.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Optional


#: Length of the workspace-hash prefix. 12 hex chars = 48 bits of entropy,
#: enough to avoid collisions across any reasonable number of local repos
#: without making audit output unreadable.
_ROOT_HASH_CHARS = 12

#: Name used when no git branch can be detected.
_FALLBACK_BRANCH = "_default"


def resolve_namespace(root: Path, branch: Optional[str] = None) -> str:
    """Return the Redis-key namespace for a workspace.

    Format: ``{root_hash}:{branch}``.

    Args:
        root: Absolute or relative path to the workspace root. Resolved
              to an absolute path before hashing so ``./foo`` and
              ``/full/path/to/foo`` share a namespace.
        branch: Explicit branch override. When ``None`` we probe
                ``git rev-parse --abbrev-ref HEAD``; a non-repo or a
                detached HEAD falls back to ``_default``.

    Returns:
        A string of the form ``"9f3b8c1a2d4e:main"``. The colon is
        intentional — Redis admin tools render segmented keys nicely.
    """
    resolved = Path(root).resolve()
    root_hash = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:_ROOT_HASH_CHARS]
    effective_branch = branch if branch is not None else _detect_branch(resolved) or _FALLBACK_BRANCH
    return f"{root_hash}:{_sanitize_branch(effective_branch)}"


def _detect_branch(root: Path) -> Optional[str]:
    """Return the current git branch name, or None if unavailable.

    We shell out to git rather than parse ``.git/HEAD`` manually so that
    worktrees and symref edge cases are handled by git itself. Any
    failure (git missing, not a repo, detached HEAD) returns None and
    the caller substitutes ``_FALLBACK_BRANCH``.
    """
    try:
        res = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if res.returncode != 0:
        return None

    name = (res.stdout or "").strip()
    if not name or name == "HEAD":
        # Detached HEAD or unborn repo — no meaningful branch name.
        return None
    return name


def _sanitize_branch(name: str) -> str:
    """Make a branch name Redis-key-safe.

    Redis keys are binary-safe but conventionally avoid whitespace and
    colons so admin tools (``redis-cli KEYS ...``) remain usable.
    We replace every non-``[A-Za-z0-9._-]`` character with ``_``.
    """
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in name)
