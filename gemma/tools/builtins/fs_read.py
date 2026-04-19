"""Filesystem *read* tools: ``read_file``, ``list_dir``, ``stat``.

Three pure-Python, side-effect-free tools. Each runs every path
argument through :mod:`gemma.safety` so a model that forges an
absolute path or a ``../../etc/passwd`` traversal is refused before
any I/O happens. All three are in the :data:`Capability.READ` tier
and therefore always mounted — there is no separate flag to turn
file reads off.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Optional

from gemma import safety as _safety
from gemma.tools import audit as _audit
from gemma.tools.capabilities import Capability
from gemma.tools.registry import ToolResult, ToolSpec, tool


# Sensible caps to keep a single tool call from flooding the context.
_READ_MAX_BYTES = 64 * 1024
_LIST_MAX_ENTRIES = 500


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

@tool(ToolSpec(
    name="read_file",
    description=(
        "Return up to the first 64 KB of a file, as text. "
        "Paths are resolved against the workspace root; traversals are refused."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file, absolute or relative to the workspace root.",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    capability=Capability.READ,
))
def read_file(path: str) -> ToolResult:
    """Read a file's content, capped at :data:`_READ_MAX_BYTES`.

    We deliberately keep the signature plain — the dispatcher already
    validated the schema, so we can assume ``path`` is a string.
    """
    target = Path(path)
    policy = _safety.default_policy(Path.cwd())
    try:
        resolved = _safety.ensure_allowed(target, policy)
    except _safety.SafetyError as exc:
        return ToolResult(ok=False, error="safety", content=str(exc))

    if not resolved.exists():
        return ToolResult(ok=False, error="not_found", content=f"no such file: {resolved}")
    if not resolved.is_file():
        return ToolResult(ok=False, error="not_file", content=f"{resolved} is not a regular file")

    try:
        data = resolved.read_bytes()
    except OSError as exc:
        return ToolResult(ok=False, error="io", content=f"read failed: {exc}")

    truncated = len(data) > _READ_MAX_BYTES
    body = data[:_READ_MAX_BYTES].decode("utf-8", errors="replace")
    if truncated:
        body += f"\n…[truncated: {len(data) - _READ_MAX_BYTES} bytes not shown]"

    return ToolResult(
        ok=True,
        content=body,
        metadata={
            "paths_touched": [
                {"path": str(resolved), "sha256": _audit.sha256_of(resolved)},
            ],
            "truncated": truncated,
        },
    )


# ---------------------------------------------------------------------------
# list_dir
# ---------------------------------------------------------------------------

@tool(ToolSpec(
    name="list_dir",
    description=(
        "List entries in a directory, optionally filtered by a glob pattern. "
        "Returns a newline-delimited listing."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path, absolute or relative to the workspace root.",
            },
            "glob": {
                "type": "string",
                "description": "Optional fnmatch-style pattern (e.g. '*.py').",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    capability=Capability.READ,
))
def list_dir(path: str, glob: Optional[str] = None) -> ToolResult:
    """List a directory's entries, at most :data:`_LIST_MAX_ENTRIES`."""
    target = Path(path)
    policy = _safety.default_policy(Path.cwd())
    try:
        resolved = _safety.ensure_allowed(target, policy)
    except _safety.SafetyError as exc:
        return ToolResult(ok=False, error="safety", content=str(exc))

    if not resolved.exists():
        return ToolResult(ok=False, error="not_found", content=f"no such dir: {resolved}")
    if not resolved.is_dir():
        return ToolResult(ok=False, error="not_dir", content=f"{resolved} is not a directory")

    entries: list[str] = []
    try:
        for child in sorted(resolved.iterdir()):
            name = child.name
            if glob and not fnmatch.fnmatchcase(name, glob):
                continue
            suffix = "/" if child.is_dir() else ""
            entries.append(name + suffix)
            if len(entries) >= _LIST_MAX_ENTRIES:
                entries.append(f"…[truncated at {_LIST_MAX_ENTRIES} entries]")
                break
    except OSError as exc:
        return ToolResult(ok=False, error="io", content=f"listdir failed: {exc}")

    return ToolResult(
        ok=True,
        content="\n".join(entries) if entries else "(empty)",
        metadata={
            "paths_touched": [
                {"path": str(resolved), "sha256": "dir"},
            ],
            "count": len(entries),
        },
    )


# ---------------------------------------------------------------------------
# stat
# ---------------------------------------------------------------------------

@tool(ToolSpec(
    name="stat",
    description=(
        "Report basic file metadata (size, mtime, is_dir, is_symlink) "
        "for a path."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to stat, absolute or relative to the workspace root.",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    capability=Capability.READ,
))
def stat(path: str) -> ToolResult:
    """Return basic filesystem metadata for a path."""
    target = Path(path)
    policy = _safety.default_policy(Path.cwd())
    try:
        resolved = _safety.ensure_allowed(target, policy)
    except _safety.SafetyError as exc:
        return ToolResult(ok=False, error="safety", content=str(exc))

    try:
        # ``lstat`` so a symlink-to-elsewhere is still describable.
        st = resolved.lstat()
    except FileNotFoundError:
        return ToolResult(ok=False, error="not_found", content=f"no such path: {resolved}")
    except OSError as exc:
        return ToolResult(ok=False, error="io", content=f"stat failed: {exc}")

    info = {
        "path": str(resolved),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "is_dir": resolved.is_dir(),
        "is_file": resolved.is_file(),
        "is_symlink": resolved.is_symlink(),
    }
    # Compact printable form for the model; machine callers can parse
    # it by splitting on ``: ``.
    lines = [f"{k}: {v}" for k, v in info.items()]
    return ToolResult(ok=True, content="\n".join(lines), metadata=info)
