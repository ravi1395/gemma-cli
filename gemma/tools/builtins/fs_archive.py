"""``archive_path`` — soft-delete via move to the archive directory.

This is the *only* sanctioned way for a tool to "remove" a file. The
never-delete rule says no tool is registered that unlinks content;
instead, ``archive_path`` relocates the target into
``<root>/archive/<ISO-ts>/<relpath>`` via
:func:`gemma.safety.archive`. The bytes are preserved, the original
location is freed.

For the model, this looks like delete: ``archive_path("old.py")``
returns the new location. For humans, a quick ``ls archive/`` recovers
the file if the model archived something important.
"""

from __future__ import annotations

from pathlib import Path

from gemma import safety as _safety
from gemma.tools import audit as _audit
from gemma.tools.capabilities import Capability
from gemma.tools.registry import ToolResult, ToolSpec, tool


@tool(ToolSpec(
    name="archive_path",
    description=(
        "Move a file or directory into the workspace archive folder. "
        "The content is preserved at a timestamped archive path. "
        "Requires --allow-writes and user confirmation. "
        "This is the only 'remove' affordance Gemma is given."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to archive (absolute or relative to the workspace root).",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    capability=Capability.ARCHIVE,
    requires_confirm=True,
))
def archive_path(path: str) -> ToolResult:
    """Move ``path`` into the workspace archive directory."""
    target = Path(path)
    policy = _safety.default_policy(Path.cwd())

    # ensure_allowed is called inside safety.archive too, but running
    # it here first lets us return a dedicated error code when the
    # path is simply missing vs. actually denylisted.
    try:
        resolved = _safety.ensure_allowed(target, policy)
    except _safety.SafetyError as exc:
        return ToolResult(ok=False, error="safety", content=str(exc))

    if not resolved.exists():
        return ToolResult(
            ok=False, error="not_found",
            content=f"no such path to archive: {resolved}",
        )

    # Hash *before* moving — after archive() the original location is
    # empty, and the new location is what we want to record.
    digest = _audit.sha256_of(resolved)

    try:
        dest = _safety.archive(resolved, policy)
    except _safety.SafetyError as exc:
        return ToolResult(ok=False, error="safety", content=str(exc))

    return ToolResult(
        ok=True,
        content=f"archived {resolved} -> {dest}",
        metadata={
            "paths_touched": [
                {"path": str(dest), "sha256": digest},
            ],
            "archived_from": str(resolved),
            "archived_to": str(dest),
        },
    )
