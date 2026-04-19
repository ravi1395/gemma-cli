"""``write_file`` — the only tool that creates or overwrites content.

Why only one write tool
-----------------------
We keep the write surface as narrow as possible. A single ``write_file``
tool covers "create new file" and "replace existing file"; anything
more complex (patch application, refactoring) is handled by the human
via ``gemma commit`` or their editor. Narrow surface = small attack
surface.

Safety invariants
-----------------
* Path must lie inside the workspace root and pass the denylist.
* Existing destinations are *archived* before overwrite — the prior
  content is preserved under ``<root>/archive/<ts>/...``. This is the
  never-delete rule applied to overwrites: no bytes are lost.
* The write is atomic: we write to a sibling ``.tmp`` and
  ``os.replace`` it. Interrupting mid-write leaves the original file
  intact (or, on first-time creates, simply absent).
"""

from __future__ import annotations

import os
from pathlib import Path

from gemma import safety as _safety
from gemma.tools import audit as _audit
from gemma.tools.capabilities import Capability
from gemma.tools.registry import ToolResult, ToolSpec, tool


# Cap the write size so a runaway tool call can't fill the disk.
_WRITE_MAX_BYTES = 1 * 1024 * 1024  # 1 MiB


@tool(ToolSpec(
    name="write_file",
    description=(
        "Create or replace a file with the given content. "
        "If the target exists, the prior version is archived. "
        "Requires --allow-writes and user confirmation."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Destination path (absolute or relative to the workspace root).",
            },
            "content": {
                "type": "string",
                "description": "Contents to write. UTF-8 only.",
            },
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    },
    capability=Capability.WRITE,
    requires_confirm=True,
))
def write_file(path: str, content: str) -> ToolResult:
    """Write ``content`` to ``path`` atomically, archiving any prior version."""
    target = Path(path)
    policy = _safety.default_policy(Path.cwd())
    try:
        resolved = _safety.ensure_allowed(target, policy)
    except _safety.SafetyError as exc:
        return ToolResult(ok=False, error="safety", content=str(exc))

    encoded = content.encode("utf-8")
    if len(encoded) > _WRITE_MAX_BYTES:
        return ToolResult(
            ok=False, error="too_large",
            content=(
                f"write rejected: {len(encoded)} bytes exceeds cap "
                f"{_WRITE_MAX_BYTES}"
            ),
        )

    archived_path: str = ""
    if resolved.exists():
        if resolved.is_dir():
            return ToolResult(
                ok=False, error="is_dir",
                content=f"{resolved} is a directory; refusing to overwrite",
            )
        try:
            dest = _safety.archive(resolved, policy)
            archived_path = str(dest)
        except _safety.SafetyError as exc:
            return ToolResult(ok=False, error="safety", content=str(exc))

    # Atomic write via tmp-file + os.replace.
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        tmp = resolved.with_suffix(resolved.suffix + ".gemma-tmp")
        tmp.write_bytes(encoded)
        os.replace(tmp, resolved)
    except OSError as exc:
        return ToolResult(ok=False, error="io", content=f"write failed: {exc}")

    meta = {
        "paths_touched": [
            {"path": str(resolved), "sha256": _audit.sha256_of(resolved)},
        ],
        "bytes_written": len(encoded),
    }
    if archived_path:
        meta["archived_prior_to"] = archived_path

    return ToolResult(
        ok=True,
        content=(
            f"wrote {len(encoded)} bytes to {resolved}"
            + (f" (prior version archived to {archived_path})" if archived_path else "")
        ),
        metadata=meta,
    )
