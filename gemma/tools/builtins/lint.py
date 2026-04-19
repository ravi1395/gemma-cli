"""``run_linter`` — invoke a language-appropriate linter on a path.

Today's language matrix is small on purpose:

============  =======================
``lang``      command
============  =======================
``python``    ``ruff check <path>``
``mypy``      ``mypy <path>``
``node``      ``npx -y eslint <path>``
============  =======================

We prefer :data:`Capability.READ` because linters do not mutate the
tree. The ``node`` entry *does* have a side effect — ``npx`` will
download the eslint package on first run — but that cost is borne by
the user's npm cache, not by the repo under review. Users who want to
block that should pass ``--no-network`` (which doesn't yet block
this path; tracked as a future tightening).

Executable resolution and output capping are handled by
:mod:`gemma.tools.subprocess_runner`.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from gemma import safety as _safety
from gemma.tools import subprocess_runner as _runner
from gemma.tools.capabilities import Capability
from gemma.tools.registry import ToolResult, ToolSpec, tool


# Map from ``lang`` to (argv template, default timeout).
#
# argv template uses the special token ``{path}`` which is replaced
# with the resolved absolute path. Tokens are never passed through
# a shell — ``subprocess_runner.run`` uses ``shell=False``.
_LINT_COMMANDS: dict[str, list[str]] = {
    "python": ["ruff", "check", "{path}"],
    "mypy":   ["mypy", "{path}"],
    "node":   ["npx", "-y", "eslint", "{path}"],
}


@tool(ToolSpec(
    name="run_linter",
    description=(
        "Run a language-appropriate linter and return its output. "
        "Supports: python (ruff), mypy, node (eslint)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "lang": {
                "type": "string",
                "enum": sorted(_LINT_COMMANDS.keys()),
                "description": "Linter to run.",
            },
            "path": {
                "type": "string",
                "description": "File or directory to lint (relative to workspace root).",
            },
        },
        "required": ["lang", "path"],
        "additionalProperties": False,
    },
    capability=Capability.READ,
    timeout_s=60,
    max_output_bytes=64 * 1024,
))
def run_linter(lang: str, path: str) -> ToolResult:
    """Resolve the path, look up the linter, invoke the subprocess runner."""
    # Schema already checked lang is in _LINT_COMMANDS; defensive lookup.
    template = _LINT_COMMANDS.get(lang)
    if template is None:
        return ToolResult(ok=False, error="unknown_lang", content=f"unsupported lang {lang!r}")

    target = Path(path)
    policy = _safety.default_policy(Path.cwd())
    try:
        resolved = _safety.ensure_allowed(target, policy)
    except _safety.SafetyError as exc:
        return ToolResult(ok=False, error="safety", content=str(exc))

    # Detect missing executables up front for a clearer error than the
    # subprocess runner's generic "FileNotFoundError".
    exe = template[0]
    if shutil.which(exe) is None:
        return ToolResult(
            ok=False, error="missing_tool",
            content=f"linter executable {exe!r} not found on PATH",
        )

    argv = [tok.replace("{path}", str(resolved)) for tok in template]
    result = _runner.run(
        argv,
        cwd=policy.root,
        timeout_s=60,
        max_output_bytes=64 * 1024,
    )

    # Linters conventionally exit non-zero when they find issues —
    # that's not a tool failure, it's the signal. Surface both
    # streams.
    body = result.stdout
    if result.stderr:
        body = (body + "\n--- stderr ---\n" + result.stderr).strip()
    if result.timed_out:
        body += "\n…[linter timed out]"

    return ToolResult(
        ok=True,
        content=body or "(no output)",
        metadata={
            "exit_code": result.exit_code,
            "duration_ms": result.duration_ms,
            "timed_out": result.timed_out,
            "truncated": result.truncated,
        },
    )
