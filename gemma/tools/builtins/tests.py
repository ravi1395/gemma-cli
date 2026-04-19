"""``run_tests`` — invoke a language-appropriate test runner.

The v1 matrix is intentionally small; extending it is a matter of
adding a row to :data:`_TEST_COMMANDS`.

Design notes
------------
* :data:`Capability.READ`. The tool reads the project tree and executes
  test code, but does not write artefacts the user asked us to. Test
  runners do create their own caches (``.pytest_cache``, ``node_modules``)
  — those are outside our scope and live outside the tool contract.
* ``include_logs=False`` by default. Test output is token-expensive; the
  first call returns just the runner's summary. A follow-up call with
  ``include_logs=True`` returns the full output, subject to the
  subprocess-runner cap.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Optional

from gemma import safety as _safety
from gemma.tools import subprocess_runner as _runner
from gemma.tools.capabilities import Capability
from gemma.tools.registry import ToolResult, ToolSpec, tool


# Order matters for the "which runner is this repo?" heuristic: we
# check for marker files top-down and pick the first that matches.
_RUNNERS = [
    ("pytest", ["pyproject.toml", "pytest.ini", "tox.ini"], ["pytest", "-q"]),
    ("npm",    ["package.json"],                             ["npm", "test", "--", "--silent"]),
    ("go",     ["go.mod"],                                   ["go", "test", "./..."]),
]

# When ``target`` is provided we append it to the argv. For pytest
# this is a node spec (``tests/test_foo.py::test_bar``), for npm it's
# ignored by the heuristic runner (passed through anyway for
# transparency), for go it's a package path.


_TEST_COMMANDS: dict[str, list[str]] = {name: argv for name, _m, argv in _RUNNERS}


def _detect_runner(root: Path) -> Optional[str]:
    """Return the runner name for ``root``, or None if unknown.

    Picks the first marker hit in :data:`_RUNNERS` order so Python
    projects that also happen to have a ``package.json`` (rare but
    real) don't get miscategorised as Node.
    """
    for name, markers, _ in _RUNNERS:
        if any((root / m).exists() for m in markers):
            return name
    return None


@tool(ToolSpec(
    name="run_tests",
    description=(
        "Run the project's test suite and return a summary. "
        "The tool auto-detects pytest / npm test / go test based on "
        "marker files in the workspace root. Set include_logs=true to "
        "return the full log instead of the summary."
    ),
    parameters={
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "Optional test selector (e.g. 'tests/test_foo.py::test_bar').",
            },
            "include_logs": {
                "type": "boolean",
                "description": "Return the full captured log instead of the summary.",
            },
        },
        "additionalProperties": False,
    },
    capability=Capability.READ,
    timeout_s=300,
    max_output_bytes=256 * 1024,
))
def run_tests(target: Optional[str] = None, include_logs: bool = False) -> ToolResult:
    """Detect the runner, invoke it, and return a summary or full log."""
    root = Path.cwd()
    runner = _detect_runner(root)
    if runner is None:
        return ToolResult(
            ok=False, error="no_runner",
            content=(
                "could not detect a test runner: no pyproject.toml, "
                "package.json, or go.mod at workspace root"
            ),
        )

    argv = list(_TEST_COMMANDS[runner])
    if target:
        # Run through safety to ensure target is inside the root even
        # though the runner would likely reject a path outside anyway.
        # Only validate if the target looks like a path (has a slash).
        if "/" in target or "\\" in target:
            policy = _safety.default_policy(root)
            try:
                _safety.ensure_allowed(Path(target), policy)
            except _safety.SafetyError as exc:
                return ToolResult(ok=False, error="safety", content=str(exc))
        argv.append(target)

    if shutil.which(argv[0]) is None:
        return ToolResult(
            ok=False, error="missing_tool",
            content=f"test runner {argv[0]!r} not found on PATH",
        )

    result = _runner.run(
        argv,
        cwd=root,
        timeout_s=300,
        max_output_bytes=256 * 1024,
    )

    full_log = (result.stdout or "") + (("\n--- stderr ---\n" + result.stderr) if result.stderr else "")
    summary = _extract_summary(runner, full_log) if not include_logs else full_log

    if result.timed_out:
        summary += "\n…[test run timed out]"

    return ToolResult(
        ok=result.exit_code == 0,
        content=summary or "(no output)",
        error=None if result.exit_code == 0 else "tests_failed",
        metadata={
            "runner": runner,
            "exit_code": result.exit_code,
            "duration_ms": result.duration_ms,
            "timed_out": result.timed_out,
            "truncated": result.truncated,
        },
    )


# ---------------------------------------------------------------------------
# Summary extraction
# ---------------------------------------------------------------------------
#
# Each test runner prints a terminal summary line; we grep for it so
# the model sees "3 passed, 2 failed" rather than 200 KB of logs by
# default. Missing summary → we return the last ~20 lines.

_SUMMARY_PATTERNS: dict[str, re.Pattern] = {
    "pytest": re.compile(r"=+ .* (passed|failed|error)s? .*=+"),
    "npm":    re.compile(r"(Tests:|tests? passed|tests? failed)"),
    "go":     re.compile(r"(^ok\s|FAIL\s|PASS$|FAIL$)", re.MULTILINE),
}


def _extract_summary(runner: str, log: str) -> str:
    """Pull the summary section out of ``log`` if we recognise it.

    Falls back to the last 20 lines so we always return *something*
    useful even if the runner's output format drifts.
    """
    pat = _SUMMARY_PATTERNS.get(runner)
    if pat is not None:
        matches = pat.findall(log)
        if matches:
            # Use the last match — it's typically the summary line that
            # appears at the very end of a run.
            tail = "\n".join(log.splitlines()[-10:])
            return tail
    tail = "\n".join(log.splitlines()[-20:])
    return tail
