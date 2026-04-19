"""User-facing ``gemma tools {list,run,audit}`` commands.

Provides a human affordance over the same dispatcher the model uses.
Running a tool from the CLI is a first-class feature because:

* It's the easiest way for a user to verify the tool matrix on their
  machine (e.g. "does ``run_linter`` actually find ruff?").
* It doubles as a test harness — new tools get exercised end-to-end
  before the model is trusted with them.
* It supports shell piping: ``gemma tools run read_file --arg
  path=foo.py | head``.
"""

from __future__ import annotations

import json
import shlex
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

from gemma import platform as _platform
from gemma.tools import audit as _audit
from gemma.tools import registry as _registry
from gemma.tools.capabilities import Capability, GatingContext, gate
from gemma.tools.dispatcher import Dispatcher
from gemma.tools.registry import ToolResult, ToolSpec


console = Console()
err_console = Console(stderr=True)


# ---------------------------------------------------------------------------
# gemma tools list
# ---------------------------------------------------------------------------

def list_command(
    allow_writes: bool = typer.Option(
        False, "--allow-writes",
        help="Show write-capability tools as mounted (otherwise they appear as gated).",
    ),
    allow_network: bool = typer.Option(
        True, "--allow-network/--no-network",
        help="Toggle network tool visibility.",
    ),
    json_output: bool = typer.Option(
        False, "--json",
        help="Emit a JSON array instead of the table.",
    ),
) -> None:
    """List every registered tool and whether it's mounted under the given flags."""
    ctx = GatingContext(
        allow_writes=allow_writes,
        allow_network=allow_network,
        is_tty=_platform.is_tty(),
    )

    rows = []
    for spec in _registry.all_specs():
        decision = gate(spec.capability, ctx)
        rows.append({
            "name": spec.name,
            "capability": spec.capability.value,
            "mounted": decision.allowed,
            "requires_confirm": decision.requires_confirm,
            "description": spec.description,
            "reason": decision.reason,
        })

    if json_output:
        print(json.dumps(rows, indent=2))
        return

    table = Table(title="gemma tools")
    table.add_column("Name", style="cyan")
    table.add_column("Cap", style="magenta")
    table.add_column("Mounted", justify="center")
    table.add_column("Description")
    for r in rows:
        mark = "[green]✓[/green]" if r["mounted"] else "[red]·[/red]"
        if r["mounted"] and r["requires_confirm"]:
            mark += " [yellow](confirms)[/yellow]"
        table.add_row(r["name"], r["capability"], mark, r["description"])
    console.print(table)


# ---------------------------------------------------------------------------
# gemma tools run
# ---------------------------------------------------------------------------

def run_command(
    name: str = typer.Argument(..., help="Tool name (see `gemma tools list`)."),
    arg: List[str] = typer.Option(
        [], "--arg",
        help="Argument as key=value. May be repeated. Use key:= to pass JSON (e.g. 'flags:=[1,2]').",
    ),
    allow_writes: bool = typer.Option(
        False, "--allow-writes",
        help="Required to invoke WRITE or ARCHIVE tools.",
    ),
    allow_network: bool = typer.Option(
        True, "--allow-network/--no-network",
        help="Toggle network tool visibility.",
    ),
    auto_approve: bool = typer.Option(
        False, "--auto-approve-writes",
        help="Skip the y/N prompt in non-TTY pipelines.",
    ),
    json_output: bool = typer.Option(
        False, "--json",
        help="Emit the full ToolResult as JSON instead of just content.",
    ),
) -> None:
    """Invoke a tool directly from the shell.

    Argument syntax: ``--arg key=value`` for strings, ``--arg key:=<json>``
    for non-string values (int, bool, list, etc.). Examples::

        gemma tools run read_file --arg path=README.md
        gemma tools run run_tests --arg include_logs:=true
        gemma tools run list_dir --arg path=gemma --arg glob='*.py'
    """
    try:
        kwargs = _parse_args(arg)
    except ValueError as exc:
        err_console.print(f"[red]bad --arg syntax: {exc}[/red]")
        raise typer.Exit(code=2)

    ctx = GatingContext(
        allow_writes=allow_writes,
        allow_network=allow_network,
        is_tty=_platform.is_tty(),
        auto_approve_writes=auto_approve,
    )
    dispatcher = Dispatcher(
        ctx=ctx,
        session_id="cli",
        confirm=_interactive_confirm,
    )

    result = dispatcher.dispatch(name, kwargs)
    _print_result(result, json_output=json_output)
    if not result.ok:
        raise typer.Exit(code=1)


def _parse_args(items: List[str]) -> dict:
    """Turn a list of ``key=value`` / ``key:=json`` strings into a dict.

    The distinction matters because the dispatcher validates against a
    JSON schema that cares about types: ``include_logs=true`` would
    parse as the string ``"true"``, but ``include_logs:=true`` parses
    as the boolean ``True``.
    """
    out = {}
    for raw in items:
        # JSON form first — it contains ``=`` but also ``:=``.
        if ":=" in raw:
            key, _, val = raw.partition(":=")
            try:
                out[key] = json.loads(val)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{raw!r}: invalid JSON ({exc})")
        elif "=" in raw:
            key, _, val = raw.partition("=")
            out[key] = val
        else:
            raise ValueError(f"{raw!r}: expected 'key=value' or 'key:=<json>'")
    return out


def _interactive_confirm(spec: ToolSpec, kwargs: dict) -> bool:
    """Rich y/N prompt used for WRITE/ARCHIVE tool confirmation."""
    pretty = shlex.join([f"{k}={v!r}" for k, v in kwargs.items()])
    err_console.print(
        f"[yellow]gemma tools: {spec.name} ({spec.capability.value}) "
        f"with {pretty}[/yellow]"
    )
    # Rich's Confirm uses stdin/stdout; in non-TTY mode it will fall
    # back to the default (False). That matches our policy: no prompt,
    # no approval.
    try:
        return Confirm.ask("Proceed?", default=False)
    except (EOFError, KeyboardInterrupt):
        return False


def _print_result(result: ToolResult, *, json_output: bool) -> None:
    """Render a :class:`ToolResult` for the user."""
    if json_output:
        payload = {
            "ok": result.ok,
            "error": result.error,
            "content": result.content,
            "metadata": result.metadata,
        }
        print(json.dumps(payload, indent=2, default=str))
        return

    if result.ok:
        # Plain print so stdout is pipe-safe.
        print(result.content)
    else:
        err_console.print(f"[red]tool refused: {result.error}[/red]")
        err_console.print(result.content)


# ---------------------------------------------------------------------------
# gemma tools audit
# ---------------------------------------------------------------------------

def audit_command(
    since: str = typer.Option(
        "1d", "--since",
        help="Keep only records newer than this duration (e.g. '1d', '6h', '30m').",
    ),
    limit: int = typer.Option(50, "--limit", "-n", help="Max records to show."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Tail the tool-audit log."""
    try:
        cutoff_delta = _parse_duration(since)
    except ValueError as exc:
        err_console.print(f"[red]bad --since: {exc}[/red]")
        raise typer.Exit(code=2)

    since_iso = (datetime.now(timezone.utc) - cutoff_delta).strftime("%Y-%m-%dT%H:%M:%SZ")
    records = _audit.tail(n=limit, since_iso=since_iso)

    if json_output:
        print(json.dumps(records, indent=2))
        return

    if not records:
        console.print("[dim]no audit records[/dim]")
        return

    table = Table(title=f"tool audit ({len(records)} records)")
    table.add_column("Time", style="cyan")
    table.add_column("Tool")
    table.add_column("Cap", style="magenta")
    table.add_column("By")
    table.add_column("Exit", justify="right")
    table.add_column("Dur ms", justify="right")
    table.add_column("Note")
    for rec in records:
        note = rec.get("refusal_reason") or ""
        table.add_row(
            rec.get("ts", "?"),
            rec.get("tool", "?"),
            rec.get("capability", "?"),
            rec.get("approved_by", "?"),
            str(rec.get("exit_code", "?")),
            str(rec.get("duration_ms", "?")),
            note[:60],
        )
    console.print(table)


def _parse_duration(spec: str) -> timedelta:
    """Parse ``5m`` / ``2h`` / ``1d`` style durations into :class:`timedelta`.

    Raises ValueError on anything else. No ``30s`` support because the
    audit log's timestamps are second-precision — sub-minute filtering
    would be misleading.
    """
    if not spec:
        raise ValueError("empty duration")
    unit = spec[-1].lower()
    try:
        n = int(spec[:-1])
    except ValueError:
        raise ValueError(f"{spec!r}: expected <int><unit>")

    if unit == "m":
        return timedelta(minutes=n)
    if unit == "h":
        return timedelta(hours=n)
    if unit == "d":
        return timedelta(days=n)
    raise ValueError(f"unknown unit {unit!r}; use m|h|d")
