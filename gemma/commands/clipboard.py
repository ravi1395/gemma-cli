"""``gemma clipboard`` subcommands and the shared ``--copy`` output helper.

Two kinds of surface live here:

1. :func:`status_command` — the user-facing ``gemma clipboard status``
   subcommand. Prints the selected backend, the probe log, and the host
   OS/SSH status so users can debug "why didn't my ``--copy`` work?"
   without reading source.

2. :func:`handle_copy_flags` — a tiny helper every output-producing
   command uses. Centralising the ``--copy`` / ``--copy-tee`` logic
   means we write the flag semantics, the success/failure printing, and
   the redaction-finding warning exactly once.
"""

from __future__ import annotations

import json
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from gemma import clipboard as _clipboard


console = Console()
err_console = Console(stderr=True)


# ---------------------------------------------------------------------------
# gemma clipboard status
# ---------------------------------------------------------------------------

def status_command(
    json_output: bool = typer.Option(
        False, "--json",
        help="Emit a JSON snapshot instead of a human-readable table.",
    ),
) -> None:
    """Show which clipboard backend gemma will use, and why.

    Useful for diagnosing "--copy did nothing" on Linux (wrong selection)
    or WSL (Wayland tools absent). The probe log lists every backend we
    considered and the reason each was accepted or skipped.
    """
    snapshot = _clipboard.describe()

    if json_output:
        print(json.dumps(snapshot, indent=2))
        return

    console.print(f"[bold]OS:[/bold]       {snapshot['os']}")
    console.print(f"[bold]SSH:[/bold]      {snapshot['ssh']}")
    selected = snapshot["selected"] or "[red]none[/red]"
    console.print(f"[bold]Selected:[/bold] {selected}")

    table = Table(title="Backend probe log")
    table.add_column("Backend", style="cyan")
    table.add_column("OK", justify="center")
    table.add_column("Reason")
    for entry in snapshot["probe_log"]:
        ok = "[green]✓[/green]" if entry["ok"] else "[dim]·[/dim]"
        table.add_row(entry["backend"], ok, entry["reason"])
    console.print(table)

    if snapshot["selected"] is None:
        console.print(
            "\n[yellow]No backend selected. Install one of the tools "
            "above (e.g. `sudo apt install xclip` on X11 Linux, "
            "`brew install pbcopy-alternatives` on macOS — though "
            "pbcopy ships by default) or `pip install pyperclip`.[/yellow]"
        )


# ---------------------------------------------------------------------------
# Shared --copy / --copy-tee helper
# ---------------------------------------------------------------------------

def handle_copy_flags(
    text: str,
    *,
    copy: bool,
    copy_tee: bool,
    allow_secrets: bool,
    tool_name: str,
) -> None:
    """Route ``text`` through the clipboard layer according to CLI flags.

    Parameters
    ----------
    text:
        The output the command produced. The caller is responsible for
        having already printed to stdout when appropriate; this helper
        only concerns itself with the clipboard side-effect and the
        accompanying stderr notification.
    copy:
        Value of the command's ``--copy`` flag. When True and
        ``copy_tee`` is False, the text has already been printed (or
        suppressed) by the caller — this helper only copies.
    copy_tee:
        Value of the command's ``--copy-tee`` flag. Semantically a
        superset of ``--copy``: the caller printed to stdout AND we
        should copy. We treat ``--copy-tee`` as implying ``--copy``.
    allow_secrets:
        Value of the ``--allow-secrets`` override. Forwarded to
        :func:`gemma.clipboard.copy`.
    tool_name:
        Short label used in the stderr notification ("sh", "commit", …).
        Keeps the messaging consistent across commands.

    The helper prints a one-line stderr notification on both success and
    failure. Output goes to stderr so stdout piping is unaffected.
    """
    if not (copy or copy_tee):
        return

    result = _clipboard.copy(text, allow_secrets=allow_secrets)

    if result.ok:
        findings_note = ""
        if result.redaction_findings:
            # We redacted something but the user asked for allow_secrets;
            # still report the count so they know the clipboard holds
            # raw material.
            kinds = sorted({f.type for f in result.redaction_findings})
            findings_note = f" (secrets kept: {', '.join(kinds)})"
        err_console.print(
            f"[green]✓ gemma {tool_name}: copied via {result.backend}{findings_note}[/green]"
        )
        return

    err_console.print(f"[yellow]gemma {tool_name}: clipboard skipped — {result.reason}[/yellow]")
    if result.redaction_findings and not allow_secrets:
        kinds = sorted({f.type for f in result.redaction_findings})
        err_console.print(
            f"[yellow]  • matched secret types: {', '.join(kinds)}[/yellow]"
        )


# ---------------------------------------------------------------------------
# Reusable flag declarations
# ---------------------------------------------------------------------------
#
# Rather than re-declare the same three typer.Options on every command,
# callers can import these module-level factories. This keeps flag
# help-text, defaults, and naming consistent across ``sh``, ``commit``,
# ``diff`` (etc.).

def copy_option() -> bool:
    """Factory for the ``--copy`` flag. Import and call inline."""
    return typer.Option(
        False, "--copy",
        help="Copy the command's output to the system clipboard.",
    )


def copy_tee_option() -> bool:
    """Factory for the ``--copy-tee`` flag — prints AND copies."""
    return typer.Option(
        False, "--copy-tee",
        help="Print the output to stdout AND copy to the clipboard.",
    )


def allow_secrets_option() -> bool:
    """Factory for the ``--allow-secrets`` override."""
    return typer.Option(
        False, "--allow-secrets",
        help="Allow --copy/--copy-tee even if secrets were detected in the output.",
    )
