"""User-facing ``gemma completion {install,print,status,uninstall}``.

This module is a thin Typer facade over :mod:`gemma.completion` so the
mechanics (script generation, fenced-block editing, atomic writes) can
be unit-tested in isolation and the CLI surface stays trivial.

Error strategy: every subcommand exits non-zero with a user-readable
message on failure and uses a ``stderr`` Console to avoid polluting
stdout (so ``gemma completion print | tee ~/.zshrc`` still works).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from gemma import completion as _completion
from gemma import platform as _platform


# Two consoles: stdout for data the user might pipe, stderr for
# messaging. Matching the pattern used by commands/clipboard.py.
console = Console()
err_console = Console(stderr=True)


# ---------------------------------------------------------------------------
# Shell resolution
# ---------------------------------------------------------------------------

def _resolve_shell(requested: str) -> _platform.Shell:
    """Turn the ``--shell`` CLI argument into a :class:`Shell`.

    ``auto`` defers to :func:`gemma.platform.detect_shell`. Anything
    else must match a :class:`Shell` member exactly; invalid input
    raises ``typer.BadParameter`` with a helpful hint.
    """
    if requested == "auto":
        detected = _platform.detect_shell()
        if detected is _platform.Shell.UNKNOWN:
            raise typer.BadParameter(
                "Could not auto-detect your shell. "
                "Pass --shell bash|zsh|fish|powershell explicitly."
            )
        return detected

    try:
        return _platform.Shell(requested)
    except ValueError:
        raise typer.BadParameter(
            f"Unknown shell {requested!r}. "
            "Valid options: auto, bash, zsh, fish, powershell."
        )


# ---------------------------------------------------------------------------
# gemma completion install
# ---------------------------------------------------------------------------

def install_command(
    shell: str = typer.Option(
        "auto", "--shell",
        help="Target shell (auto|bash|zsh|fish|powershell). Default auto-detects from $SHELL.",
    ),
    force: bool = typer.Option(
        False, "--force",
        help="Install even if an existing block is present (replace it).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Print the file changes that would be made and exit without writing.",
    ),
) -> None:
    """Install gemma tab completion into your shell's rc file.

    The completion block is delimited by a sentinel fence so it can be
    cleanly replaced on re-install or removed by ``gemma completion
    uninstall``. The prior rc file content is archived (never deleted)
    before any rewrite.
    """
    resolved = _resolve_shell(shell)

    try:
        plan = _completion.install(resolved, dry_run=dry_run, force=force)
    except ValueError as exc:
        err_console.print(f"[red]completion install: {exc}[/red]")
        raise typer.Exit(code=1)

    if plan.action == "noop":
        err_console.print(
            f"[green]completion: already up to date at {plan.rc_path}[/green]"
        )
        return

    if dry_run:
        err_console.print(
            f"[yellow]dry-run: would {plan.action} {plan.rc_path}[/yellow]"
        )
        # Emit the *new* content on stdout so a user can diff it.
        console.print(plan.new_content, end="", markup=False, highlight=False)
        return

    err_console.print(
        f"[green]✓ completion installed ({plan.action}) at {plan.rc_path}[/green]"
    )
    _hint_next_steps(resolved, plan.rc_path)


# ---------------------------------------------------------------------------
# gemma completion print
# ---------------------------------------------------------------------------

def print_command(
    shell: str = typer.Option(
        "auto", "--shell",
        help="Target shell (auto|bash|zsh|fish|powershell).",
    ),
) -> None:
    """Print the completion script to stdout without writing any files.

    For users who manage rc files via chezmoi / stow / Nix home-manager
    and want to paste the script into a managed file themselves.
    """
    resolved = _resolve_shell(shell)

    try:
        script = _completion.generate_script(resolved)
    except ValueError as exc:
        err_console.print(f"[red]completion print: {exc}[/red]")
        raise typer.Exit(code=1)

    # Direct ``print`` (not Rich) so the output is byte-for-byte what
    # a shell would source.
    print(script, end="" if script.endswith("\n") else "\n")


# ---------------------------------------------------------------------------
# gemma completion status
# ---------------------------------------------------------------------------

def status_command(
    shell: str = typer.Option(
        "auto", "--shell",
        help="Target shell (auto|bash|zsh|fish|powershell).",
    ),
) -> None:
    """Report which shell gemma completion thinks it should install for.

    Shows: detected shell, target rc file, whether the block is
    present, and any warnings (e.g. missing ``compinit`` on zsh).
    """
    resolved = _resolve_shell(shell) if shell != "auto" else _platform.detect_shell()

    status = _completion.inspect_installation(resolved)

    console.print(f"[bold]Detected shell:[/bold] {status.shell.value}")
    if status.rc_path is None:
        console.print("[yellow]No default rc-file location for this shell.[/yellow]")
        return

    console.print(f"[bold]rc file:[/bold]        {status.rc_path}")
    console.print(
        f"[bold]rc exists:[/bold]      "
        + ("[green]yes[/green]" if status.rc_exists else "[dim]no[/dim]")
    )
    console.print(
        f"[bold]block installed:[/bold] "
        + ("[green]yes[/green]" if status.block_present else "[red]no[/red]")
    )
    if status.warning:
        console.print(f"[yellow]warning:[/yellow] {status.warning}")
    if not status.block_present:
        console.print(
            "\n[dim]Run `gemma completion install` to add the completion block.[/dim]"
        )


# ---------------------------------------------------------------------------
# gemma completion uninstall
# ---------------------------------------------------------------------------

def uninstall_command(
    shell: str = typer.Option(
        "auto", "--shell",
        help="Target shell (auto|bash|zsh|fish|powershell).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Print what would happen without writing.",
    ),
) -> None:
    """Remove the gemma completion block from your shell's rc file.

    The rc file is archived first — no bytes are lost. For fish, the
    standalone completion file is moved to an archive copy.
    """
    resolved = _resolve_shell(shell)

    try:
        plan = _completion.uninstall(resolved, dry_run=dry_run)
    except ValueError as exc:
        err_console.print(f"[red]completion uninstall: {exc}[/red]")
        raise typer.Exit(code=1)

    if plan.action == "noop":
        err_console.print(
            f"[dim]completion: nothing to remove at {plan.rc_path}[/dim]"
        )
        return

    if dry_run:
        err_console.print(
            f"[yellow]dry-run: would {plan.action} {plan.rc_path}[/yellow]"
        )
        return

    err_console.print(
        f"[green]✓ completion removed ({plan.action}) at {plan.rc_path}[/green]"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hint_next_steps(shell: _platform.Shell, rc_path: Path) -> None:
    """Print the shell-specific 'source this to activate' hint."""
    if shell in (_platform.Shell.BASH, _platform.Shell.ZSH):
        err_console.print(
            f"[dim]Open a new shell or run `source {rc_path}` to activate.[/dim]"
        )
    elif shell is _platform.Shell.FISH:
        err_console.print(
            "[dim]Fish auto-loads completion files; restart your shell to pick it up.[/dim]"
        )
    elif shell is _platform.Shell.POWERSHELL:
        err_console.print(
            "[dim]Reload your PowerShell profile: `. $PROFILE`.[/dim]"
        )
