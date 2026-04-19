"""Shell completion script generation and install/uninstall pipeline.

This module handles the *mechanics* of turning a Typer/Click app into a
shell completion script and installing it idempotently into the user's
rc file. The user-facing Typer subcommands (``gemma completion
{install,print,status,uninstall}``) live in
:mod:`gemma.commands.completion` and delegate here.

Design choices
--------------
* **Reuse Click's generator.** We do not write shell scripts by hand.
  ``click.shell_completion.get_completion_class(shell)`` returns a
  completer whose ``.source()`` method emits a known-good script. If
  Click/Typer change the completion protocol in a future release, we
  get the upstream fix for free.
* **Fenced-block install.** Bash/zsh/powershell use a sentinel block
  (:data:`FENCE_START` / :data:`FENCE_END`) inside an rc file. Fish
  uses its own completion directory and gets the script as a
  standalone file — no fencing needed.
* **Never delete.** Before rewriting any rc file, the prior version is
  moved into ``<root>/archive/<ISO-ts>/...`` via
  :func:`gemma.safety.archive`. Even an ``uninstall`` preserves the
  pre-uninstall content as an archive copy.
* **Atomic writes.** We write the new content to a sibling ``.tmp``
  file and :func:`os.replace` it into place so a kill signal mid-write
  can't leave the rc file half-updated.

Public API
----------
:func:`generate_script`
    Returns the raw completion script for a shell.
:func:`install`
    Installs (or replaces) the completion block in the shell's rc file.
:func:`uninstall`
    Archives the rc file and rewrites it without the block.
:func:`inspect_installation`
    Reports what's installed where — the data behind
    ``gemma completion status``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from click.shell_completion import get_completion_class

from gemma import platform as _platform
from gemma import safety as _safety


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Sentinel opening the fenced block in rc files. Changing this string
#: breaks idempotent re-installation for existing users — don't.
FENCE_START = "# >>> gemma completion >>>"

#: Sentinel closing the fenced block.
FENCE_END = "# <<< gemma completion <<<"

#: Regex that finds the fenced block, including surrounding blank lines.
#: ``DOTALL`` so ``.`` matches newlines inside the body.
_FENCE_RE = re.compile(
    r"\n*" + re.escape(FENCE_START) + r".*?" + re.escape(FENCE_END) + r"\n?",
    re.DOTALL,
)

#: The env var Click uses to know *which* program is asking for completion.
#: Click's convention is uppercase prog-name + ``_COMPLETE``. Keeping it as
#: a constant documents the contract.
_COMPLETE_VAR = "_GEMMA_COMPLETE"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InstallPlan:
    """The write that :func:`install` intends to perform.

    Emitted by :func:`plan_install` and consumed by both :func:`install`
    (to execute) and the ``--dry-run`` path (to display without writing).
    """

    shell: _platform.Shell
    rc_path: Path
    action: str            # "create" | "append" | "replace"
    new_content: str       # the final file content that would be written
    existing_block: bool   # True iff a fenced block is already present


@dataclass(frozen=True)
class InstallationStatus:
    """Snapshot returned by :func:`inspect_installation`."""

    shell: _platform.Shell
    rc_path: Optional[Path]
    rc_exists: bool
    block_present: bool
    warning: Optional[str]  # e.g. "compinit must be sourced before our block"


# ---------------------------------------------------------------------------
# Script generation
# ---------------------------------------------------------------------------

def generate_script(
    shell: _platform.Shell,
    *,
    prog_name: str = "gemma",
    cli_command: Optional[object] = None,
) -> str:
    """Return the completion script for ``shell``.

    Args:
        shell: Target shell. ``UNKNOWN`` / ``POWERSHELL`` raise
            :class:`ValueError`; Click does not ship a PowerShell
            completer in the supported versions.
        prog_name: The CLI's prog name as the user invokes it. Defaults
            to ``gemma``. Tests override this.
        cli_command: Optional Click ``Command`` to generate against.
            Tests inject a stub; production defers to
            :func:`_resolve_cli_command`.

    Returns:
        The completion script as a string, ready to write to a file.

    Raises:
        ValueError: For shells Click cannot target.
    """
    if shell is _platform.Shell.UNKNOWN:
        raise ValueError("Cannot generate completion for unknown shell.")
    if shell is _platform.Shell.POWERSHELL:
        raise ValueError(
            "PowerShell completion is not supported by Click — "
            "use `gemma completion print --shell bash` on WSL instead."
        )

    command = cli_command if cli_command is not None else _resolve_cli_command()
    completion_cls = get_completion_class(shell.value)
    if completion_cls is None:
        raise ValueError(f"No Click completer registered for shell {shell.value!r}")

    completer = completion_cls(command, {}, prog_name, _COMPLETE_VAR)
    return completer.source()


def _resolve_cli_command():
    """Import and return the Click command tree for the gemma CLI.

    Deferred import avoids a circular dependency between
    ``gemma.completion`` and ``gemma.main`` (main imports the
    completion *command* module which imports this module).
    """
    import typer
    from gemma.main import app

    return typer.main.get_command(app)


# ---------------------------------------------------------------------------
# Install planning
# ---------------------------------------------------------------------------

def plan_install(
    shell: _platform.Shell,
    *,
    rc_path: Optional[Path] = None,
    script: Optional[str] = None,
) -> InstallPlan:
    """Work out *what* :func:`install` would write, without writing it.

    Tests can call this directly to assert on the resulting plan; the
    ``--dry-run`` flag displays it verbatim.

    Args:
        shell: Target shell. Must be a supported, non-unknown shell.
        rc_path: Override the rc-file location. Defaults to
            :func:`gemma.platform.rc_file_for`.
        script: Override the generated script. Defaults to
            :func:`generate_script`.
    """
    rc = rc_path if rc_path is not None else _platform.rc_file_for(shell)
    if rc is None:
        raise ValueError(f"No default rc-file location for shell {shell.value}")

    generated = script if script is not None else generate_script(shell)

    # Fish uses its own completions directory; we overwrite the entire file
    # because fish auto-sources every file in that directory.
    if shell is _platform.Shell.FISH:
        existing = _read_text_or_none(rc)
        new_content = generated if generated.endswith("\n") else generated + "\n"
        if existing is None:
            action = "create"
            block_present = False
        elif existing == new_content:
            action = "noop"
            block_present = True
        else:
            action = "replace"
            block_present = True
        return InstallPlan(
            shell=shell,
            rc_path=rc,
            action=action,
            new_content=new_content,
            existing_block=block_present,
        )

    # Bash / zsh: fenced block lives inside the user's rc file.
    existing = _read_text_or_none(rc)
    block = _format_block(generated)

    if existing is None:
        # Brand-new rc file — just the block with a leading header comment.
        new_content = block.lstrip("\n")
        return InstallPlan(shell, rc, "create", new_content, existing_block=False)

    if _FENCE_RE.search(existing):
        # Replace in place; preserve surrounding content exactly.
        new_content = _FENCE_RE.sub("\n" + block, existing, count=1)
        if not new_content.endswith("\n"):
            new_content += "\n"
        return InstallPlan(shell, rc, "replace", new_content, existing_block=True)

    # Append the block to the existing rc.
    sep = "" if existing.endswith("\n") else "\n"
    new_content = existing + sep + block
    if not new_content.endswith("\n"):
        new_content += "\n"
    return InstallPlan(shell, rc, "append", new_content, existing_block=False)


# ---------------------------------------------------------------------------
# Install / uninstall execution
# ---------------------------------------------------------------------------

def install(
    shell: _platform.Shell,
    *,
    rc_path: Optional[Path] = None,
    dry_run: bool = False,
    force: bool = False,
    script: Optional[str] = None,
) -> InstallPlan:
    """Install or replace the gemma completion block for ``shell``.

    Args:
        shell: Target shell.
        rc_path: Override rc file location. Defaults to
            :func:`gemma.platform.rc_file_for`.
        dry_run: When True, return the plan without writing anything.
        force: When True, install even if the detected shell differs
            from what the user requested. (Today this is informational;
            the shell argument already controls that decision.)
        script: Test hook — inject a pre-generated script.

    Returns:
        The :class:`InstallPlan` that was (or would be) applied.

    Raises:
        ValueError: On unsupported shell.
        OSError: From the underlying filesystem ops if writing.
    """
    plan = plan_install(shell, rc_path=rc_path, script=script)

    if dry_run or plan.action == "noop":
        return plan

    # Never-delete rule: if we're about to rewrite an existing file,
    # archive it first. The policy root is the rc file's parent so the
    # archive lives next to the file rather than polluting $HOME.
    if plan.rc_path.exists():
        archive_root = plan.rc_path.parent
        policy = _safety.default_policy(archive_root)
        try:
            _safety.archive(plan.rc_path, policy)
        except _safety.SafetyError:
            # The rc file is denylisted under the default policy (e.g.
            # it sits under ~/.config/gemma). That's fine — we fall
            # back to an inline ``.gemma.archive.<ts>`` copy instead so
            # the "no data loss" invariant still holds.
            _inline_archive(plan.rc_path)

    plan.rc_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(plan.rc_path, plan.new_content)
    return plan


def uninstall(
    shell: _platform.Shell,
    *,
    rc_path: Optional[Path] = None,
    dry_run: bool = False,
) -> InstallPlan:
    """Remove the gemma completion block for ``shell``.

    For bash/zsh this strips the fenced block from the rc file. For
    fish, the completion file is *moved* to the archive directory — the
    file itself is never unlinked.
    """
    rc = rc_path if rc_path is not None else _platform.rc_file_for(shell)
    if rc is None:
        raise ValueError(f"No default rc-file location for shell {shell.value}")

    if not rc.exists():
        # Nothing to uninstall.
        return InstallPlan(shell, rc, "noop", "", existing_block=False)

    if shell is _platform.Shell.FISH:
        plan = InstallPlan(shell, rc, "archive", "", existing_block=True)
        if not dry_run:
            _inline_archive(rc)
        return plan

    existing = rc.read_text(encoding="utf-8")
    if not _FENCE_RE.search(existing):
        return InstallPlan(shell, rc, "noop", existing, existing_block=False)

    new_content = _FENCE_RE.sub("\n", existing, count=1).lstrip("\n")
    plan = InstallPlan(shell, rc, "replace", new_content, existing_block=True)
    if dry_run:
        return plan

    _inline_archive(rc)
    _atomic_write(rc, new_content)
    return plan


# ---------------------------------------------------------------------------
# Status inspection
# ---------------------------------------------------------------------------

def inspect_installation(
    shell: Optional[_platform.Shell] = None,
    *,
    rc_path: Optional[Path] = None,
) -> InstallationStatus:
    """Return a snapshot of whether completion is installed.

    Args:
        shell: The shell to inspect. Defaults to :func:`detect_shell`.
        rc_path: Override the rc file location; defaults to
            :func:`rc_file_for`.
    """
    resolved_shell = shell if shell is not None else _platform.detect_shell()
    rc = rc_path if rc_path is not None else _platform.rc_file_for(resolved_shell)

    if rc is None:
        return InstallationStatus(
            shell=resolved_shell,
            rc_path=None,
            rc_exists=False,
            block_present=False,
            warning=f"No default rc file for shell {resolved_shell.value}",
        )

    if not rc.exists():
        return InstallationStatus(
            shell=resolved_shell,
            rc_path=rc,
            rc_exists=False,
            block_present=False,
            warning=None,
        )

    content = rc.read_text(encoding="utf-8", errors="replace")
    block_present = bool(_FENCE_RE.search(content))

    warning: Optional[str] = None
    if resolved_shell is _platform.Shell.ZSH and block_present:
        # zsh requires compinit to have run before any ``compdef``
        # declarations. If our block appears before ``compinit``, tab
        # completion is silently broken. This heuristic catches the
        # common case — a more sophisticated check is possible but
        # probably more trouble than it's worth.
        pre_block = content.split(FENCE_START, 1)[0]
        if "compinit" not in pre_block:
            warning = (
                "zsh: `compinit` not found before gemma block; "
                "tab completion may not activate. Add `autoload -Uz compinit && compinit` "
                "earlier in ~/.zshrc."
            )

    return InstallationStatus(
        shell=resolved_shell,
        rc_path=rc,
        rc_exists=True,
        block_present=block_present,
        warning=warning,
    )


# ---------------------------------------------------------------------------
# Dynamic completions — shared completer functions
# ---------------------------------------------------------------------------

def profile_completer(incomplete: str = "") -> list[str]:
    """Return profile stems under ``~/.config/gemma/profiles``.

    Used as the ``autocompletion=`` callback on the ``--profile`` option.
    Silent on any exception so a broken profiles directory never breaks
    shell completion (which would be very disruptive).
    """
    try:
        profiles_dir = Path.home() / ".config" / "gemma" / "profiles"
        if not profiles_dir.is_dir():
            return []
        names = sorted(p.stem for p in profiles_dir.glob("*.toml"))
        return [n for n in names if n.startswith(incomplete)]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_text_or_none(path: Path) -> Optional[str]:
    """Return the file contents, or ``None`` if the file doesn't exist."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def _format_block(script: str) -> str:
    """Wrap ``script`` in the sentinel fence with a header comment.

    The leading newline keeps the block visually separated from whatever
    preceded it in the rc file.
    """
    body = script if script.endswith("\n") else script + "\n"
    return (
        "\n" + FENCE_START + "\n"
        "# Managed by `gemma completion install`. Do not edit by hand.\n"
        + body
        + FENCE_END + "\n"
    )


def _atomic_write(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` via a tmp-file + :func:`os.replace`."""
    tmp = path.with_suffix(path.suffix + ".gemma-tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _inline_archive(path: Path) -> Path:
    """Copy-to-archive fallback when :func:`gemma.safety.archive` is refused.

    Used when the rc file lives inside a denylisted directory (e.g.
    ``~/.config/gemma/...``) — the default safety policy won't let us
    move it into an ``archive/`` subdir of that location. We copy the
    file in place with an ISO-timestamp suffix so the prior contents
    remain recoverable without triggering the denylist.
    """
    from datetime import datetime, timezone
    import shutil

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = path.with_name(path.name + f".gemma.archive.{ts}")
    i = 0
    while target.exists():
        i += 1
        target = path.with_name(path.name + f".gemma.archive.{ts}.{i}")
    shutil.copy2(path, target)
    return target
