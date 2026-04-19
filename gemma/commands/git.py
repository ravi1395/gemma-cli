"""Git-assistant commands: ``commit`` and ``diff``.

``commit``
----------
Generates a conventional-commits-style message from staged changes (``git diff
--cached``).  Prints the message for review.  Pass ``--apply`` to create the
commit automatically via ``git commit -m``.

``diff``
--------
Produces a plain-English summary of ``git diff <refspec>`` output.  By default
each changed file gets a one-sentence summary.  Pass ``--overall`` for a single
paragraph covering the whole diff.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from gemma.cache import build_cache
from gemma.client import chat as client_chat
from gemma.commands.clipboard import handle_copy_flags
from gemma.config import Config


console = Console()
err_console = Console(stderr=True)

# Diffs larger than this are truncated before being sent to the model.
_MAX_DIFF_BYTES = 20_000

_COMMIT_SYSTEM_PROMPT = (
    "You write conventional commit messages. "
    "Output ONLY the commit message in this exact format:\n\n"
    "<type>(<scope>): <subject>\n\n"
    "<optional body>\n\n"
    "Rules:\n"
    "- type must be one of: feat, fix, docs, style, refactor, test, chore, perf, ci\n"
    "- scope is optional (the module or component affected)\n"
    "- subject: imperative mood, ≤72 characters, no trailing period\n"
    "- body: optional; explain the motivation. Wrap at 72 characters.\n"
    "- Output ONLY the raw commit message. No markdown fences, no preamble, "
    "no 'Here is your message:' prefix."
)

# Per-file summary prompt — asks for one output line per changed file.
_DIFF_PER_FILE_PROMPT = (
    "You are a code reviewer. "
    "Summarize the following git diff. "
    "For each changed file output exactly one line in this format:\n"
    "  <filename> — <one sentence describing what changed>\n"
    "No preamble, no code fences, no extra commentary. "
    "List every changed file."
)

# Overall summary prompt — produces a single prose paragraph.
_DIFF_OVERALL_PROMPT = (
    "You are a code reviewer. "
    "Write a concise plain-English summary of the following git diff in 2–4 sentences. "
    "Focus on what changed and why it matters. "
    "No code fences, no bullet points — just prose."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_git(*args: str, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    """Run a git sub-command and return the CompletedProcess.

    Parameters
    ----------
    *args:
        Arguments forwarded to ``git`` (e.g. ``"diff"``, ``"--cached"``).
    cwd:
        Working directory for the subprocess.  ``None`` inherits the process cwd.

    Returns
    -------
    subprocess.CompletedProcess
        Always completes — callers check ``returncode``.
    """
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
    )


def _check_git_repo(cwd: Optional[Path] = None) -> bool:
    """Return True if *cwd* (or the current directory) is inside a git repo."""
    return _run_git("rev-parse", "--git-dir", cwd=cwd).returncode == 0


def _truncate_diff(diff: str, max_bytes: int = _MAX_DIFF_BYTES) -> tuple[str, bool]:
    """Truncate *diff* to *max_bytes* bytes (UTF-8).

    Returns
    -------
    tuple[str, bool]
        The (possibly truncated) diff string and a boolean that is ``True``
        when truncation occurred.
    """
    encoded = diff.encode("utf-8")
    if len(encoded) <= max_bytes:
        return diff, False
    truncated = encoded[:max_bytes].decode("utf-8", errors="replace")
    return truncated, True


# ---------------------------------------------------------------------------
# commit
# ---------------------------------------------------------------------------

def commit_command(
    apply: bool = typer.Option(
        False, "--apply",
        help="Create the commit automatically after generating the message.",
    ),
    type_: Optional[str] = typer.Option(
        None, "--type",
        help="Force a conventional-commit type (feat, fix, chore, docs, …).",
    ),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override the Ollama model."),
    keep_alive: Optional[str] = typer.Option(
        None, "--keep-alive",
        help="Ollama model-residency duration (e.g. '30m', '2h').",
    ),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the response cache."),
    cache_only: bool = typer.Option(
        False, "--cache-only",
        help="Error if no cache hit (useful for verifying cached state).",
    ),
    copy: bool = typer.Option(
        False, "--copy",
        help="Copy the generated commit message to the system clipboard.",
    ),
    copy_tee: bool = typer.Option(
        False, "--copy-tee",
        help="Print the message AND copy it to the clipboard.",
    ),
    allow_secrets: bool = typer.Option(
        False, "--allow-secrets",
        help="Allow clipboard copy even if secrets are detected in the output.",
    ),
) -> None:
    """Generate a conventional-commit message from staged changes.

    Reads ``git diff --cached``, asks the model for a message, and prints it
    for review.  Pass ``--apply`` to create the commit automatically.
    """
    if not _check_git_repo():
        err_console.print("[red]gemma commit: not inside a git repository.[/red]")
        raise typer.Exit(code=1)

    git_result = _run_git("diff", "--cached")
    if git_result.returncode != 0:
        err_console.print(f"[red]gemma commit: git error — {git_result.stderr.strip()}[/red]")
        raise typer.Exit(code=1)

    diff = git_result.stdout
    if not diff.strip():
        err_console.print(
            "[yellow]gemma commit: nothing staged. Run 'git add <files>' first.[/yellow]"
        )
        raise typer.Exit(code=1)

    diff, was_truncated = _truncate_diff(diff)
    if was_truncated:
        err_console.print(
            f"[yellow]gemma commit: diff exceeds {_MAX_DIFF_BYTES} bytes — "
            "truncated for the prompt.[/yellow]"
        )

    cfg = Config()
    cfg.temperature = 0.2
    cfg.memory_enabled = False
    if model:
        cfg.model = model
    if keep_alive:
        cfg.ollama_keep_alive = keep_alive

    user_msg = f"Write a commit message for the following diff:\n\n{diff}"
    if type_:
        user_msg = f"Use commit type '{type_}'.\n\n" + user_msg

    messages = [
        {"role": "system", "content": _COMMIT_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    # Cache path: commit always runs at temperature 0.2, well within threshold.
    cache = (
        build_cache(cfg)
        if (not no_cache and cfg.cache_enabled and cfg.temperature <= cfg.cache_temperature_max)
        else None
    )
    commit_msg: Optional[str] = cache.get(messages, cfg) if cache else None

    if commit_msg is None:
        if cache_only:
            err_console.print(
                "[red]gemma commit: no cache hit and --cache-only was set.[/red]"
            )
            raise typer.Exit(code=1)

        raw_parts: list[str] = []
        try:
            for _chunk_type, text in client_chat(messages, cfg, stream=False):
                raw_parts.append(text)
        except Exception as exc:
            err_console.print(f"[red]gemma commit: model error — {exc}[/red]")
            raise typer.Exit(code=1)

        commit_msg = "".join(raw_parts).strip()
        if cache and commit_msg:
            cache.put(messages, cfg, commit_msg)

    if not commit_msg:
        err_console.print("[yellow]gemma commit: model returned an empty response.[/yellow]")
        raise typer.Exit(code=1)

    # Display the generated message for review.
    console.print(Panel(commit_msg, title="[bold cyan]gemma commit[/bold cyan]", border_style="cyan"))

    # Clipboard integration runs before --apply so the message is on the
    # clipboard regardless of whether the user auto-creates the commit.
    handle_copy_flags(
        commit_msg,
        copy=copy, copy_tee=copy_tee,
        allow_secrets=allow_secrets, tool_name="commit",
    )

    if not apply:
        console.print("[dim]Tip: pass --apply to create the commit automatically.[/dim]")
        return

    # Split into subject (first line) and optional body (lines after blank line).
    lines = commit_msg.splitlines()
    subject = lines[0].strip()
    body_lines = lines[2:] if len(lines) > 2 else []
    body = "\n".join(body_lines).strip()

    git_commit_args = ["commit", "-m", subject]
    if body:
        git_commit_args += ["-m", body]

    commit_result = _run_git(*git_commit_args)
    if commit_result.returncode != 0:
        err_console.print(
            f"[red]gemma commit: git commit failed — {commit_result.stderr.strip()}[/red]"
        )
        raise typer.Exit(code=commit_result.returncode)

    console.print("[green]✓ Commit created.[/green]")


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------

def diff_command(
    refspec: Optional[str] = typer.Argument(
        None,
        help="Git refspec to diff (e.g. HEAD~1, main..feature). Default: unstaged working tree.",
    ),
    staged: bool = typer.Option(
        False, "--staged", "--cached",
        help="Summarize staged changes (equivalent to git diff --cached).",
    ),
    overall: bool = typer.Option(
        False, "--overall",
        help="Produce one global summary paragraph instead of per-file summaries.",
    ),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override the Ollama model."),
    keep_alive: Optional[str] = typer.Option(
        None, "--keep-alive",
        help="Ollama model-residency duration (e.g. '30m', '2h').",
    ),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the response cache."),
    cache_only: bool = typer.Option(
        False, "--cache-only",
        help="Error if no cache hit (useful for verifying cached state).",
    ),
    copy: bool = typer.Option(
        False, "--copy",
        help="Copy the diff summary to the system clipboard.",
    ),
    copy_tee: bool = typer.Option(
        False, "--copy-tee",
        help="Print the summary AND copy it to the clipboard.",
    ),
    allow_secrets: bool = typer.Option(
        False, "--allow-secrets",
        help="Allow clipboard copy even if secrets are detected in the output.",
    ),
) -> None:
    """Summarize git diff output in plain English.

    By default each changed file gets a one-sentence description.
    Pass ``--overall`` for a single paragraph covering the whole diff.
    """
    if not _check_git_repo():
        err_console.print("[red]gemma diff: not inside a git repository.[/red]")
        raise typer.Exit(code=1)

    git_args = ["diff"]
    if staged:
        git_args.append("--cached")
    if refspec:
        git_args.append(refspec)

    git_result = _run_git(*git_args)
    if git_result.returncode != 0:
        err_console.print(f"[red]gemma diff: git error — {git_result.stderr.strip()}[/red]")
        raise typer.Exit(code=1)

    diff = git_result.stdout
    if not diff.strip():
        console.print("[dim]No changes to summarize.[/dim]")
        return

    diff, was_truncated = _truncate_diff(diff)
    if was_truncated:
        err_console.print(
            f"[yellow]gemma diff: diff exceeds {_MAX_DIFF_BYTES} bytes — "
            "truncated for the prompt.[/yellow]"
        )

    cfg = Config()
    cfg.temperature = 0.3
    cfg.memory_enabled = False
    if model:
        cfg.model = model
    if keep_alive:
        cfg.ollama_keep_alive = keep_alive

    system_prompt = _DIFF_OVERALL_PROMPT if overall else _DIFF_PER_FILE_PROMPT
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Summarize this diff:\n\n{diff}"},
    ]

    # Cache path: diff runs at temperature 0.3, at the threshold boundary.
    cache = (
        build_cache(cfg)
        if (not no_cache and cfg.cache_enabled and cfg.temperature <= cfg.cache_temperature_max)
        else None
    )
    summary: Optional[str] = cache.get(messages, cfg) if cache else None

    if summary is None:
        if cache_only:
            err_console.print(
                "[red]gemma diff: no cache hit and --cache-only was set.[/red]"
            )
            raise typer.Exit(code=1)

        parts: list[str] = []
        try:
            for _chunk_type, text in client_chat(messages, cfg, stream=False):
                parts.append(text)
        except Exception as exc:
            err_console.print(f"[red]gemma diff: model error — {exc}[/red]")
            raise typer.Exit(code=1)

        summary = "".join(parts).strip()
        if cache and summary:
            cache.put(messages, cfg, summary)

    if not summary:
        err_console.print("[yellow]gemma diff: model returned an empty response.[/yellow]")
        raise typer.Exit(code=1)

    console.print(summary)
    handle_copy_flags(
        summary,
        copy=copy, copy_tee=copy_tee,
        allow_secrets=allow_secrets, tool_name="diff",
    )
