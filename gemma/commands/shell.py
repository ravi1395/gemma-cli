"""Shell-assistant commands: ``sh``, ``why``, and ``install-shell``.

``sh``
------
Translates a natural-language description into a single shell command.
The model is called with a tight system prompt at low temperature so output
is a bare command with no prose or fences.  If stdout is a TTY the user is
prompted to confirm before the command runs.  Use ``--no-exec`` to always
print only (good for piping to clipboard, etc.).

``why``
-------
Explains why the last shell command failed.  Reads the file written by the
shell hook installed via ``install-shell`` (default: ``~/.gemma_last_cmd``).

``install-shell``
-----------------
Prints (or appends) a small shell snippet that captures the last command and
exit code into ``$GEMMA_LAST_FILE`` after every prompt.  Supports bash and
zsh.  v1 captures command + exit code only; stderr capture is deferred.
"""

from __future__ import annotations

import os
import subprocess
import sys
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

# ---------------------------------------------------------------------------
# Safety blocklist for `sh --exec` path.
# These substrings are rejected regardless of surrounding context.
# Kept intentionally short to avoid false-positives on legitimate commands.
# ---------------------------------------------------------------------------
_DANGEROUS_PATTERNS: list[str] = [
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    "dd if=",
    ":(){:|:&};:",   # fork bomb
    ">(){ :|:& };:", # zsh fork bomb variant
]

_SH_SYSTEM_PROMPT = (
    "You are a shell-command generator. "
    "Output EXACTLY one shell command that fulfils the user's request. "
    "Rules: no prose, no explanation, no markdown, no code fences, no comments. "
    "If the request is ambiguous, choose the safest, most common interpretation. "
    "If the request cannot be expressed as a single safe shell command, output nothing."
)

_WHY_SYSTEM_PROMPT = (
    "You are a shell debugging assistant. "
    "The user ran a command that failed. "
    "Explain the most likely cause in two to four sentences, then suggest a concrete fix. "
    "Be direct and specific. Do not repeat the command verbatim unless needed for clarity."
)

# Default path for the last-command record written by the shell hook.
_DEFAULT_LAST_CMD_FILE = Path("~/.gemma_last_cmd").expanduser()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detected_shell() -> str:
    """Return the basename of the user's login shell (e.g. 'bash', 'zsh')."""
    shell_path = os.environ.get("SHELL", "/bin/sh")
    return Path(shell_path).name


def _is_dangerous(cmd: str) -> bool:
    """Return True if the command matches any entry in the blocklist."""
    lower = cmd.lower()
    return any(pat in lower for pat in _DANGEROUS_PATTERNS)


def _clean_model_command(raw: str) -> str:
    """Strip whitespace and accidental code fences from model output.

    The system prompt asks for no fences, but models occasionally add them
    anyway.  We strip the outermost fence block if present.
    """
    text = raw.strip()
    # Strip ```...``` or ```shell...``` wrappers
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first line (``` or ```shell) and last ``` line
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner).strip()
    return text


# ---------------------------------------------------------------------------
# sh
# ---------------------------------------------------------------------------

def sh_command(
    prompt: str = typer.Argument(..., help="Natural-language description of the command you want."),
    no_exec: bool = typer.Option(
        False, "--no-exec",
        help="Print the command only; never prompt to run. Safe for piping.",
    ),
    shell: Optional[str] = typer.Option(
        None, "--shell",
        help="Target shell syntax: bash, zsh, sh. Defaults to $SHELL.",
    ),
    explain: bool = typer.Option(
        False, "--explain",
        help="Print a one-line comment above the command describing what it does.",
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
        help="Copy the generated command to the system clipboard.",
    ),
    copy_tee: bool = typer.Option(
        False, "--copy-tee",
        help="Print the command AND copy it to the clipboard.",
    ),
    allow_secrets: bool = typer.Option(
        False, "--allow-secrets",
        help="Allow clipboard copy even if secrets are detected in the output.",
    ),
) -> None:
    """Translate a natural-language description into a shell command."""
    cfg = Config()
    cfg.temperature = 0.2
    cfg.memory_enabled = False
    if model:
        cfg.model = model
    if keep_alive:
        cfg.ollama_keep_alive = keep_alive

    target_shell = shell or _detected_shell()

    user_msg = (
        f"Target shell: {target_shell}\n"
        f"Request: {prompt}"
    )
    if explain:
        user_msg += "\nAlso output a ONE-LINE shell comment (# ...) on the line above the command."

    messages = [
        {"role": "system", "content": _SH_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    # Cache path: sh runs at temperature 0.2, well within threshold.
    cache = (
        build_cache(cfg)
        if (not no_cache and cfg.cache_enabled and cfg.temperature <= cfg.cache_temperature_max)
        else None
    )
    raw: Optional[str] = cache.get(messages, cfg) if cache else None

    if raw is None:
        if cache_only:
            err_console.print(
                "[red]gemma sh: no cache hit and --cache-only was set.[/red]"
            )
            raise typer.Exit(code=1)

        # Collect the full response (non-streaming; we need it before acting)
        raw_parts: list[str] = []
        try:
            for _chunk_type, text in client_chat(messages, cfg, stream=False):
                raw_parts.append(text)
        except Exception as exc:
            err_console.print(f"[red]gemma sh: model error — {exc}[/red]")
            raise typer.Exit(code=1)

        raw = "".join(raw_parts)
        if cache and raw:
            cache.put(messages, cfg, raw)
    cmd = _clean_model_command(raw)

    if not cmd:
        err_console.print("[yellow]gemma sh: model returned an empty response[/yellow]")
        raise typer.Exit(code=1)

    # Extract the last non-comment, non-empty line as the executable command
    # (when --explain is on the model prepends a comment line).
    exec_line = cmd
    for line in reversed(cmd.splitlines()):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            exec_line = stripped
            break

    # ------------------------------------------------------------------
    # Pipe-mode: just emit the bare command, no formatting, no prompt.
    # ------------------------------------------------------------------
    if not sys.stdout.isatty() or no_exec:
        # If --explain was requested, include the comment line too
        print(cmd)
        # Clipboard integration: copy the bare command, not the possibly
        # commented-out --explain prefix, so paste-into-terminal "just
        # works".
        handle_copy_flags(
            exec_line,
            copy=copy, copy_tee=copy_tee,
            allow_secrets=allow_secrets, tool_name="sh",
        )
        return

    # ------------------------------------------------------------------
    # Interactive mode: show a panel, then optionally run.
    # ------------------------------------------------------------------
    console.print(Panel(cmd, title="[bold cyan]gemma sh[/bold cyan]", border_style="cyan"))

    # Interactive clipboard copy happens before the run-prompt so the
    # command is already on the clipboard even if the user declines to
    # execute it (the common "copy, edit slightly, paste" flow).
    handle_copy_flags(
        exec_line,
        copy=copy, copy_tee=copy_tee,
        allow_secrets=allow_secrets, tool_name="sh",
    )

    # Safety check — warn and refuse to exec dangerous patterns.
    if _is_dangerous(exec_line):
        err_console.print(
            "[bold red]⚠ gemma sh: this command matches a safety blocklist "
            "and will not be executed.[/bold red]"
        )
        return

    answer = typer.prompt("Run this?", default="N").strip().lower()
    if answer not in {"y", "yes"}:
        return

    shell_bin = shell or os.environ.get("SHELL", "/bin/sh")
    result = subprocess.run(exec_line, shell=True, executable=shell_bin)
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)


# ---------------------------------------------------------------------------
# why
# ---------------------------------------------------------------------------

def why_command(
    last_file: Optional[str] = typer.Option(
        None, "--last-file", envvar="GEMMA_LAST_FILE",
        help="Path to the last-command record (default: ~/.gemma_last_cmd).",
    ),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    keep_alive: Optional[str] = typer.Option(None, "--keep-alive"),
) -> None:
    """Explain why the last shell command failed.

    Requires the shell hook to be installed via ``gemma install-shell``.
    """
    record_path = Path(last_file).expanduser() if last_file else _DEFAULT_LAST_CMD_FILE

    if not record_path.exists():
        err_console.print(
            "[yellow]gemma why: no last-command record found. "
            "Run 'gemma install-shell' and source the snippet first.[/yellow]"
        )
        raise typer.Exit(code=1)

    raw = record_path.read_text(encoding="utf-8").strip()
    if not raw:
        err_console.print("[yellow]gemma why: last-command record is empty.[/yellow]")
        raise typer.Exit(code=1)

    # Expected format written by the hook: "<exit_code>\t<command>\t<stderr>"
    # v1 may only have two fields (no stderr capture yet).
    parts = raw.split("\t", maxsplit=2)
    exit_code = parts[0] if len(parts) > 0 else "?"
    command = parts[1] if len(parts) > 1 else raw
    stderr_snippet = parts[2] if len(parts) > 2 else ""

    if exit_code == "0":
        console.print("[green]Last command succeeded (exit 0). Nothing to explain.[/green]")
        return

    cfg = Config()
    cfg.memory_enabled = False
    if model:
        cfg.model = model
    if keep_alive:
        cfg.ollama_keep_alive = keep_alive

    body = f"Command: {command}\nExit code: {exit_code}"
    if stderr_snippet:
        body += f"\nStderr:\n{stderr_snippet}"

    messages = [
        {"role": "system", "content": _WHY_SYSTEM_PROMPT},
        {"role": "user", "content": body},
    ]

    console.print(f"[dim]last command:[/dim] {command}  [dim](exit {exit_code})[/dim]\n")
    try:
        for _chunk_type, text in client_chat(messages, cfg, stream=True):
            console.print(text, end="", soft_wrap=True, highlight=False)
        console.print()
    except Exception as exc:
        err_console.print(f"[red]gemma why: model error — {exc}[/red]")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# install-shell
# ---------------------------------------------------------------------------

_BASH_SNIPPET = """\
# ── gemma-cli shell hook (bash) ──────────────────────────────────────────────
export GEMMA_LAST_FILE="${GEMMA_LAST_FILE:-$HOME/.gemma_last_cmd}"
_gemma_pre()  { _gemma_last_cmd="$BASH_COMMAND"; }
_gemma_post() {
  local ec=$?
  # Only record when a real command ran (skip empty prompts).
  [ -n "$_gemma_last_cmd" ] || return
  printf '%s\\t%s\\n' "$ec" "$_gemma_last_cmd" > "$GEMMA_LAST_FILE"
}
trap '_gemma_pre' DEBUG
PROMPT_COMMAND="_gemma_post${PROMPT_COMMAND:+;$PROMPT_COMMAND}"
# ─────────────────────────────────────────────────────────────────────────────
"""

_ZSH_SNIPPET = """\
# ── gemma-cli shell hook (zsh) ───────────────────────────────────────────────
export GEMMA_LAST_FILE="${GEMMA_LAST_FILE:-$HOME/.gemma_last_cmd}"
_gemma_preexec()  { _gemma_last_cmd="$1"; }
_gemma_precmd() {
  local ec=$?
  [ -n "$_gemma_last_cmd" ] || return
  printf '%s\\t%s\\n' "$ec" "$_gemma_last_cmd" > "$GEMMA_LAST_FILE"
  _gemma_last_cmd=""
}
autoload -Uz add-zsh-hook
add-zsh-hook preexec _gemma_preexec
add-zsh-hook precmd  _gemma_precmd
# ─────────────────────────────────────────────────────────────────────────────
"""

_SUPPORTED_SHELLS: dict[str, str] = {
    "bash": _BASH_SNIPPET,
    "zsh": _ZSH_SNIPPET,
}


def install_shell_command(
    shell: Optional[str] = typer.Option(
        None, "--shell",
        help="Target shell: bash or zsh. Defaults to $SHELL.",
    ),
    append: Optional[str] = typer.Option(
        None, "--append",
        help="Append the snippet to this rc file (e.g. ~/.bashrc). Backs up the file first.",
    ),
) -> None:
    """Print (or append) the shell hook snippet for ``gemma why``.

    Source the printed snippet in your shell rc file, or pass ``--append
    ~/.bashrc`` to have gemma-cli do it automatically.
    """
    target = (shell or _detected_shell()).lower()
    if target not in _SUPPORTED_SHELLS:
        err_console.print(
            f"[red]gemma install-shell: unsupported shell '{target}'. "
            f"Supported: {', '.join(_SUPPORTED_SHELLS)}[/red]"
        )
        raise typer.Exit(code=1)

    snippet = _SUPPORTED_SHELLS[target]

    if not append:
        # Default: just print; user sources it themselves.
        print(snippet, end="")
        return

    rc_path = Path(append).expanduser()
    # Use name + suffix so dotfiles (.bashrc) become .bashrc.gemma-backup
    # rather than the incorrect .gemma-backup produced by with_suffix on a
    # file whose suffix is the empty string.
    backup = rc_path.parent / (rc_path.name + ".gemma-backup")

    # Back up before modifying.
    if rc_path.exists():
        rc_path.replace(backup)
        rc_path.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
        console.print(f"[dim]backed up {rc_path} → {backup}[/dim]")

    with rc_path.open("a", encoding="utf-8") as fh:
        fh.write("\n")
        fh.write(snippet)

    console.print(
        f"[green]✓ Appended {target} hook to {rc_path}. "
        f"Run:[/green] [bold]source {rc_path}[/bold]"
    )
