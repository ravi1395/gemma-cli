"""Model-management subcommands: ``gemma model {pull,list,use,info}``.

Pull
----
Downloads a model from HuggingFace into LM Studio's local cache.
Currently delegates to the ``lms`` CLI (``lms get <repo>``) because the
Python SDK does not expose a download API. If ``lms`` is missing we
print a clear install pointer rather than attempting our own HTTP
download — that path would re-implement chunked downloads, ETag
caching, and quantisation selection that ``lms`` already handles.

List
----
Lists models known to LM Studio. Two flavours:

  * ``--loaded`` (default) — currently resident in RAM. Maps to the
    SDK's ``list_loaded_models()``.
  * ``--downloaded`` — every model on disk, loaded or not. Maps to
    ``list_downloaded_models()``.

Use
---
Writes ``model = "<key>"`` into ``~/.config/gemma/profiles/<name>.toml``
so the chosen model becomes sticky across CLI invocations. Creating the
profile file if absent is intentional — it's the standard config
location and the user gets a single, editable source of truth.

Info
----
Prints the resolved chat + embedding models and the active backend, so
``gemma model info`` is a one-shot answer to "what is gemma actually
talking to right now?".
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from gemma.config import Config
from gemma.platform import (
    default_chat_model,
    default_embedding_model,
    is_apple_silicon,
)


_console = Console()


# ---------------------------------------------------------------------------
# pull
# ---------------------------------------------------------------------------

def pull_command(
    repo: str = typer.Argument(
        ...,
        help=(
            "HuggingFace repository in 'owner/repo' form, e.g. "
            "'mlx-community/gemma-4-E4B-it-4bit' or "
            "'lmstudio-community/Llama-3.2-3B-Instruct-GGUF'."
        ),
    ),
) -> None:
    """Download a HuggingFace model into LM Studio.

    Shells out to LM Studio's ``lms`` CLI (the Python SDK has no
    download API). On a fresh install ``lms`` is on PATH after running
    LM Studio at least once; otherwise we print the bootstrap command.
    """
    if shutil.which("lms") is None:
        _console.print(
            "[red]error:[/red] ``lms`` CLI not found on PATH.\n"
            "Bootstrap it once with: [cyan]npx lmstudio install-cli[/cyan]\n"
            "Or open LM Studio → Settings → Developer → Install lms CLI."
        )
        raise typer.Exit(code=1)

    _console.print(f"[dim]→ lms get {repo}[/dim]")
    # ``lms get`` is interactive (it asks the user to pick a quantisation
    # when the repo has multiple). Inherit stdio so the user can answer.
    proc = subprocess.run(["lms", "get", repo])
    if proc.returncode != 0:
        raise typer.Exit(code=proc.returncode)
    _console.print(
        f"[green]✓[/green] {repo} downloaded. "
        f"Use it with: [cyan]gemma --profile <name> ask ...[/cyan] "
        f"(after [cyan]gemma model use {repo}[/cyan])."
    )


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

def list_command(
    loaded: bool = typer.Option(
        True,
        "--loaded/--downloaded",
        help=(
            "``--loaded`` (default) shows models currently in RAM. "
            "``--downloaded`` lists every model on disk."
        ),
    ),
) -> None:
    """List LM Studio models — loaded in RAM or downloaded on disk."""
    try:
        import lmstudio
    except ImportError:
        _console.print(
            "[red]error:[/red] lmstudio package not installed. Run [cyan]uv sync[/cyan]."
        )
        raise typer.Exit(code=1)

    try:
        models = (
            lmstudio.list_loaded_models()
            if loaded
            else lmstudio.list_downloaded_models()
        )
    except Exception as exc:  # connection refused, server down, etc.
        _console.print(f"[red]error:[/red] cannot reach LM Studio: {exc}")
        _console.print(
            "Is LM Studio running? Start it from the app, or run: [cyan]lms server start[/cyan]"
        )
        raise typer.Exit(code=1)

    if not models:
        scope = "loaded" if loaded else "downloaded"
        _console.print(f"[yellow]No {scope} models.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Identifier")
    table.add_column("Type")
    table.add_column("Size", justify="right")
    for m in models:
        ident = (
            getattr(m, "model_key", None)
            or getattr(m, "identifier", None)
            or getattr(m, "path", "?")
        )
        kind = getattr(m, "type", "") or ("loaded" if loaded else "downloaded")
        size_b = getattr(m, "size_bytes", 0) or 0
        size = _human_bytes(size_b) if size_b else "—"
        table.add_row(str(ident), str(kind), size)
    _console.print(table)


def _human_bytes(n: int) -> str:
    """Format byte counts as a short human string (e.g. '4.2 GB')."""
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(n)
    for u in units:
        if f < 1024 or u == units[-1]:
            return f"{f:.1f} {u}" if u != "B" else f"{int(f)} {u}"
        f /= 1024
    return f"{n} B"


# ---------------------------------------------------------------------------
# use
# ---------------------------------------------------------------------------

def use_command(
    model_key: str = typer.Argument(
        ...,
        help="Model identifier. Either an LM Studio model_key or HF 'owner/repo'.",
    ),
    profile: str = typer.Option(
        "default",
        "--profile",
        "-p",
        help="Write the choice to ~/.config/gemma/profiles/<name>.toml.",
    ),
) -> None:
    """Pin ``model`` in a profile so it's the default for that profile."""
    profile_path = Path.home() / ".config" / "gemma" / "profiles" / f"{profile}.toml"
    profile_path.parent.mkdir(parents=True, exist_ok=True)

    existing = ""
    if profile_path.exists():
        existing = profile_path.read_text()

    new_text = _upsert_toml_field(existing, "model", model_key)
    profile_path.write_text(new_text)
    _console.print(
        f"[green]✓[/green] {profile_path} now pins [cyan]model = {model_key!r}[/cyan]."
    )
    _console.print(
        f"Activate with: [cyan]gemma --profile {profile} ask ...[/cyan]"
    )


def _upsert_toml_field(text: str, key: str, value: str) -> str:
    """Replace or append ``key = "value"`` in a flat TOML document.

    A full TOML round-trip (load → mutate → dump) would lose comments,
    so we do a targeted line-based replacement at the top scope.
    Sufficient for gemma-cli's flat profiles.
    """
    needle = f"{key} = "
    new_line = f'{key} = "{value}"'
    out_lines: list[str] = []
    found = False
    for line in text.splitlines():
        if line.lstrip().startswith(needle) and not found:
            out_lines.append(new_line)
            found = True
        else:
            out_lines.append(line)
    if not found:
        if out_lines and out_lines[-1].strip():
            out_lines.append("")
        out_lines.append(new_line)
    return "\n".join(out_lines) + ("\n" if not text.endswith("\n") or not text else "")


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------

def info_command(
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        "-p",
        help="Resolve against a named profile instead of the default config.",
    ),
) -> None:
    """Show the resolved backend + chat + embedding models for the current config."""
    if profile:
        try:
            cfg = Config.load_profile(profile)
        except FileNotFoundError as exc:
            _console.print(f"[red]error:[/red] {exc}")
            raise typer.Exit(code=1)
    else:
        # Honour the active profile / --backend override applied by the
        # top-level callback. ``main._active_profile`` holds the merged
        # config when the user passed ``--profile`` or ``--backend`` on
        # the parent command. Falling back to a bare Config() means the
        # subcommand still works when invoked stand-alone in tests.
        from gemma import main as _main

        cfg = _main._active_profile or Config()

    table = Table(show_header=False, box=None)
    table.add_column(style="cyan")
    table.add_column()
    table.add_row("backend", cfg.backend)
    table.add_row("chat model", cfg.model or "(unresolved)")
    table.add_row("embedding model", cfg.embedding_model or "(unresolved)")
    table.add_row(
        "platform default (chat)", default_chat_model()
    )
    table.add_row(
        "platform default (embed)", default_embedding_model()
    )
    table.add_row(
        "apple silicon", "yes" if is_apple_silicon() else "no"
    )
    if cfg.backend == "lmstudio":
        host = cfg.lmstudio_host or "(SDK default: localhost:1234)"
        table.add_row("lmstudio host", host)
    else:
        table.add_row("ollama host", cfg.ollama_host)
    table.add_row("keep_alive / TTL", cfg.ollama_keep_alive)
    _console.print(table)
