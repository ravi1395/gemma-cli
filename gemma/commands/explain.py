"""``gemma explain`` — plain-English explanation of text, files, or commands.

Input is auto-detected from the invocation context:

  - **stdin pipe** (``echo "error" | gemma explain``) — reads all of stdin and
    explains it.  Useful for piping in stack traces, log snippets, man-page
    output, etc.
  - **file path** argument — reads the file and explains it.  The ``--lines``
    flag limits how many lines are read (default: all, capped at 20 KB).
  - ``--cmd "…"`` — explains what a shell command does, without running it.
  - ``--error "…"`` — explains a specific error string or message.

If more than one input source is provided the priority order is:
  file argument > ``--cmd`` > ``--error`` > stdin.

By default the command is stateless (no memory read/write).  Pass
``--with-memory`` to retrieve relevant context from Redis before answering.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown

from gemma.cache import build_cache
from gemma.client import chat as client_chat
from gemma.config import Config
from gemma.memory import MemoryManager


console = Console()
err_console = Console(stderr=True)

# Maximum bytes read from a file or stdin before we truncate.
_MAX_BYTES = 20_000

_SYSTEM_PROMPT = (
    "You are a technical explainer. "
    "Start with a single-sentence summary on its own line, then list key points as bullet points. "
    "Be concise and precise. Avoid repeating the input verbatim unless quoting a specific part "
    "is essential for clarity. Tailor the depth of explanation to the complexity of the input."
)


def _build_user_message(input_kind: str, content: str) -> str:
    """Format the user message with a labelled input block."""
    return f"Explain the following {input_kind}:\n\n{content}"


def _read_file(path: Path, lines: Optional[int]) -> str:
    """Read a file up to _MAX_BYTES (or first `lines` lines if given)."""
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")
    if lines is not None:
        with path.open(encoding="utf-8", errors="replace") as fh:
            raw = "\n".join(fh.readline().rstrip("\n") for _ in range(lines))
    else:
        raw = path.read_bytes()[:_MAX_BYTES].decode("utf-8", errors="replace")
    return raw


def explain_command(
    file: Optional[str] = typer.Argument(
        None, help="Path to a file to explain."
    ),
    cmd: Optional[str] = typer.Option(
        None, "--cmd", help="Shell command to explain (not executed)."
    ),
    error: Optional[str] = typer.Option(
        None, "--error", help="Error message or exception string to explain."
    ),
    lines: Optional[int] = typer.Option(
        None, "--lines", "-n",
        help="For file mode: only read and explain the first N lines.",
    ),
    with_memory: bool = typer.Option(
        False, "--with-memory",
        help="Retrieve relevant context from the memory system before answering.",
    ),
    no_stream: bool = typer.Option(False, "--no-stream"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    keep_alive: Optional[str] = typer.Option(None, "--keep-alive"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the response cache."),
    cache_only: bool = typer.Option(
        False, "--cache-only",
        help="Error if no cache hit (requires --no-stream).",
    ),
) -> None:
    """Explain text, a file, a command, or an error in plain English."""

    # ------------------------------------------------------------------
    # Resolve input source (priority: file > --cmd > --error > stdin)
    # ------------------------------------------------------------------
    input_kind: str
    content: str

    if file is not None:
        path = Path(file).expanduser()
        try:
            content = _read_file(path, lines)
        except FileNotFoundError as exc:
            err_console.print(f"[red]gemma explain: {exc}[/red]")
            raise typer.Exit(code=1)
        if not content.strip():
            err_console.print("[yellow]gemma explain: file is empty[/yellow]")
            raise typer.Exit(code=1)
        input_kind = f"file ({path.name})"

    elif cmd is not None:
        content = cmd
        input_kind = "shell command"

    elif error is not None:
        content = error
        input_kind = "error message"

    elif not sys.stdin.isatty():
        raw = sys.stdin.buffer.read(_MAX_BYTES).decode("utf-8", errors="replace")
        content = raw.strip()
        if not content:
            err_console.print("[yellow]gemma explain: stdin was empty[/yellow]")
            raise typer.Exit(code=1)
        input_kind = "text"

    else:
        # No input source available — print usage hint.
        err_console.print(
            "[yellow]gemma explain: provide a file path, --cmd, --error, or pipe text via stdin.[/yellow]\n"
            "[dim]Examples:[/dim]\n"
            "  [dim]gemma explain error.log[/dim]\n"
            "  [dim]gemma explain --cmd 'find . -mtime -1 -name \"*.py\"'[/dim]\n"
            "  [dim]echo 'Segmentation fault' | gemma explain[/dim]"
        )
        raise typer.Exit(code=1)

    # ------------------------------------------------------------------
    # Build config and messages
    # ------------------------------------------------------------------
    cfg = Config()
    cfg.memory_enabled = with_memory
    if model:
        cfg.model = model
    if keep_alive:
        cfg.ollama_keep_alive = keep_alive

    user_msg = _build_user_message(input_kind, content)

    if with_memory:
        # Use the full memory-augmented context path.
        mgr = MemoryManager(cfg)
        mgr.initialize()
        if mgr.degraded and cfg.memory_enabled:
            console.print(
                "[yellow]memory: Redis unreachable, running without memory context[/yellow]"
            )
        messages = mgr.get_context_messages(user_msg, system_prompt=_SYSTEM_PROMPT)
    else:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

    # ------------------------------------------------------------------
    # Stream / collect response
    # ------------------------------------------------------------------
    stream = not no_stream

    # Cache path: only applicable for non-streaming calls within the
    # configured temperature threshold.
    cache = (
        build_cache(cfg)
        if (not stream and not no_cache and cfg.cache_enabled
                and cfg.temperature <= cfg.cache_temperature_max)
        else None
    )
    cached_content: Optional[str] = cache.get(messages, cfg) if cache else None

    if cached_content is not None:
        console.print(Markdown(cached_content))
        return

    if cache_only and not stream:
        err_console.print(
            "[red]gemma explain: no cache hit and --cache-only was set.[/red]"
        )
        raise typer.Exit(code=1)

    chunks: list[str] = []

    try:
        if stream and sys.stdout.isatty():
            # Rich streaming — print as chunks arrive.
            for _chunk_type, text in client_chat(messages, cfg, stream=True):
                chunks.append(text)
                console.print(text, end="", soft_wrap=True, highlight=False)
            console.print()
        else:
            # Collect then render as Markdown (nicer for piped output too).
            for _chunk_type, text in client_chat(messages, cfg, stream=stream):
                chunks.append(text)
            console.print(Markdown("".join(chunks)))
    except Exception as exc:
        err_console.print(f"[red]gemma explain: model error — {exc}[/red]")
        raise typer.Exit(code=1)

    # Store the collected non-streaming response in cache.
    if cache and chunks and not stream:
        cache.put(messages, cfg, "".join(chunks))
