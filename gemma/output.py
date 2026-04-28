"""Output rendering for gemma-cli.

Provides the OutputMode enum and render_response function used by CLI
subcommands to format and emit model responses. Centralising rendering here
allows scripting-friendly modes (JSON, field extraction, code-only) to be
added once and reused across every command.

Modes
-----
RICH  — default; streams Rich-rendered Markdown to the terminal.
JSON  — emits a single JSON object (content, model, elapsed_ms, cache_hit).
ONLY  — prints the value of a single JSON field, naked (no quotes/braces).
CODE  — strips prose and emits only fenced Markdown code blocks.
"""

from __future__ import annotations

import json
import re
import sys
import time
from enum import Enum
from typing import Generator, Iterator, Optional, Tuple

from rich.console import Console

# ``rich.markdown.Markdown`` pulls the ``markdown_it`` parser
# (~50 modules, ~6 ms cumulative). Only the RICH non-stream path
# renders a parsed Markdown block; deferring the import keeps fast
# CLI invocations (``gemma --help``, ``gemma history show``) cheap.

console = Console()

# Fields available in JSON / ONLY output.
_JSON_FIELDS = {"content", "model", "elapsed_ms", "cache_hit"}

# Regex for fenced code blocks: ```[lang]\n<body>```.
# re.DOTALL so '.' matches newlines inside the block.
_FENCE_RE = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)


class OutputMode(Enum):
    """Rendering mode for a model response.

    RICH  — Rich Markdown streaming (default; best for interactive use).
    JSON  — Full JSON object suitable for piping to ``jq``.
    ONLY  — A single JSON field printed naked (no surrounding structure).
    CODE  — Only fenced code blocks extracted from the Markdown response.
    """

    RICH = "rich"
    JSON = "json"
    ONLY = "only"
    CODE = "code"


def display_context_metrics(
    prompt_tokens: int,
    eval_tokens: int,
    context_window: int = 128000,
) -> None:
    """Print a dim token-count footer after a response.

    Args:
        prompt_tokens:  Tokens consumed by the prompt (Ollama prompt_eval_count).
        eval_tokens:    Tokens generated in the completion (Ollama eval_count).
        context_window: Model context window size for the usage percentage.
    """
    if not (prompt_tokens or eval_tokens):
        return
    total = prompt_tokens + eval_tokens
    pct = int(100 * total / context_window) if context_window else 0
    console.print(
        f"[dim]● {prompt_tokens:,} prompt + {eval_tokens:,} completion "
        f"= {total:,} tokens  ({pct}% of {context_window:,} ctx)[/dim]"
    )


def render_response(
    generator: Iterator[Tuple[str, str]],
    mode: OutputMode = OutputMode.RICH,
    stream: bool = True,
    field: Optional[str] = None,
    model: Optional[str] = None,
    context_window: int = 128000,
    show_metrics: bool = True,
) -> Tuple[str, bool]:
    """Consume a response generator and render output according to *mode*.

    The generator yields ``(chunk_type, text)`` tuples where *chunk_type* is
    ``"think"`` (extended reasoning, shown dimmed in RICH mode),
    ``"content"`` (the actual response text), or ``"metrics"`` (a JSON string
    with prompt/completion token counts, emitted once at the end by
    ``gemma.client``).

    Args:
        generator:      Iterable yielding ``(chunk_type, text)`` pairs.
        mode:           How to format and emit the response.
        stream:         For RICH mode only — stream chunks as they arrive
                        (``True``) or buffer and render as Markdown (``False``).
        field:          For ONLY mode — the JSON field name to print.
        model:          Model tag included in JSON / ONLY output.
        context_window: Model context window size used to compute the usage
                        percentage in the metrics footer.
        show_metrics:   When True (default), print a dim token-count footer
                        after the response in RICH mode.

    Returns:
        ``(reply, finished)`` where ``reply`` is the full response text
        (content chunks joined; thinking excluded) and ``finished`` is
        True only when the generator was exhausted cleanly. Stream-and-
        cache (#6) gates ``cache.put`` on ``finished`` so an interrupted
        stream never pollutes the cache with a truncated reply.
    """
    start = time.monotonic()
    chunks: list[str] = []
    # Only the RICH non-stream path renders thinking back out, so other
    # modes drop think chunks on the floor instead of buffering them.
    # ``None`` is the "do not collect" sentinel so a long reasoning chain
    # in JSON/ONLY/CODE mode never costs RAM.
    thinking_parts: Optional[list[str]] = (
        [] if mode == OutputMode.RICH and not stream else None
    )
    # Backends emit metrics exactly once (end-of-stream), so a scalar is
    # enough — no need to retain a list whose only consumer reads [-1].
    metrics_raw: Optional[str] = None
    finished = False

    if mode == OutputMode.RICH:
        # Streaming path — render inline as chunks arrive.
        finished, metrics_raw = _render_rich(
            generator, stream, chunks, thinking_parts
        )
        if show_metrics and metrics_raw is not None:
            _apply_metrics(metrics_raw, context_window)
    else:
        # Non-streaming: collect content + last metrics, drop the rest.
        try:
            for chunk_type, text in generator:
                if chunk_type == "think":
                    pass  # thinking is unused in JSON/ONLY/CODE modes
                elif chunk_type == "metrics":
                    metrics_raw = text
                else:
                    chunks.append(text)
            finished = True
        except Exception:
            finished = False

    content = "".join(chunks)
    elapsed_ms = int((time.monotonic() - start) * 1000)

    if mode == OutputMode.JSON:
        _render_json(content, model, elapsed_ms)
    elif mode == OutputMode.ONLY:
        _render_only(content, field, model, elapsed_ms)
    elif mode == OutputMode.CODE:
        _render_code(content)

    return content, finished


# ---------------------------------------------------------------------------
# Private renderers
# ---------------------------------------------------------------------------

def _apply_metrics(metrics_raw: str, context_window: int) -> None:
    """Parse a metrics JSON string and call display_context_metrics.

    Args:
        metrics_raw:    JSON string with 'prompt_eval_count' and 'eval_count'.
        context_window: Passed through to display_context_metrics.
    """
    try:
        m = json.loads(metrics_raw)
        display_context_metrics(
            m.get("prompt_eval_count", 0),
            m.get("eval_count", 0),
            context_window,
        )
    except Exception:
        pass


def _render_rich(
    generator: Iterator[Tuple[str, str]],
    stream: bool,
    chunks: list[str],
    thinking_parts: Optional[list[str]],
) -> Tuple[bool, Optional[str]]:
    """Render in RICH mode (streaming or batched Markdown).

    Mutates *chunks* in-place so the caller has the collected content
    available for the return value. *thinking_parts*, when not ``None``,
    is also mutated; ``None`` means "drop thinking chunks" (the
    streaming path renders them inline and never collects).

    Args:
        generator:      Source of (chunk_type, text) pairs.
        stream:         True → print each chunk as it arrives.
        chunks:         Accumulator for content chunks.
        thinking_parts: Accumulator for thinking chunks, or ``None`` to
                        skip collection entirely.

    Returns:
        ``(finished, metrics_raw)``. ``finished`` is True only when the
        generator was consumed cleanly to its end; a mid-stream exception
        (network drop, ``generator.throw()``, Ctrl-C) returns False so
        callers can skip cache writes (#6). ``metrics_raw`` is the last
        metrics JSON string emitted, or ``None`` if no metrics arrived.
    """
    metrics_raw: Optional[str] = None
    if stream:
        had_thinking = False
        first_content = True
        try:
            for chunk_type, text in generator:
                if chunk_type == "think":
                    if not had_thinking:
                        console.print("[dim italic]thinking…[/dim italic]")
                        had_thinking = True
                    console.print(text, end="", soft_wrap=True, highlight=False, style="dim italic")
                elif chunk_type == "metrics":
                    metrics_raw = text
                else:
                    if had_thinking and first_content:
                        console.print()  # newline after the last thinking chunk
                        first_content = False
                    chunks.append(text)
                    console.print(text, end="", soft_wrap=True, highlight=False)
        except Exception:
            console.print()
            return False, metrics_raw
        console.print()
        return True, metrics_raw
    else:
        try:
            for chunk_type, text in generator:
                if chunk_type == "think":
                    if thinking_parts is not None:
                        thinking_parts.append(text)
                elif chunk_type == "metrics":
                    metrics_raw = text
                else:
                    chunks.append(text)
        except Exception:
            return False, metrics_raw
        if thinking_parts:
            console.rule("[dim]thinking[/dim]", style="dim")
            console.print("".join(thinking_parts), style="dim italic")
            console.rule(style="dim")
        from rich.markdown import Markdown

        console.print(Markdown("".join(chunks)))
        return True, metrics_raw


def _render_json(content: str, model: Optional[str], elapsed_ms: int) -> None:
    """Emit a full JSON object to stdout.

    Output shape:
        {"content": "...", "model": "...", "elapsed_ms": 1234, "cache_hit": false}

    Args:
        content:    Full response text.
        model:      Model tag (may be None — emitted as empty string).
        elapsed_ms: Wall-clock milliseconds from first token to last.
    """
    payload = {
        "content": content,
        "model": model or "",
        "elapsed_ms": elapsed_ms,
        "cache_hit": False,
    }
    print(json.dumps(payload))


def _render_only(
    content: str,
    field: Optional[str],
    model: Optional[str],
    elapsed_ms: int,
) -> None:
    """Print the value of a single JSON field, unquoted and unstructured.

    Args:
        content:    Full response text.
        field:      One of the keys in *_JSON_FIELDS*.
        model:      Model tag.
        elapsed_ms: Wall-clock milliseconds.

    Raises:
        SystemExit(1): If *field* is not a valid key.
    """
    if field not in _JSON_FIELDS:
        console.print(
            f"[red]error: --only: unknown field {field!r}. "
            f"Valid fields: {', '.join(sorted(_JSON_FIELDS))}[/red]"
        )
        raise SystemExit(1)

    payload: dict = {
        "content": content,
        "model": model or "",
        "elapsed_ms": elapsed_ms,
        "cache_hit": False,
    }
    print(payload[field])


def _render_code(content: str) -> None:
    """Extract and print only fenced Markdown code blocks.

    Uses a simple regex rather than a full Markdown parser.  If the response
    contains no fenced blocks the full content is emitted as a graceful
    fallback so the caller always gets usable output.

    Args:
        content: Full Markdown response text from the model.
    """
    blocks = _FENCE_RE.findall(content)
    if blocks:
        print("\n".join(block.rstrip() for block in blocks))
    else:
        # No fenced blocks found — emit raw content so piping still works.
        print(content.rstrip())
