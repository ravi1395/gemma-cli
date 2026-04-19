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
from rich.markdown import Markdown

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


def render_response(
    generator: Iterator[Tuple[str, str]],
    mode: OutputMode = OutputMode.RICH,
    stream: bool = True,
    field: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Consume a response generator and render output according to *mode*.

    The generator yields ``(chunk_type, text)`` tuples where *chunk_type* is
    ``"think"`` (extended reasoning, shown dimmed in RICH mode) or
    ``"content"`` (the actual response text).

    Args:
        generator: Iterable yielding ``(chunk_type, text)`` pairs.
        mode:      How to format and emit the response.
        stream:    For RICH mode only — stream chunks as they arrive
                   (``True``) or buffer and render as Markdown (``False``).
        field:     For ONLY mode — the JSON field name to print.
        model:     Model tag included in JSON / ONLY output.

    Returns:
        The full response text (content chunks joined; thinking excluded).
    """
    start = time.monotonic()
    chunks: list[str] = []
    thinking_parts: list[str] = []

    if mode == OutputMode.RICH:
        # Streaming path — render inline as chunks arrive.
        _render_rich(generator, stream, chunks, thinking_parts)
    else:
        # Non-streaming: collect everything first, then render.
        for chunk_type, text in generator:
            if chunk_type == "think":
                thinking_parts.append(text)
            else:
                chunks.append(text)

    content = "".join(chunks)
    elapsed_ms = int((time.monotonic() - start) * 1000)

    if mode == OutputMode.JSON:
        _render_json(content, model, elapsed_ms)
    elif mode == OutputMode.ONLY:
        _render_only(content, field, model, elapsed_ms)
    elif mode == OutputMode.CODE:
        _render_code(content)

    return content


# ---------------------------------------------------------------------------
# Private renderers
# ---------------------------------------------------------------------------

def _render_rich(
    generator: Iterator[Tuple[str, str]],
    stream: bool,
    chunks: list[str],
    thinking_parts: list[str],
) -> None:
    """Render in RICH mode (streaming or batched Markdown).

    Mutates *chunks* and *thinking_parts* in-place so the caller has the
    collected text available after return.

    Args:
        generator:     Source of (chunk_type, text) pairs.
        stream:        True → print each chunk as it arrives.
        chunks:        Accumulator for content chunks.
        thinking_parts: Accumulator for thinking chunks.
    """
    if stream:
        had_thinking = False
        first_content = True
        for chunk_type, text in generator:
            if chunk_type == "think":
                if not had_thinking:
                    console.print("[dim italic]thinking…[/dim italic]")
                    had_thinking = True
                console.print(text, end="", soft_wrap=True, highlight=False, style="dim italic")
            else:
                if had_thinking and first_content:
                    console.print()  # newline after the last thinking chunk
                    first_content = False
                chunks.append(text)
                console.print(text, end="", soft_wrap=True, highlight=False)
        console.print()
    else:
        for chunk_type, text in generator:
            if chunk_type == "think":
                thinking_parts.append(text)
            else:
                chunks.append(text)
        if thinking_parts:
            console.rule("[dim]thinking[/dim]", style="dim")
            console.print("".join(thinking_parts), style="dim italic")
            console.rule(style="dim")
        console.print(Markdown("".join(chunks)))


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
