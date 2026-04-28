"""Backend-agnostic chat shim.

Historically this module called the ``ollama`` Python client directly.
Today it is a thin façade in front of :mod:`gemma.backends`: every call
resolves the active backend from ``Config.backend`` and forwards. The
public API (``chat``, ``ask``) is preserved so existing call sites
across :mod:`gemma.main`, :mod:`gemma.commands`, and the agent loop
need no changes when a user flips ``Config.backend``.

Both functions yield the standard ``(kind, text)`` tuples described in
:mod:`gemma.backends.base`: ``"think"``, ``"content"``, or ``"metrics"``.
"""

from __future__ import annotations

from typing import Generator, Optional

from gemma.backends import get_backend
from gemma.config import Config


def chat(
    messages: list[dict],
    config: Config,
    stream: bool = True,
) -> Generator[tuple[str, str], None, None]:
    """Send a pre-built message list to the active backend.

    Args:
        messages: List of ``{"role": ..., "content": ...}`` dicts.
        config:   Runtime configuration; ``config.backend`` selects
            between LM Studio and Ollama.
        stream:   When True (default), yield chunks as they arrive.

    Yields:
        ``("think", text)`` for extended-reasoning tokens (when the
        model + backend expose them and ``thinking_mode`` is on),
        ``("content", text)`` for response tokens, and exactly one
        terminal ``("metrics", json_str)`` carrying token counts.
    """
    backend = get_backend(config)
    yield from backend.chat(messages, config, stream=stream)


def ask(
    prompt: str,
    config: Config,
    system_prompt: Optional[str] = None,
    stream: bool = True,
) -> Generator[tuple[str, str], None, None]:
    """Single-shot query with no history.

    Convenience wrapper that builds a two-message ``[system, user]``
    list before delegating to :func:`chat`. The system prompt comes
    from the caller or falls back to ``config.system_prompt``.
    """
    messages = [
        {"role": "system", "content": system_prompt or config.system_prompt},
        {"role": "user", "content": prompt},
    ]
    yield from chat(messages, config, stream=stream)
