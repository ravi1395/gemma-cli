"""Backend abstraction for local LLM runtimes.

The ``LLMBackend`` ABC is the single point of contact between gemma-cli
and any model-serving process. The rest of the codebase treats every
runtime as if it spoke this contract — chat, embeddings, warm-up, and
availability probes — so swapping Ollama for LM Studio (or any future
runtime) is a configuration change, not a code change.

Output contract for ``chat``
----------------------------
Both backends emit the same three-kind tuple stream so renderers in
:mod:`gemma.output` and the agent loop in :mod:`gemma.main` don't care
which backend produced them:

  * ``("think", text)`` — extended-reasoning tokens (Gemma 4 thinking
    mode, DeepSeek-R1, etc.). Backends that don't expose a separate
    reasoning channel never emit this kind.
  * ``("content", text)`` — normal response tokens.
  * ``("metrics", json_str)`` — emitted exactly once at the end with
    ``{"prompt_eval_count": int, "eval_count": int}``. ``json_str`` is
    a JSON-encoded dict so callers that only stream ``content`` can
    detect the sentinel without parsing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generator

if TYPE_CHECKING:  # avoid runtime import cycle with config.py
    import numpy as np

    from gemma.config import Config


# Public type alias — tuples emitted by ``chat()``. ``str`` for the kind
# rather than an Enum so callers (most of which ``match`` on the literal)
# stay simple.
ChatChunk = tuple[str, str]


class LLMBackend(ABC):
    """Common surface for chat + embedding + warm-up across runtimes."""

    #: Short identifier — useful for diagnostics and ``backend describe``.
    name: str = "abstract"

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        config: "Config",
        *,
        stream: bool = True,
    ) -> Generator[ChatChunk, None, None]:
        """Send a pre-built message list and yield ``(kind, text)`` chunks.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` dicts.
                ``role`` is one of ``system | user | assistant``. Order
                matters — backends pass it through verbatim.
            config:   Runtime configuration. Backends read whatever they
                need from it (model, temperature, host, TTL, etc.) so
                the call site stays uniform.
            stream:   When True, yield chunks as they arrive. When False,
                yield the complete response as a single content tuple
                followed by the metrics tuple.

        Yields:
            Three-kind tuples per the module docstring contract.
        """

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    @abstractmethod
    def embed(
        self,
        text: str,
        *,
        model: str,
        config: "Config | None" = None,
    ) -> np.ndarray:
        """Return the embedding vector for a single string as float32.

        Empty inputs must return a length-0 array so the caller can
        detect "this chunk was unembeddable" without an exception.

        ``config`` is optional and only consulted by backends that
        support per-call tuning — currently :class:`LMStudioBackend`
        reads ``config.embed_keep_alive`` to set the JIT-loaded
        embedding model's TTL. ``None`` falls back to backend defaults
        so existing call sites and test stubs need no changes.
        """

    @abstractmethod
    def embed_batch(
        self,
        texts: list[str],
        *,
        model: str,
        config: "Config | None" = None,
    ) -> list[np.ndarray]:
        """Embed a batch of strings.

        Implementations should fall back to per-item embedding when
        the batch call rejects an oversized input — only the offending
        chunk should be lost, never the whole batch.

        ``config`` semantics match :meth:`embed` — optional, used for
        per-call TTL on backends that support it.
        """

    # ------------------------------------------------------------------
    # Warm-up + probes
    # ------------------------------------------------------------------

    @abstractmethod
    def warm_chat(self, config: "Config") -> None:
        """Best-effort: bring the chat model into RAM. Never raise.

        Called from a daemon thread at CLI startup; any exception is
        swallowed because warm-up is fire-and-forget — the foreground
        request will surface a useful error if the runtime is down.
        """

    @abstractmethod
    def warm_embed(self, config: "Config") -> None:
        """Best-effort: warm the embedding model. Same swallow-all rule."""

    @abstractmethod
    def is_embedding_available(self, model: str) -> bool:
        """Probe to confirm the embedding model is loaded and reachable."""


# ---------------------------------------------------------------------------
# Helpers — duration parsing for the keep_alive ↔ TTL mapping
# ---------------------------------------------------------------------------

def parse_keep_alive_seconds(value: str | int | float | None) -> int | None:
    """Convert an Ollama-style keep-alive string to seconds.

    Ollama accepts strings like ``"30m"``, ``"2h"``, ``"-1"`` (forever),
    ``"0"`` (evict now). LM Studio's TTL is an int seconds with ``None``
    meaning "no auto-unload". This helper normalises both worlds:

      * ``"-1"``, ``"forever"``, ``None`` → ``None`` (no expiry)
      * ``"0"``, ``"0s"``                 → ``0`` (evict immediately)
      * ``"30m"`` / ``"2h"`` / ``"45s"``  → seconds as int
      * Plain ``int``                     → returned as-is

    Returns:
        Seconds as ``int``, or ``None`` for "no expiry".

    Raises:
        ValueError: when the string cannot be parsed.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = int(value)
        return None if v < 0 else v

    s = str(value).strip().lower()
    if s in ("-1", "forever", "none", ""):
        return None
    if s in ("0", "0s", "0m", "0h"):
        return 0

    # Trailing unit suffix: "30m", "2h", "45s"
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if s[-1] in units and s[:-1].lstrip("-").isdigit():
        n = int(s[:-1])
        return None if n < 0 else n * units[s[-1]]

    # Bare integer string
    if s.lstrip("-").isdigit():
        n = int(s)
        return None if n < 0 else n

    raise ValueError(f"Cannot parse keep-alive duration {value!r}")
