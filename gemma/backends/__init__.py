"""Pluggable LLM-runtime backends (Ollama, LM Studio, ...).

Every interaction with a model server flows through the ``LLMBackend``
abstraction defined in :mod:`gemma.backends.base`. Two implementations
ship in-tree:

  * :mod:`gemma.backends.lmstudio_backend` — default; uses the native
    ``lmstudio`` Python SDK (TTL/JIT loading, structured ``reasoning_type``
    fragments, embeddings).
  * :mod:`gemma.backends.ollama_backend` — legacy; wraps the ``ollama``
    Python client. Selected when ``Config.backend == "ollama"``.

Pick a backend at runtime with :func:`get_backend`. Callers should never
import the implementations directly so a future swap (e.g. to
llama-cpp-server) is a one-line change.
"""

from __future__ import annotations

from gemma.backends.base import LLMBackend


def get_backend(config: "Config") -> LLMBackend:  # noqa: F821 — forward ref
    """Resolve the configured backend implementation.

    Looks at ``config.backend`` and returns a fresh, ready-to-call
    backend instance. Imports are lazy so a user with only LM Studio
    installed does not pay the cost of importing the ``ollama`` client
    package (and vice versa).

    Raises:
        ValueError: if ``config.backend`` is not a recognised name.
        ImportError: if the optional dependency for the chosen backend
            is not installed (e.g. ``ollama`` extra missing).
    """
    name = (config.backend or "lmstudio").lower()
    if name == "lmstudio":
        from gemma.backends.lmstudio_backend import LMStudioBackend

        return LMStudioBackend()
    if name == "ollama":
        from gemma.backends.ollama_backend import OllamaBackend

        return OllamaBackend()
    raise ValueError(
        f"Unknown backend {name!r}. Set Config.backend to 'lmstudio' or 'ollama'."
    )


__all__ = ["LLMBackend", "get_backend"]
