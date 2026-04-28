"""Backend-agnostic text embedding.

This module used to talk directly to Ollama's ``/api/embed`` endpoint.
After the LM Studio migration it's a thin adapter on top of
:mod:`gemma.backends`: every embed call routes through the active
backend (``LMStudioBackend`` by default, ``OllamaBackend`` when
selected via ``Config.backend``).

The :class:`Embedder` class keeps its original constructor signature
(``model``, ``host``, ``keep_alive``) for backwards compatibility — the
``host`` and ``keep_alive`` arguments are accepted but no longer
authoritative; backends own their connection details. Tests that
already construct an Embedder with positional args continue to work.

The recommended path is to pass either an explicit ``backend`` or a
``Config`` so the active runtime is unambiguous:

    embedder = Embedder(config=cfg)              # auto-resolves backend
    embedder = Embedder(model="…", backend=be)   # for tests / DI
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from gemma.backends import LLMBackend, get_backend


class Embedder:
    """Embed text via the active backend (LM Studio or Ollama).

    The constructor is intentionally permissive so legacy call sites
    in :mod:`gemma.session`, :mod:`gemma.memory.manager`, and
    :mod:`gemma.commands.rag` keep working without churn. New code
    should prefer the ``config=`` form.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,         # kept for backwards-compat; unused
        keep_alive: Optional[str] = None,   # kept for backwards-compat; unused
        *,
        backend: Optional[LLMBackend] = None,
        config: Optional["Config"] = None,  # noqa: F821 — forward ref
    ) -> None:
        # Defer the import so importing this module never pulls in the
        # full Config dataclass at module-load time (matters for the
        # tools subsystem which imports embeddings to type-check).
        from gemma.config import Config

        self._config = config or Config()
        self._backend = backend or get_backend(self._config)

        # Resolve the embedding model: explicit arg > config > platform default.
        self._model = model or self._config.embedding_model
        if not self._model:
            from gemma.platform import default_embedding_model

            self._model = default_embedding_model()

        # Reference legacy kwargs to silence linters; they are no-ops
        # but kept in the signature so existing callers don't blow up.
        _ = host, keep_alive

    @property
    def model(self) -> str:
        return self._model

    def embed(self, text: str) -> np.ndarray:
        """Return the embedding vector for a single string as float32.

        Empty input returns a length-0 array — caller decides whether to
        store or skip it. Errors propagate up; the memory pipeline
        catches and degrades to importance-based retrieval.
        """
        # Forward the held config so the backend can read TTL knobs
        # (e.g. ``embed_keep_alive`` on the LM Studio path).
        return self._backend.embed(text, model=self._model, config=self._config)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a list of strings.

        Backends fall back to per-item embedding on context-length
        errors so a single oversized chunk is the only one lost.
        """
        return self._backend.embed_batch(
            texts, model=self._model, config=self._config
        )

    def is_available(self) -> bool:
        """Probe with a tiny input to check the model is loaded and reachable."""
        return self._backend.is_embedding_available(self._model)
