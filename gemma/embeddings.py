"""Thin wrapper around Ollama's embedding endpoint.

We use `nomic-embed-text` (274MB, 768-dim, BERT-family) because it is the
smallest first-party embedding model in the Ollama registry that produces
production-quality dense vectors. It runs on CPU alongside Gemma 4 E4B
without exhausting 16GB unified memory.

If the model is not pulled, `embed()` will raise; callers in the memory
pipeline catch this and degrade to importance-based retrieval.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import ollama


class Embedder:
    """Embed text via Ollama's local embedding API."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
        keep_alive: str = "30m",
    ) -> None:
        self._model = model
        self._client = ollama.Client(host=host)
        # Keep the embedding model resident between calls. Embedding is called
        # on every retrieval and after every condensation, so a cold reload
        # (~1s for nomic-embed-text) would dominate perceived latency.
        self._keep_alive = keep_alive

    @property
    def model(self) -> str:
        return self._model

    def embed(self, text: str) -> np.ndarray:
        """Return the embedding vector for a single string as float32."""
        if not text:
            # Return a zero vector; caller decides whether to store it.
            return np.zeros(0, dtype=np.float32)
        response = self._client.embed(
            model=self._model, input=text, keep_alive=self._keep_alive
        )
        vectors = response.get("embeddings") or []
        if not vectors:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(vectors[0], dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a list of strings. Ollama's embed supports list input directly."""
        if not texts:
            return []
        response = self._client.embed(
            model=self._model, input=texts, keep_alive=self._keep_alive
        )
        vectors = response.get("embeddings") or []
        return [np.asarray(v, dtype=np.float32) for v in vectors]

    def is_available(self) -> bool:
        """Probe with a tiny input to check the model is pulled and reachable."""
        try:
            self.embed("probe")
            return True
        except Exception:
            return False
