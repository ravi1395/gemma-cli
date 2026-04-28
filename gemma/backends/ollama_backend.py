"""Ollama implementation of :class:`gemma.backends.base.LLMBackend`.

This module owns every direct call to the ``ollama`` Python client. It
preserves the behaviour of the pre-refactor ``gemma.client`` /
``gemma.embeddings`` modules verbatim — same chunk shape, same metrics
keys, same fallback-on-batch-error logic — so callers that switch to
this backend get bit-for-bit equivalent output.

The ``ollama`` package is imported lazily inside methods so installations
that only use LM Studio do not fail at import time when the optional
``ollama`` extra is absent.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Generator

import numpy as np

from gemma.backends.base import ChatChunk, LLMBackend

if TYPE_CHECKING:
    from gemma.config import Config


# Module-level import. We catch ImportError so installations that drop
# the optional ``ollama`` extra still load this file (the in-tree
# ``LMStudioBackend`` is the default and shouldn't pull ``ollama`` into
# the dependency closure). Tests patch ``ollama.Client`` via
# ``gemma.backends.ollama_backend.ollama.Client`` — that handle stays
# stable here, mirroring the legacy ``gemma.client.ollama`` seam.
try:
    import ollama  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover — exercised only when extra is missing
    ollama = None  # type: ignore[assignment]


def _require_ollama():
    """Return the imported ``ollama`` module or raise a friendly error."""
    if ollama is None:
        raise ImportError(
            "The Ollama backend requires the optional 'ollama' extra. "
            "Install with: uv add 'gemma-cli[ollama]'  (or "
            "'pip install ollama>=0.5')."
        )
    return ollama


def _extract_metrics(chunk: object) -> dict:
    """Pull prompt/completion token counts off an Ollama chunk.

    Handles both dict-style (raw API) and Pydantic-object-style (SDK).
    """
    def _get(key: str) -> int:
        if isinstance(chunk, dict):
            return int(chunk.get(key) or 0)
        return int(getattr(chunk, key, 0) or 0)

    return {
        "prompt_eval_count": _get("prompt_eval_count"),
        "eval_count": _get("eval_count"),
    }


class OllamaBackend(LLMBackend):
    """Backend that talks to a local Ollama daemon over HTTP."""

    name = "ollama"

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        config: "Config",
        *,
        stream: bool = True,
    ) -> Generator[ChatChunk, None, None]:
        ollama = _require_ollama()
        client = ollama.Client(host=config.ollama_host)
        response = client.chat(
            model=config.model,
            messages=messages,
            stream=stream,
            think=config.thinking_mode,
            # ``keep_alive`` keeps the model resident between calls — worth
            # ~1–2 s of TTFT on warm re-invocations.
            keep_alive=config.ollama_keep_alive,
            options={"temperature": config.temperature},
        )
        if stream:
            last_chunk: object = {}
            for chunk in response:
                last_chunk = chunk
                thinking = (chunk["message"].get("thinking") or "")
                content = (chunk["message"].get("content") or "")
                if thinking:
                    yield ("think", thinking)
                if content:
                    yield ("content", content)
            yield ("metrics", json.dumps(_extract_metrics(last_chunk)))
        else:
            thinking = (response["message"].get("thinking") or "")
            content = (response["message"].get("content") or "")
            if thinking:
                yield ("think", thinking)
            yield ("content", content)
            yield ("metrics", json.dumps(_extract_metrics(response)))

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def _client(self, config: "Config" | None = None):
        ollama = _require_ollama()
        host = (config.ollama_host if config is not None else None) or "http://localhost:11434"
        return ollama.Client(host=host)

    def embed(
        self,
        text: str,
        *,
        model: str,
        config: "Config | None" = None,
    ) -> np.ndarray:
        if not text:
            return np.zeros(0, dtype=np.float32)
        ollama = _require_ollama()
        client = ollama.Client()  # uses default OLLAMA_HOST env var fallback
        # Ollama's embed-side keep_alive is hard-coded to a sensible
        # 30m here; ``config.embed_keep_alive`` is honoured by the
        # LM Studio backend but Ollama's tagged models don't benefit
        # from a shorter TTL the same way (they're already small).
        keep_alive = (
            getattr(config, "embed_keep_alive", None) if config else None
        ) or "30m"
        response = client.embed(model=model, input=text, keep_alive=keep_alive)
        vectors = response.get("embeddings") or []
        if not vectors:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(vectors[0], dtype=np.float32)

    def embed_batch(
        self,
        texts: list[str],
        *,
        model: str,
        config: "Config | None" = None,
    ) -> list[np.ndarray]:
        if not texts:
            return []
        ollama = _require_ollama()
        client = ollama.Client()
        keep_alive = (
            getattr(config, "embed_keep_alive", None) if config else None
        ) or "30m"
        try:
            response = client.embed(model=model, input=texts, keep_alive=keep_alive)
            vectors = response.get("embeddings") or []
            return [np.asarray(v, dtype=np.float32) for v in vectors]
        except Exception as exc:
            # On context-length errors only, retry per-item so a single
            # oversized chunk doesn't poison the whole batch.
            msg = str(exc).lower()
            if "exceeds" not in msg and "context" not in msg:
                raise
            results: list[np.ndarray] = []
            for text in texts:
                try:
                    single = client.embed(
                        model=model, input=text, keep_alive=keep_alive
                    )
                    vecs = single.get("embeddings") or []
                    results.append(
                        np.asarray(vecs[0], dtype=np.float32)
                        if vecs
                        else np.zeros(0, dtype=np.float32)
                    )
                except Exception:
                    results.append(np.zeros(0, dtype=np.float32))
            return results

    # ------------------------------------------------------------------
    # Warm-up + probes
    # ------------------------------------------------------------------

    def warm_chat(self, config: "Config") -> None:
        try:
            ollama = _require_ollama()
            ollama.Client(host=config.ollama_host).chat(
                model=config.model,
                messages=[{"role": "user", "content": "_"}],
                keep_alive=config.ollama_keep_alive,
                options={"num_predict": 1, "temperature": 0.0},
            )
        except Exception:
            # Fire-and-forget: never raise into the foreground.
            pass

    def warm_embed(self, config: "Config") -> None:
        try:
            ollama = _require_ollama()
            ollama.Client(host=config.ollama_host).embed(
                model=config.embedding_model,
                input=" ",
            )
        except Exception:
            pass

    def is_embedding_available(self, model: str) -> bool:
        try:
            self.embed("probe", model=model)
            return True
        except Exception:
            return False
