"""LM Studio implementation of :class:`gemma.backends.base.LLMBackend`.

Uses the native ``lmstudio`` Python SDK (not the OpenAI-compat REST
shim) so we get first-class access to:

  * **JIT model loading** — ``lmstudio.llm(model_key, ttl=...)`` loads
    a model on demand and unloads it after ``ttl`` seconds of idleness.
    Maps cleanly onto the existing ``ollama_keep_alive`` semantics.
  * **Structured reasoning fragments** — every streaming fragment
    carries a ``reasoning_type`` literal (``none`` / ``reasoning`` /
    ``reasoningStartTag`` / ``reasoningEndTag``) so we can route think
    vs. content tokens to the right channel without parsing
    ``<think>`` tags ourselves.
  * **Embedding models** — ``lmstudio.embedding_model(...)`` is the
    embedding-side counterpart with the same JIT/TTL behaviour.

Token counts come back on ``PredictionResult.stats``; we forward them as
``prompt_eval_count`` / ``eval_count`` to keep the metrics tuple shape
compatible with the Ollama backend.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Generator

import numpy as np

from gemma.backends.base import (
    ChatChunk,
    LLMBackend,
    parse_keep_alive_seconds,
)

if TYPE_CHECKING:
    from gemma.config import Config


# Module-level import so test patches can target
# ``gemma.backends.lmstudio_backend.lmstudio.llm`` directly. We catch
# ImportError in case a future user strips the ``lmstudio`` dependency
# in favour of an Ollama-only install — the backend still imports
# cleanly; ``_require_lmstudio()`` raises with a clear hint at call
# time.
try:
    import lmstudio  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover — exercised only when dep missing
    lmstudio = None  # type: ignore[assignment]


def _require_lmstudio():
    """Return the imported ``lmstudio`` module or raise a friendly error."""
    if lmstudio is None:
        raise ImportError(
            "The LM Studio backend requires the 'lmstudio' package. "
            "Install with: uv sync  (or 'pip install lmstudio>=1.3')."
        )
    return lmstudio


def _build_chat(messages: list[dict]):
    """Translate ``[{role, content}, ...]`` into an ``lmstudio.Chat``.

    The SDK's ``Chat`` object is the canonical history container —
    ``respond`` / ``respond_stream`` accept it directly. System prompts,
    user messages, and assistant turns each have their own appender.
    Unknown roles (``tool``, ``function``) fall back to user messages
    with a clear prefix so they remain visible to the model rather than
    silently dropped.
    """
    lmstudio = _require_lmstudio()
    chat = lmstudio.Chat()
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "") or ""
        if not content:
            continue
        if role == "system":
            chat.add_system_prompt(content)
        elif role == "assistant":
            chat.add_assistant_response(content)
        elif role == "user":
            chat.add_user_message(content)
        else:
            # Tool/function results — preserve content visibility.
            chat.add_user_message(f"[{role}] {content}")
    return chat


def _stats_to_metrics(stats) -> dict:
    """Map ``LlmPredictionStats`` to the ``prompt/eval`` metric shape."""
    if stats is None:
        return {"prompt_eval_count": 0, "eval_count": 0}
    return {
        "prompt_eval_count": int(getattr(stats, "prompt_tokens_count", 0) or 0),
        "eval_count": int(getattr(stats, "predicted_tokens_count", 0) or 0),
    }


class LMStudioBackend(LLMBackend):
    """Backend that talks to LM Studio via its native Python SDK.

    Stateless: every call resolves the model handle through
    ``lmstudio.llm(...)`` / ``lmstudio.embedding_model(...)``. The SDK
    caches the underlying connection so this is cheap.
    """

    name = "lmstudio"

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
        lmstudio = _require_lmstudio()

        ttl = parse_keep_alive_seconds(config.ollama_keep_alive)
        # Resolve the chat model handle. ``ttl`` controls auto-unload; we
        # forward the unified keep-alive value so a single ``--keep-alive``
        # flag still works regardless of backend.
        model = lmstudio.llm(config.model, ttl=ttl)

        chat_history = _build_chat(messages)
        prediction_config = {"temperature": float(config.temperature)}

        if stream:
            stream_handle = model.respond_stream(
                chat_history, config=prediction_config
            )
            yield from self._stream_fragments(stream_handle, config)
        else:
            result = model.respond(chat_history, config=prediction_config)
            yield from self._yield_complete(result, config)

    @staticmethod
    def _stream_fragments(stream_handle, config: "Config"):
        """Iterate over streaming fragments, emitting ``ChatChunk`` tuples.

        ``reasoning_type`` is the routing key:

          * ``"reasoning"`` → ``("think", text)`` (only when
            ``config.thinking_mode`` is True; otherwise suppressed so
            ``--no-think`` users don't see reasoning leak through)
          * ``"none"``      → ``("content", text)``
          * ``"reasoningStartTag"`` / ``"reasoningEndTag"`` → swallowed.
            They are channel markers, not user-visible content.
        """
        show_thinking = bool(getattr(config, "thinking_mode", False))
        for fragment in stream_handle:
            text = fragment.content or ""
            kind = fragment.reasoning_type
            if not text:
                continue
            if kind == "reasoning":
                if show_thinking:
                    yield ("think", text)
            elif kind == "none":
                yield ("content", text)
            # reasoningStartTag / reasoningEndTag are intentionally dropped.

        # Final stats live on the stream handle's terminal result.
        result = getattr(stream_handle, "result", None)
        if callable(result):
            try:
                result = result()
            except Exception:
                result = None
        stats = getattr(result, "stats", None) if result is not None else None
        yield ("metrics", json.dumps(_stats_to_metrics(stats)))

    @staticmethod
    def _yield_complete(result, config: "Config"):
        """Non-streaming path: emit content + metrics in one go.

        The SDK does not expose a separate reasoning channel on
        ``PredictionResult.content`` — the field is post-strip. If a
        future SDK release surfaces ``reasoning_content`` we route it
        here too; for now non-streaming callers see content only.
        """
        text = getattr(result, "content", "") or ""
        reasoning = getattr(result, "reasoning_content", "") or ""
        if reasoning and getattr(config, "thinking_mode", False):
            yield ("think", reasoning)
        yield ("content", text)
        yield ("metrics", json.dumps(_stats_to_metrics(getattr(result, "stats", None))))

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(self, text: str, *, model: str) -> np.ndarray:
        if not text:
            return np.zeros(0, dtype=np.float32)
        lmstudio = _require_lmstudio()
        handle = lmstudio.embedding_model(model)
        vector = handle.embed(text)
        return np.asarray(vector, dtype=np.float32)

    def embed_batch(self, texts: list[str], *, model: str) -> list[np.ndarray]:
        if not texts:
            return []
        lmstudio = _require_lmstudio()
        handle = lmstudio.embedding_model(model)
        try:
            vectors = handle.embed(texts)
            return [np.asarray(v, dtype=np.float32) for v in vectors]
        except Exception as exc:
            # Mirror Ollama's behaviour: on context-length errors fall
            # back to per-item embedding so one oversized chunk doesn't
            # poison the whole batch. Other errors propagate.
            msg = str(exc).lower()
            if "context" not in msg and "exceed" not in msg and "length" not in msg:
                raise
            results: list[np.ndarray] = []
            for text in texts:
                try:
                    vec = handle.embed(text)
                    results.append(np.asarray(vec, dtype=np.float32))
                except Exception:
                    results.append(np.zeros(0, dtype=np.float32))
            return results

    # ------------------------------------------------------------------
    # Warm-up + probes
    # ------------------------------------------------------------------

    def warm_chat(self, config: "Config") -> None:
        try:
            lmstudio = _require_lmstudio()
            ttl = parse_keep_alive_seconds(config.ollama_keep_alive)
            model = lmstudio.llm(config.model, ttl=ttl)
            # 1-token probe — same trick as Ollama. The SDK has no
            # ``num_predict`` shortcut so we cap via prediction config.
            list(
                model.respond_stream(
                    "_",
                    config={"temperature": 0.0, "maxTokens": 1},
                )
            )
        except Exception:
            # Fire-and-forget; never raise into the foreground.
            pass

    def warm_embed(self, config: "Config") -> None:
        try:
            lmstudio = _require_lmstudio()
            handle = lmstudio.embedding_model(config.embedding_model)
            handle.embed(" ")
        except Exception:
            pass

    def is_embedding_available(self, model: str) -> bool:
        try:
            self.embed("probe", model=model)
            return True
        except Exception:
            return False
