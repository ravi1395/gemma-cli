"""Unit tests for :class:`gemma.backends.lmstudio_backend.LMStudioBackend`.

The lmstudio SDK is mocked with ``unittest.mock`` so these tests never
touch a real LM Studio process. Coverage focuses on:

* The (kind, text) tuple contract emitted by ``chat()`` for both
  streaming and non-streaming, including the ``reasoning_type``
  routing that replaces Ollama's separate ``thinking`` field.
* ``thinking_mode = False`` suppresses the reasoning channel even when
  the SDK emits it (otherwise reasoning would leak into the user-
  visible content stream).
* Embedding round-trips return ``np.float32`` arrays.
* Keep-alive ↔ TTL mapping is forwarded to ``lmstudio.llm(ttl=...)``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gemma.backends.lmstudio_backend import LMStudioBackend
from gemma.config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fragment(content: str, reasoning_type: str = "none"):
    """Build a fake LlmPredictionFragment using SimpleNamespace.

    Real fragments are msgspec structs; SimpleNamespace satisfies the
    attribute-access protocol the backend uses (``.content`` /
    ``.reasoning_type``) without dragging the SDK into the test.
    """
    return SimpleNamespace(content=content, reasoning_type=reasoning_type)


def _stats(prompt: int, predicted: int):
    return SimpleNamespace(prompt_tokens_count=prompt, predicted_tokens_count=predicted)


def _stream(fragments, stats=None):
    """Iterable that also exposes a ``.result()`` carrying terminal stats.

    Mirrors how ``lmstudio.PredictionStream`` behaves: iterate to consume
    fragments, then call ``.result()`` to retrieve the final stats blob.
    """
    class _Stream:
        def __iter__(self):
            return iter(fragments)

        def result(self):
            return SimpleNamespace(stats=stats or _stats(0, 0), content="")

    return _Stream()


@pytest.fixture
def cfg() -> Config:
    return Config(backend="lmstudio")


# ---------------------------------------------------------------------------
# chat() — streaming
# ---------------------------------------------------------------------------


def test_streaming_emits_content_and_metrics(cfg):
    """Plain content fragments produce ('content', text) plus terminal metrics."""
    fake_model = MagicMock()
    fake_model.respond_stream.return_value = _stream(
        [_fragment("Hello "), _fragment("world")],
        stats=_stats(10, 5),
    )
    with patch(
        "gemma.backends.lmstudio_backend.lmstudio.llm",
        return_value=fake_model,
    ):
        be = LMStudioBackend()
        tuples = list(be.chat([{"role": "user", "content": "hi"}], cfg))

    kinds = [t for t, _ in tuples]
    assert kinds == ["content", "content", "metrics"]
    assert "".join(v for t, v in tuples if t == "content") == "Hello world"
    metrics = json.loads(tuples[-1][1])
    assert metrics == {"prompt_eval_count": 10, "eval_count": 5}


def test_streaming_routes_reasoning_to_think_when_enabled(cfg):
    """``reasoning_type='reasoning'`` fragments become ('think', text) tuples."""
    cfg.thinking_mode = True
    fake_model = MagicMock()
    fake_model.respond_stream.return_value = _stream(
        [
            _fragment("let me think", reasoning_type="reasoning"),
            _fragment("answer"),
        ],
        stats=_stats(3, 2),
    )
    with patch(
        "gemma.backends.lmstudio_backend.lmstudio.llm",
        return_value=fake_model,
    ):
        be = LMStudioBackend()
        tuples = list(be.chat([{"role": "user", "content": "hi"}], cfg))

    assert ("think", "let me think") in tuples
    assert ("content", "answer") in tuples


def test_streaming_suppresses_reasoning_when_thinking_mode_off(cfg):
    """With thinking_mode=False, reasoning fragments are dropped (not leaked)."""
    cfg.thinking_mode = False
    fake_model = MagicMock()
    fake_model.respond_stream.return_value = _stream(
        [
            _fragment("internal monologue", reasoning_type="reasoning"),
            _fragment("final answer"),
        ],
        stats=_stats(0, 0),
    )
    with patch(
        "gemma.backends.lmstudio_backend.lmstudio.llm",
        return_value=fake_model,
    ):
        be = LMStudioBackend()
        tuples = list(be.chat([{"role": "user", "content": "hi"}], cfg))

    kinds = [t for t, _ in tuples]
    assert "think" not in kinds
    assert "".join(v for t, v in tuples if t == "content") == "final answer"


def test_streaming_drops_reasoning_tag_markers(cfg):
    """``reasoningStartTag`` / ``reasoningEndTag`` are channel markers, not content."""
    cfg.thinking_mode = True
    fake_model = MagicMock()
    fake_model.respond_stream.return_value = _stream(
        [
            _fragment("<think>", reasoning_type="reasoningStartTag"),
            _fragment("hidden", reasoning_type="reasoning"),
            _fragment("</think>", reasoning_type="reasoningEndTag"),
            _fragment("visible"),
        ],
        stats=_stats(0, 0),
    )
    with patch(
        "gemma.backends.lmstudio_backend.lmstudio.llm",
        return_value=fake_model,
    ):
        be = LMStudioBackend()
        tuples = list(be.chat([{"role": "user", "content": "hi"}], cfg))

    contents = [v for t, v in tuples if t == "content"]
    assert contents == ["visible"]
    assert ("think", "hidden") in tuples


# ---------------------------------------------------------------------------
# chat() — non-streaming + ttl mapping
# ---------------------------------------------------------------------------


def test_non_streaming_returns_content_and_metrics(cfg):
    fake_model = MagicMock()
    fake_model.respond.return_value = SimpleNamespace(
        content="answer",
        stats=_stats(7, 3),
    )
    with patch(
        "gemma.backends.lmstudio_backend.lmstudio.llm",
        return_value=fake_model,
    ):
        be = LMStudioBackend()
        tuples = list(
            be.chat(
                [{"role": "user", "content": "hi"}],
                cfg,
                stream=False,
            )
        )

    assert ("content", "answer") in tuples
    metrics = json.loads([v for t, v in tuples if t == "metrics"][0])
    assert metrics == {"prompt_eval_count": 7, "eval_count": 3}


def test_keep_alive_string_maps_to_ttl_seconds(cfg):
    """``ollama_keep_alive='30m'`` becomes ``lmstudio.llm(ttl=1800)``."""
    cfg.ollama_keep_alive = "30m"
    fake_model = MagicMock()
    fake_model.respond_stream.return_value = _stream([_fragment("ok")], stats=_stats(0, 0))
    with patch(
        "gemma.backends.lmstudio_backend.lmstudio.llm",
        return_value=fake_model,
    ) as mock_llm:
        be = LMStudioBackend()
        list(be.chat([{"role": "user", "content": "hi"}], cfg))

    mock_llm.assert_called_once()
    assert mock_llm.call_args.kwargs["ttl"] == 1800


def test_keep_alive_forever_maps_to_none_ttl(cfg):
    """``-1`` (Ollama "never evict") maps to LM Studio's ``ttl=None``."""
    cfg.ollama_keep_alive = "-1"
    fake_model = MagicMock()
    fake_model.respond_stream.return_value = _stream([_fragment("ok")], stats=_stats(0, 0))
    with patch(
        "gemma.backends.lmstudio_backend.lmstudio.llm",
        return_value=fake_model,
    ) as mock_llm:
        be = LMStudioBackend()
        list(be.chat([{"role": "user", "content": "hi"}], cfg))

    assert mock_llm.call_args.kwargs["ttl"] is None


# ---------------------------------------------------------------------------
# embed() / embed_batch()
# ---------------------------------------------------------------------------


def test_embed_returns_float32_array():
    fake_handle = MagicMock()
    fake_handle.embed.return_value = [0.1, 0.2, 0.3]
    with patch(
        "gemma.backends.lmstudio_backend.lmstudio.embedding_model",
        return_value=fake_handle,
    ):
        be = LMStudioBackend()
        vec = be.embed("hello", model="nomic-ai/nomic-embed-text-v1.5")

    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float32
    assert vec.tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_embed_empty_returns_zero_length():
    be = LMStudioBackend()
    vec = be.embed("", model="nomic-ai/nomic-embed-text-v1.5")
    assert vec.shape == (0,)


def test_embed_batch_falls_back_per_item_on_context_overflow():
    """A context-length error on the batch call must not lose other items."""
    fake_handle = MagicMock()
    # Batch call raises; per-item calls succeed except for the second.
    call_log = {"batch": 0, "items": []}

    def _embed(input_):
        if isinstance(input_, list):
            call_log["batch"] += 1
            raise RuntimeError("input length exceeds context length")
        call_log["items"].append(input_)
        if input_ == "too long":
            raise RuntimeError("exceeds context length")
        return [0.5, 0.5]

    fake_handle.embed.side_effect = _embed

    with patch(
        "gemma.backends.lmstudio_backend.lmstudio.embedding_model",
        return_value=fake_handle,
    ):
        be = LMStudioBackend()
        vecs = be.embed_batch(
            ["fine", "too long", "also fine"],
            model="nomic-ai/nomic-embed-text-v1.5",
        )

    assert call_log["batch"] == 1
    assert len(vecs) == 3
    assert vecs[0].shape == (2,)
    assert vecs[1].shape == (0,)  # zero-length sentinel for the failed item
    assert vecs[2].shape == (2,)


# ---------------------------------------------------------------------------
# warm helpers
# ---------------------------------------------------------------------------


def test_warm_chat_swallows_exceptions():
    """Even if every SDK call raises, warm_chat must return cleanly."""
    cfg = Config(backend="lmstudio")
    with patch(
        "gemma.backends.lmstudio_backend.lmstudio.llm",
        side_effect=RuntimeError("server down"),
    ):
        LMStudioBackend().warm_chat(cfg)  # must not raise


def test_warm_embed_swallows_exceptions():
    cfg = Config(backend="lmstudio")
    with patch(
        "gemma.backends.lmstudio_backend.lmstudio.embedding_model",
        side_effect=RuntimeError("server down"),
    ):
        LMStudioBackend().warm_embed(cfg)  # must not raise


# ---------------------------------------------------------------------------
# embed_keep_alive → ttl plumbing
# ---------------------------------------------------------------------------

def test_embed_uses_embed_keep_alive_ttl():
    """``Config.embed_keep_alive='30s'`` → ``embedding_model(ttl=30)``."""
    fake_handle = MagicMock()
    fake_handle.embed.return_value = [0.1, 0.2]
    cfg = Config(backend="lmstudio", embed_keep_alive="30s")

    with patch(
        "gemma.backends.lmstudio_backend.lmstudio.embedding_model",
        return_value=fake_handle,
    ) as mock_em:
        be = LMStudioBackend()
        be.embed("hello", model="some/embed", config=cfg)

    mock_em.assert_called_once()
    assert mock_em.call_args.kwargs["ttl"] == 30


def test_embed_batch_uses_embed_keep_alive_ttl():
    fake_handle = MagicMock()
    fake_handle.embed.return_value = [[0.1], [0.2]]
    cfg = Config(backend="lmstudio", embed_keep_alive="2m")

    with patch(
        "gemma.backends.lmstudio_backend.lmstudio.embedding_model",
        return_value=fake_handle,
    ) as mock_em:
        be = LMStudioBackend()
        be.embed_batch(["a", "b"], model="some/embed", config=cfg)

    assert mock_em.call_args.kwargs["ttl"] == 120


def test_warm_embed_uses_embed_keep_alive_ttl():
    fake_handle = MagicMock()
    cfg = Config(backend="lmstudio", embed_keep_alive="45s")

    with patch(
        "gemma.backends.lmstudio_backend.lmstudio.embedding_model",
        return_value=fake_handle,
    ) as mock_em:
        LMStudioBackend().warm_embed(cfg)

    mock_em.assert_called_once()
    assert mock_em.call_args.kwargs["ttl"] == 45


def test_embed_without_config_falls_back_to_default_ttl():
    """Legacy callers (no config kwarg) still get a sensible short TTL."""
    fake_handle = MagicMock()
    fake_handle.embed.return_value = [0.1]

    with patch(
        "gemma.backends.lmstudio_backend.lmstudio.embedding_model",
        return_value=fake_handle,
    ) as mock_em:
        LMStudioBackend().embed("hi", model="some/embed")

    # 30s is the documented default in ``_embed_ttl`` so a stub call
    # without config still gets the same short-lived handle real code does.
    assert mock_em.call_args.kwargs["ttl"] == 30
