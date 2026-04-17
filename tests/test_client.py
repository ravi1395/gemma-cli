"""Unit tests for the Ollama client wrapper.

Mocks `ollama.Client.chat` so tests run without a running Ollama server.
"""

from unittest.mock import MagicMock, patch

import pytest

from gemma.client import ask, chat
from gemma.config import Config


@pytest.fixture
def config() -> Config:
    return Config()


def _fake_stream(texts: list[str]):
    """Yield chunks in the shape Ollama returns when stream=True (no thinking)."""
    for t in texts:
        yield {"message": {"content": t, "thinking": ""}}


def _fake_stream_with_thinking(thinking: str, texts: list[str]):
    """Yield a thinking chunk followed by content chunks."""
    yield {"message": {"content": "", "thinking": thinking}}
    for t in texts:
        yield {"message": {"content": t, "thinking": ""}}


def _fake_blocking(text: str) -> dict:
    """Return the dict shape Ollama returns when stream=False (no thinking)."""
    return {"message": {"content": text, "thinking": ""}}


def _fake_blocking_with_thinking(thinking: str, text: str) -> dict:
    """Return the dict shape Ollama returns when stream=False with thinking."""
    return {"message": {"content": text, "thinking": thinking}}


@patch("gemma.client.ollama.Client")
def test_ask_streams_tokens(mock_client_cls, config):
    mock_inst = MagicMock()
    mock_inst.chat.return_value = _fake_stream(["Hello ", "world"])
    mock_client_cls.return_value = mock_inst

    result = "".join(text for _, text in ask("hi", config, stream=True))

    assert result == "Hello world"
    mock_inst.chat.assert_called_once()
    call_kwargs = mock_inst.chat.call_args.kwargs
    assert call_kwargs["model"] == config.model
    assert call_kwargs["stream"] is True
    assert call_kwargs["messages"][0]["role"] == "system"
    assert call_kwargs["messages"][1]["role"] == "user"
    assert call_kwargs["messages"][1]["content"] == "hi"


@patch("gemma.client.ollama.Client")
def test_ask_blocking_returns_full_text(mock_client_cls, config):
    mock_inst = MagicMock()
    mock_inst.chat.return_value = _fake_blocking("done")
    mock_client_cls.return_value = mock_inst

    result = "".join(text for _, text in ask("hi", config, stream=False))

    assert result == "done"
    assert mock_inst.chat.call_args.kwargs["stream"] is False


@patch("gemma.client.ollama.Client")
def test_ask_respects_custom_system_prompt(mock_client_cls, config):
    mock_inst = MagicMock()
    mock_inst.chat.return_value = _fake_stream(["ok"])
    mock_client_cls.return_value = mock_inst

    list(ask("hi", config, system_prompt="You are a pirate.", stream=True))

    messages = mock_inst.chat.call_args.kwargs["messages"]
    assert messages[0]["content"] == "You are a pirate."


@patch("gemma.client.ollama.Client")
def test_chat_passes_messages_through(mock_client_cls, config):
    mock_inst = MagicMock()
    mock_inst.chat.return_value = _fake_stream(["reply"])
    mock_client_cls.return_value = mock_inst

    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    result = "".join(text for _, text in chat(msgs, config, stream=True))

    assert result == "reply"
    assert mock_inst.chat.call_args.kwargs["messages"] == msgs


@patch("gemma.client.ollama.Client")
def test_thinking_mode_streaming(mock_client_cls, config):
    """Thinking chunks are yielded as ("think", ...) before ("content", ...) tuples."""
    config.thinking_mode = True
    mock_inst = MagicMock()
    mock_inst.chat.return_value = _fake_stream_with_thinking("let me reason", ["answer"])
    mock_client_cls.return_value = mock_inst

    tuples = list(ask("hi", config, stream=True))

    types = [t for t, _ in tuples]
    texts = [v for _, v in tuples]
    assert "think" in types
    assert "content" in types
    assert "".join(v for t, v in tuples if t == "think") == "let me reason"
    assert "".join(v for t, v in tuples if t == "content") == "answer"
    assert mock_inst.chat.call_args.kwargs["think"] is True


@patch("gemma.client.ollama.Client")
def test_thinking_mode_blocking(mock_client_cls, config):
    """Non-streaming thinking response yields think tuple then content tuple."""
    config.thinking_mode = True
    mock_inst = MagicMock()
    mock_inst.chat.return_value = _fake_blocking_with_thinking("reasoning here", "final answer")
    mock_client_cls.return_value = mock_inst

    tuples = list(ask("hi", config, stream=False))

    assert tuples[0] == ("think", "reasoning here")
    assert tuples[1] == ("content", "final answer")


@patch("gemma.client.ollama.Client")
def test_thinking_mode_off_by_default(mock_client_cls, config):
    """think=False is passed to Ollama when thinking_mode is off."""
    mock_inst = MagicMock()
    mock_inst.chat.return_value = _fake_stream(["hi"])
    mock_client_cls.return_value = mock_inst

    list(ask("hello", config, stream=True))

    assert mock_inst.chat.call_args.kwargs["think"] is False


@patch("gemma.client.ollama.Client")
def test_keep_alive_propagated(mock_client_cls, config):
    """The configured ollama_keep_alive is forwarded to ollama.Client.chat."""
    config.ollama_keep_alive = "2h"
    mock_inst = MagicMock()
    mock_inst.chat.return_value = _fake_stream(["ok"])
    mock_client_cls.return_value = mock_inst

    list(ask("hi", config, stream=True))

    assert mock_inst.chat.call_args.kwargs["keep_alive"] == "2h"
