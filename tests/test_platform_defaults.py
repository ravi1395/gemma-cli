"""Unit tests for the platform-aware default model resolution.

Apple Silicon → MLX builds; everything else → GGUF Q4. The functions
read ``platform.machine()`` and our internal ``detect_os()`` so we
monkeypatch both to drive every branch deterministically.
"""

from __future__ import annotations

import platform as _stdlib_platform

import pytest

from gemma import platform as gp
from gemma.config import Config


def _force_apple_silicon(monkeypatch):
    monkeypatch.setattr(gp, "detect_os", lambda: gp.OS.MACOS)
    monkeypatch.setattr(_stdlib_platform, "machine", lambda: "arm64")


def _force_macos_intel(monkeypatch):
    monkeypatch.setattr(gp, "detect_os", lambda: gp.OS.MACOS)
    monkeypatch.setattr(_stdlib_platform, "machine", lambda: "x86_64")


def _force_linux(monkeypatch):
    monkeypatch.setattr(gp, "detect_os", lambda: gp.OS.LINUX)
    monkeypatch.setattr(_stdlib_platform, "machine", lambda: "x86_64")


def test_apple_silicon_returns_mlx_default(monkeypatch):
    _force_apple_silicon(monkeypatch)
    assert gp.is_apple_silicon() is True
    assert gp.default_chat_model() == "mlx-community/gemma-4-E4B-it-4bit"
    assert gp.default_embedding_model() == "nomic-ai/nomic-embed-text-v1.5"


def test_intel_mac_falls_back_to_gguf(monkeypatch):
    """An x86_64 Python under Rosetta is *not* Apple Silicon — MLX won't load."""
    _force_macos_intel(monkeypatch)
    assert gp.is_apple_silicon() is False
    assert gp.default_chat_model() == "lmstudio-community/gemma-3-4B-it-GGUF"


def test_linux_x86_returns_gguf_default(monkeypatch):
    _force_linux(monkeypatch)
    assert gp.is_apple_silicon() is False
    assert gp.default_chat_model() == "lmstudio-community/gemma-3-4B-it-GGUF"
    assert gp.default_embedding_model() == "nomic-ai/nomic-embed-text-v1.5-GGUF"


def test_config_post_init_resolves_model_field(monkeypatch):
    """``Config(model=None)`` populates ``model`` via the platform default."""
    _force_apple_silicon(monkeypatch)
    cfg = Config()
    assert cfg.model == "mlx-community/gemma-4-E4B-it-4bit"


def test_explicit_model_wins_over_default(monkeypatch):
    """User/profile-specified model overrides the platform default."""
    _force_apple_silicon(monkeypatch)
    cfg = Config(model="lmstudio-community/Llama-3.2-3B-Instruct-GGUF")
    assert cfg.model == "lmstudio-community/Llama-3.2-3B-Instruct-GGUF"


def test_empty_string_model_resolves_to_default(monkeypatch):
    """An empty string opts back into the auto-default (profile pattern)."""
    _force_linux(monkeypatch)
    cfg = Config(model="")
    assert cfg.model == "lmstudio-community/gemma-3-4B-it-GGUF"
