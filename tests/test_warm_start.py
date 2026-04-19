"""Tests for item #4 — warm-start Ollama from the top-level callback.

The three contracts we pin here:

1. When ``warm_start=True`` and ``in_test_mode=False``, exactly two
   daemon threads are spawned (one chat, one embed), with the expected
   target functions.
2. When ``in_test_mode=True``, zero threads are spawned — even if
   ``warm_start`` is True. This is the guard the rest of the test suite
   relies on via the autouse conftest fixture.
3. ``_warm_ollama`` / ``_warm_embedder`` swallow every exception raised
   by the ``ollama`` client, so a missing or misconfigured server never
   surfaces as a user-visible traceback.

The session-wide conftest patches ``_spawn_warm_start`` to a no-op so
existing tests stay hermetic. We capture the *real* function at module
import time — before the autouse fixture runs — and restore it for the
tests that need to exercise the spawn logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List
from unittest.mock import MagicMock

import pytest

import gemma.main as main_module
from gemma.config import Config


# The conftest autouse fixture replaces ``_spawn_warm_start`` at the start
# of each test. To observe the real implementation we grab a reference
# here at module import time, which happens *before* any fixture fires.
_ORIGINAL_SPAWN = main_module._spawn_warm_start


@pytest.fixture
def real_spawn(monkeypatch):
    """Restore the production ``_spawn_warm_start`` for one test."""
    monkeypatch.setattr(main_module, "_spawn_warm_start", _ORIGINAL_SPAWN)


# ---------------------------------------------------------------------------
# Thread spy — captures every ``threading.Thread(...)`` kwargs so assertions
# can inspect target/daemon/name without the threads actually running.
# ---------------------------------------------------------------------------

@dataclass
class _ThreadSpy:
    calls: List[dict] = field(default_factory=list)

    def __call__(
        self,
        *,
        target: Callable,
        args: tuple = (),
        daemon: bool = False,
        name: str = "",
        **_kw: Any,
    ) -> "_ThreadSpy":
        self.calls.append(
            {"target": target, "args": args, "daemon": daemon, "name": name}
        )
        return self

    def start(self) -> None:  # no-op so the warm-up body never runs
        pass


# ---------------------------------------------------------------------------
# Contract 1: spawn count + daemon flag
# ---------------------------------------------------------------------------

def test_warm_start_spawns_two_daemon_threads(monkeypatch, real_spawn):
    spy = _ThreadSpy()
    monkeypatch.setattr("gemma.main.threading.Thread", spy)

    cfg = Config(warm_start=True, in_test_mode=False)
    main_module._spawn_warm_start(cfg)

    assert len(spy.calls) == 2, f"expected 2 spawns, got {len(spy.calls)}"
    # Both must be daemons so process exit doesn't hang on warm-up.
    assert all(call["daemon"] is True for call in spy.calls)
    # And they should be the two distinct helpers — not the same one twice.
    targets = {call["target"] for call in spy.calls}
    assert main_module._warm_ollama in targets
    assert main_module._warm_embedder in targets


# ---------------------------------------------------------------------------
# Contract 2: ``in_test_mode`` + ``warm_start=False`` both gate off the spawn
# ---------------------------------------------------------------------------

def test_warm_start_skipped_under_in_test_mode(monkeypatch, real_spawn):
    spy = _ThreadSpy()
    monkeypatch.setattr("gemma.main.threading.Thread", spy)

    cfg = Config(warm_start=True, in_test_mode=True)
    main_module._spawn_warm_start(cfg)

    assert spy.calls == []


def test_warm_start_skipped_when_disabled(monkeypatch, real_spawn):
    spy = _ThreadSpy()
    monkeypatch.setattr("gemma.main.threading.Thread", spy)

    cfg = Config(warm_start=False, in_test_mode=False)
    main_module._spawn_warm_start(cfg)

    assert spy.calls == []


# ---------------------------------------------------------------------------
# Contract 3: helpers swallow every exception
# ---------------------------------------------------------------------------

def test_warm_ollama_swallows_connection_errors(monkeypatch):
    """Any error from ``ollama.Client.chat`` must stay inside the helper."""
    import ollama

    fake_client = MagicMock()
    fake_client.chat.side_effect = ConnectionRefusedError("no ollama here")
    monkeypatch.setattr(ollama, "Client", lambda host=None: fake_client)

    cfg = Config()
    main_module._warm_ollama(cfg)  # must not raise
    fake_client.chat.assert_called_once()


def test_warm_embedder_swallows_client_construction_failure(monkeypatch):
    """Even if ``ollama.Client(...)`` itself raises, the helper returns cleanly."""
    import ollama

    def _boom(*_a, **_kw):
        raise RuntimeError("ollama client exploded")

    monkeypatch.setattr(ollama, "Client", _boom)

    cfg = Config()
    main_module._warm_embedder(cfg)  # must not raise
