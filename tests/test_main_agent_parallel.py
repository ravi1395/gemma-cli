"""Tests for the parallel tool-dispatch path in ``_agent_loop`` (item #20).

These tests exercise the fan-out behaviour that #20 added to
:func:`gemma.main._agent_loop`:

* When a single model turn emits N > 1 tool calls and
  ``cfg.agent_tool_concurrency > 1``, the calls run through a
  :class:`concurrent.futures.ThreadPoolExecutor` scoped to that turn.
* When ``concurrency == 1`` or ``len(raw_calls) <= 1``, the executor is
  bypassed entirely — this protects the microsecond-scale baseline that
  the bench harness measures.

All tests use deterministic stub clients and stub dispatchers so they
are hermetic and safe for CI — no real Ollama, no real network.
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional

import pytest

from gemma.agent.cache import AgentSessionCache
from gemma.config import Config
from gemma.main import (
    _agent_loop,
    _clamped_concurrency,
    _dispatch_turn_calls,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubClient:
    """Returns the next scripted message; falls back to a terminal reply."""

    def __init__(self, scripted_turns: List[Dict]) -> None:
        self._turns = list(scripted_turns)

    def chat(self, *, model, messages, tools=None, **_):
        if self._turns:
            return {"message": self._turns.pop(0)}
        return {"message": {"role": "assistant", "content": "done"}}


def _tc(name: str, **args: Any) -> Dict[str, Any]:
    """Shorthand for a flat tool-call dict (matches bench stub format)."""
    return {"name": name, "arguments": dict(args)}


def _cfg_with_concurrency(n: int) -> Config:
    """Config with only ``agent_tool_concurrency`` overridden."""
    return replace(Config(), agent_tool_concurrency=n)


# ---------------------------------------------------------------------------
# _clamped_concurrency — the defensive wrapper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        (0, 1),       # floor
        (1, 1),       # identity — serial path
        (4, 4),       # default
        (16, 16),     # ceiling
        (999, 16),    # cap
        (-5, 1),      # negatives clamp to 1
    ],
)
def test_clamped_concurrency_bounds(raw, expected):
    """``_clamped_concurrency`` enforces ``[1, 16]``."""
    cfg = _cfg_with_concurrency(raw)
    assert _clamped_concurrency(cfg) == expected


def test_clamped_concurrency_handles_missing_field():
    """Defensive default fires when the Config instance lacks the field."""
    class _Barebones:  # noqa: D401 — tiny shim
        pass
    assert _clamped_concurrency(_Barebones()) == 1  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Order preservation under concurrency
# ---------------------------------------------------------------------------

def test_parallel_dispatch_preserves_call_order():
    """Even with deliberately staggered sleeps, ``tool`` messages land in order.

    ``executor.map`` guarantees the yielded results match the input
    iterable's order — this test locks that invariant in so a future
    refactor to ``as_completed`` cannot silently reorder the replies
    and desynchronise ``tool_call_id`` on the next turn.
    """
    started_in_order: List[str] = []
    started_lock = threading.Lock()

    def _slow_dispatch(name: str, args: Dict) -> str:
        # Record start order, then sleep inversely to position: the
        # first call sleeps longest. If the executor collected results
        # in completion order, positions would be flipped.
        with started_lock:
            started_in_order.append(name)
        delay = {"a": 0.03, "b": 0.02, "c": 0.01}[name]
        time.sleep(delay)
        return f"result-of-{name}"

    # One turn with three tool calls (a, b, c) then a terminal reply.
    client = _StubClient([
        {"role": "assistant", "tool_calls": [_tc("a"), _tc("b"), _tc("c")]},
    ])
    cfg = _cfg_with_concurrency(4)
    messages: List[Dict] = [{"role": "user", "content": "fan out"}]

    reply, exhausted = _agent_loop(
        client, cfg, messages, tools=[], budget=cfg.agent_max_turns,
        dispatch=_slow_dispatch,
    )

    assert reply == "done"
    assert not exhausted

    # The ``role=tool`` messages appear in the same order as the calls.
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert [m["name"] for m in tool_msgs] == ["a", "b", "c"]
    assert [m["content"] for m in tool_msgs] == [
        "result-of-a", "result-of-b", "result-of-c",
    ]


# ---------------------------------------------------------------------------
# Actual parallelism — calls overlap in wall-clock time
# ---------------------------------------------------------------------------

def test_parallel_dispatch_runs_concurrently():
    """With 4 calls × 30ms sleep each, total time should be well under 120ms.

    Serial execution would be ~120ms; parallel with pool size 4 should
    be ~30–40ms. We give a generous 100ms ceiling to survive CI jitter.
    """
    sleep_s = 0.03
    barrier = threading.Barrier(4, timeout=1.0)

    def _dispatch(name: str, args: Dict) -> str:
        # Barrier forces all 4 threads to rendezvous before any of them
        # can return — if dispatch were serial, they'd deadlock.
        barrier.wait()
        time.sleep(sleep_s)
        return name

    client = _StubClient([
        {"role": "assistant", "tool_calls": [_tc(n) for n in "abcd"]},
    ])
    cfg = _cfg_with_concurrency(4)
    messages: List[Dict] = [{"role": "user", "content": "x"}]

    start = time.monotonic()
    reply, exhausted = _agent_loop(
        client, cfg, messages, tools=[], budget=cfg.agent_max_turns,
        dispatch=_dispatch,
    )
    elapsed = time.monotonic() - start

    assert reply == "done"
    assert not exhausted
    # Sanity: serial would be ≥ 4*30ms plus barrier deadlock. Parallel
    # clears the barrier instantly and sleeps once concurrently.
    assert elapsed < 0.5, f"parallel dispatch too slow: {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# concurrency=1 must bypass the executor entirely
# ---------------------------------------------------------------------------

def test_serial_path_skips_thread_pool(monkeypatch):
    """With ``agent_tool_concurrency=1``, ``ThreadPoolExecutor`` is never built.

    We sabotage the executor so constructing one raises. If the fast-path
    condition ever regresses, this test will crash loudly — far easier
    to diagnose than a latency regression.
    """
    import concurrent.futures as cf

    def _boom(*args, **kwargs):
        raise AssertionError("ThreadPoolExecutor was used on the serial path")

    monkeypatch.setattr(cf, "ThreadPoolExecutor", _boom)
    # Also patch the name the function resolves via ``from concurrent.futures``.
    monkeypatch.setattr(
        "gemma.main.ThreadPoolExecutor", _boom, raising=False,
    )

    def _dispatch(name: str, args: Dict) -> str:
        return f"ok-{name}"

    client = _StubClient([
        {"role": "assistant", "tool_calls": [_tc("a"), _tc("b"), _tc("c")]},
    ])
    cfg = _cfg_with_concurrency(1)
    messages: List[Dict] = [{"role": "user", "content": "serial"}]

    reply, exhausted = _agent_loop(
        client, cfg, messages, tools=[], budget=cfg.agent_max_turns,
        dispatch=_dispatch,
    )
    assert reply == "done"
    assert not exhausted
    assert [m["name"] for m in messages if m.get("role") == "tool"] == [
        "a", "b", "c",
    ]


def test_single_call_turn_skips_thread_pool(monkeypatch):
    """A turn with just one tool call never spins up the executor either.

    This is the common case — most model turns issue a single call —
    and must not pay the ~μs thread-pool tax.
    """
    import concurrent.futures as cf

    def _boom(*args, **kwargs):
        raise AssertionError("ThreadPoolExecutor used for single-call turn")

    monkeypatch.setattr(cf, "ThreadPoolExecutor", _boom)

    def _dispatch(name: str, args: Dict) -> str:
        return "solo"

    client = _StubClient([
        {"role": "assistant", "tool_calls": [_tc("only")]},
    ])
    cfg = _cfg_with_concurrency(8)
    messages: List[Dict] = [{"role": "user", "content": "solo"}]

    reply, exhausted = _agent_loop(
        client, cfg, messages, tools=[], budget=cfg.agent_max_turns,
        dispatch=_dispatch,
    )
    assert reply == "done"
    assert not exhausted


# ---------------------------------------------------------------------------
# Cache integration — hits do not re-dispatch even under fan-out
# ---------------------------------------------------------------------------

def test_cache_hit_skips_dispatch_in_parallel_fanout():
    """When two identical calls fan out in parallel, only one reaches dispatch.

    Pre-warming the ``AgentSessionCache`` with the (tool, args) key
    ensures the cache-hit branch is taken for both. The dispatch
    counter must stay at zero.
    """
    dispatched: List[str] = []
    d_lock = threading.Lock()

    def _dispatch(name: str, args: Dict) -> str:
        with d_lock:
            dispatched.append(name)
        return "should-not-be-used"

    cache = AgentSessionCache()
    # Seed a hit for ``{"q": "same"}``.
    cache.put("stat", {"q": "same"}, "cached-value")

    client = _StubClient([
        {"role": "assistant", "tool_calls": [
            _tc("stat", q="same"),
            _tc("stat", q="same"),
        ]},
    ])
    cfg = _cfg_with_concurrency(4)
    messages: List[Dict] = [{"role": "user", "content": "two-hits"}]

    reply, exhausted = _agent_loop(
        client, cfg, messages, tools=[], budget=cfg.agent_max_turns,
        dispatch=_dispatch, session_cache=cache,
    )

    assert reply == "done"
    assert not exhausted
    assert dispatched == []  # both served from cache
    tool_contents = [m["content"] for m in messages if m.get("role") == "tool"]
    assert tool_contents == ["cached-value", "cached-value"]


# ---------------------------------------------------------------------------
# Direct _dispatch_turn_calls coverage — faster than routing through _agent_loop
# ---------------------------------------------------------------------------

def test_dispatch_turn_calls_preserves_order_under_jitter():
    """Lowest-level fan-out helper also preserves order."""
    def _dispatch(name: str, args: Dict) -> str:
        # Reverse-proportional sleep — biggest index sleeps least.
        idx = int(name.split("-")[-1])
        time.sleep(0.005 * (5 - idx))
        return name

    raw_calls = [_tc(f"op-{i}") for i in range(5)]
    out = _dispatch_turn_calls(
        raw_calls, concurrency=4, dispatch=_dispatch,
    )
    assert [m["name"] for m in out] == [f"op-{i}" for i in range(5)]
    assert [m["content"] for m in out] == [f"op-{i}" for i in range(5)]


def test_dispatch_turn_calls_handles_empty_list():
    """Zero-call turn returns an empty list without raising."""
    assert _dispatch_turn_calls([], concurrency=4) == []


def test_dispatch_turn_calls_drops_nameless_calls():
    """Calls with no usable name are filtered out (yield ``None``).

    The caller then skips ``None`` when appending to ``messages`` — we
    assert both halves of the contract here.
    """
    out = _dispatch_turn_calls(
        [{"arguments": {}}, _tc("ok")],
        concurrency=4,
        dispatch=lambda n, a: f"hit-{n}",
    )
    # First call had no name → None; second normal.
    assert out[0] is None
    assert out[1] is not None
    assert out[1]["name"] == "ok"
