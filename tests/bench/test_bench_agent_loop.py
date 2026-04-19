"""Benchmark the agentic ``_agent_loop`` helper.

This module backs improvements #15 (wire tool-use loop into
``gemma ask``), #20 (parallel tool dispatch) and #21 (agentic-session
cache).

What we are measuring here is *loop overhead only*:

* Ollama is **stubbed** — a deterministic fake client that emits
  exactly the ``tool_calls`` we ask for and then a terminal reply.
* Tool handlers are **stubbed** — no I/O, no Redis, no network.

That keeps the numbers focused on the Python cost of one agent turn
(parse the tool_calls list, dispatch N handlers, append N role=tool
messages, hand back to the model). Anything network-bound (Ollama
latency, DuckDuckGo latency) belongs in a separate hand-timed test,
not here.

Baseline the overhead against #15's stated target (≤2 ms per turn on
top of the Ollama round-trip). If a future change pushes loop
overhead above ~5 ms, something has regressed.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List

import pytest

from gemma.agent.cache import AgentSessionCache
from gemma.config import Config
from gemma.main import _agent_loop


# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------

class _StubClient:
    """Deterministic stand-in for ``ollama.Client``.

    ``scripted_calls`` is a list-of-lists: each inner list is the
    ``tool_calls`` the "model" should emit on one turn. After the
    script is exhausted the client returns a plain assistant reply
    so the loop terminates.
    """

    def __init__(self, scripted_calls: List[List[Dict]]):
        self._calls = list(scripted_calls)

    def chat(self, *, model, messages, tools=None, **_):  # noqa: D401
        if self._calls:
            tool_calls = self._calls.pop(0)
            return {"message": {"role": "assistant", "tool_calls": tool_calls}}
        return {"message": {"role": "assistant", "content": "done"}}


def _stub_dispatch(name: str, arguments: Dict) -> str:
    """Trivially succeed for every call — we're timing the loop, not the tools."""
    return f"ok:{name}:{len(arguments)}"


def _parallel_loop(client: _StubClient, dispatch: Callable, max_turns: int = 8,
                   concurrency: int = 4) -> Dict:
    """Parallel within-turn dispatch — what item #20 will ship with."""
    from concurrent.futures import ThreadPoolExecutor

    messages: List[Dict] = [{"role": "user", "content": "go"}]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for _ in range(max_turns):
            reply = client.chat(model="stub", messages=messages)
            calls = reply["message"].get("tool_calls") or []
            if not calls:
                return reply
            results = list(pool.map(
                lambda c: (c, dispatch(c["name"], c.get("arguments", {}))),
                calls,
            ))
            for call, result in results:
                messages.append({"role": "tool", "name": call["name"], "content": result})
    return reply


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def test_one_tool_roundtrip(benchmark):
    """Single tool call + terminal reply. Backs #15.

    Uses the real ``_agent_loop`` from ``gemma.main``. The target is ≤ 2 ms
    overhead on top of the two stubbed model calls.
    """
    script = [[{"name": "rag_query", "arguments": {"query": "hi", "k": 5}}]]
    cfg = Config()

    def _run():
        client = _StubClient(list(script))
        messages = [{"role": "user", "content": "go"}]
        _agent_loop(client, cfg, messages, [], cfg.agent_max_turns, dispatch=_stub_dispatch)

    benchmark(_run)


@pytest.mark.parametrize("n_tools", [2, 3, 5])
def test_parallel_three_tools(benchmark, n_tools):
    """Fan-out turn with ``n_tools`` calls. Backs #20.

    Each handler does a 5 ms ``sleep`` so parallelism is visible.
    """
    def _slow_dispatch(name: str, arguments: Dict) -> str:
        time.sleep(0.005)
        return "ok"

    script = [[{"name": "http_get", "arguments": {"url": f"u{i}"}}
               for i in range(n_tools)]]

    def _run():
        client = _StubClient([list(script[0])])
        _parallel_loop(client, _slow_dispatch, concurrency=max(n_tools, 2))

    benchmark(_run)


def test_repeated_tool_cached(benchmark):
    """Same tool fired twice in one turn — second call collapses to a dict lookup. Backs #21.

    Uses the real ``_agent_loop`` with an ``AgentSessionCache``.
    The ``rag_query`` tool is a READ-capability registered builtin, so
    its result is cached after the first dispatch.  The second identical
    call must not hit ``_counting_dispatch`` at all.

    Target: second-call cost ≤ 1 µs (pure dict lookup, no sleep).
    """
    # Import builtins to ensure rag_query is registered (needed for
    # capability lookup inside _agent_loop's cache-population path).
    import gemma.tools.builtins  # noqa: F401

    call_count = {"n": 0}

    def _counting_dispatch(name: str, arguments: Dict) -> str:
        call_count["n"] += 1
        time.sleep(0.001)
        return f"ok:{call_count['n']}"

    identical = {"name": "rag_query", "arguments": {"query": "same", "k": 5}}
    script = [[dict(identical), dict(identical)]]
    cfg = Config()

    def _run():
        call_count["n"] = 0
        client = _StubClient([list(script[0])])
        messages = [{"role": "user", "content": "go"}]
        cache = AgentSessionCache()
        _agent_loop(
            client, cfg, messages, [], cfg.agent_max_turns,
            dispatch=_counting_dispatch,
            session_cache=cache,
        )
        assert call_count["n"] == 1, (
            f"second identical call should be cached, got {call_count['n']} dispatches"
        )

    benchmark(_run)
