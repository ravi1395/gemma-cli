"""Unit tests for gemma/main.py — specifically the _agent_loop helper.

Two scenarios are covered as required by improvement #15:

(a) One-tool round-trip with a stub client — verifies that the loop
    calls the model, dispatches the tool, and returns the final content.
(b) Budget exhaustion — verifies that the loop stops after ``budget``
    turns and signals the caller via the second return value.
"""

from __future__ import annotations

from typing import Dict, List

from gemma.config import Config
from gemma.main import _agent_loop


# ---------------------------------------------------------------------------
# Helpers shared by both tests
# ---------------------------------------------------------------------------

class _StubClient:
    """Deterministic Ollama client stand-in.

    ``scripted_turns`` is a list-of-dicts.  Each dict becomes the
    "message" for that model turn.  When the script is exhausted the
    client emits a plain terminal reply.
    """

    def __init__(self, scripted_turns: List[Dict]) -> None:
        self._turns = list(scripted_turns)

    def chat(self, *, model, messages, tools=None, **_):  # noqa: D401
        if self._turns:
            return {"message": self._turns.pop(0)}
        return {"message": {"role": "assistant", "content": "done"}}


def _make_tool_call(name: str, arguments: Dict) -> Dict:
    """Build a flat tool-call dict matching the bench stub format."""
    return {"name": name, "arguments": arguments}


# ---------------------------------------------------------------------------
# Test (a): one-tool round-trip
# ---------------------------------------------------------------------------

def test_one_tool_roundtrip_returns_final_content():
    """_agent_loop calls the model twice and dispatches one tool call.

    Turn 1: model emits one tool_call.
    Turn 2: model sees the tool result and replies "done".
    Expected: loop returns "done", budget_exhausted=False.
    """
    call_log: List[str] = []

    def _dispatch(name: str, args: Dict) -> str:
        call_log.append(name)
        return f"result-of-{name}"

    client = _StubClient([
        # Turn 1 — one tool call.
        {"role": "assistant", "tool_calls": [_make_tool_call("list_dir", {"path": "."})]},
    ])
    cfg = Config()
    messages = [{"role": "user", "content": "list files"}]

    reply, budget_exhausted = _agent_loop(
        client, cfg, messages, tools=[], budget=cfg.agent_max_turns,
        dispatch=_dispatch,
    )

    assert reply == "done"
    assert not budget_exhausted
    assert call_log == ["list_dir"], "expected exactly one dispatched tool call"
    # messages should now include: user, assistant(tool_calls), tool, assistant(final)
    roles = [m["role"] for m in messages]
    assert "tool" in roles


# ---------------------------------------------------------------------------
# Test (b): budget exhaustion returns partial output
# ---------------------------------------------------------------------------

def test_budget_exhaustion_signals_caller():
    """_agent_loop honours the budget cap and returns budget_exhausted=True.

    Every model turn emits a tool call so the loop never reaches a
    terminal reply.  With budget=2 the loop should stop after 2 turns.
    """
    dispatch_count = {"n": 0}

    def _dispatch(name: str, args: Dict) -> str:
        dispatch_count["n"] += 1
        return "ok"

    def _make_scripted_turn():
        return {"role": "assistant", "tool_calls": [_make_tool_call("stat", {"path": "."})]}

    # Provide more scripted turns than the budget so we can confirm the
    # loop stops early rather than running all of them.
    client = _StubClient([_make_scripted_turn() for _ in range(10)])
    cfg = Config()
    messages = [{"role": "user", "content": "go"}]

    reply, budget_exhausted = _agent_loop(
        client, cfg, messages, tools=[], budget=2,
        dispatch=_dispatch,
    )

    assert budget_exhausted, "loop should report budget exhaustion"
    assert dispatch_count["n"] == 2, (
        f"expected 2 dispatch calls (one per budget turn), got {dispatch_count['n']}"
    )
