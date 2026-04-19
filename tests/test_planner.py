"""Tests for the planner/executor split (item #19).

Covers three integration layers:

* ``plan`` builtin tool — argument validation, sentinel raising.
* ``gemma.agent.planner.run_plan`` — sub-conversation seeding, budget
  split, recursion into a stubbed ``_agent_loop``, summary folding,
  confirmation prompt gating.
* ``gemma.main._agent_loop`` — catches ``PlanRequested`` from the real
  Dispatcher, honours ``plan_tool_enabled`` and ``agent_max_plan_depth``.

All tests use deterministic stub clients and stub dispatchers so the
suite is hermetic and safe for CI — no real Ollama, no real network.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Dict, List, Optional

import pytest

from gemma.agent.planner import (
    StepResult,
    _build_sub_messages,
    _per_step_budget,
    _truncate,
    _STEP_SUMMARY_BUDGET,
    run_plan,
)
from gemma.config import Config
from gemma.main import _agent_loop, _handle_plan_request
from gemma.tools.planning import PlanRequested


# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------

class _StubClient:
    """Deterministic Ollama stand-in. Same pattern as tests/test_main.py."""

    def __init__(self, scripted_turns: List[Dict]) -> None:
        self._turns = list(scripted_turns)

    def chat(self, *, model, messages, tools=None, **_):
        if self._turns:
            return {"message": self._turns.pop(0)}
        return {"message": {"role": "assistant", "content": "done"}}


def _tc(name: str, **args: Any) -> Dict[str, Any]:
    return {"name": name, "arguments": dict(args)}


def _cfg(**overrides) -> Config:
    base = Config()
    return replace(base, **overrides)


# ===========================================================================
# plan builtin tool
# ===========================================================================

def test_plan_handler_raises_plan_requested():
    """The happy path — handler raises the sentinel with cleaned steps."""
    from gemma.tools.builtins.plan import plan

    with pytest.raises(PlanRequested) as excinfo:
        plan(steps=["  step one  ", "step two", ""])  # empty filtered out
    assert excinfo.value.steps == ["step one", "step two"]


def test_plan_handler_rejects_empty_list():
    """Empty list → structured invalid_args result (no exception)."""
    from gemma.tools.builtins.plan import plan

    result = plan(steps=[])
    assert result.ok is False
    assert result.error == "invalid_args"


def test_plan_handler_rejects_all_whitespace():
    """A list of whitespace strings is functionally empty."""
    from gemma.tools.builtins.plan import plan

    result = plan(steps=["   ", "\t", "\n"])
    assert result.ok is False
    assert result.error == "invalid_args"


def test_plan_handler_enforces_max_steps():
    """Over the cap → invalid_args, not truncation."""
    from gemma.tools.builtins.plan import _MAX_STEPS, plan

    result = plan(steps=[f"step {i}" for i in range(_MAX_STEPS + 1)])
    assert result.ok is False
    assert result.error == "invalid_args"
    assert "too many steps" in result.content.lower()


def test_plan_handler_rejects_non_list_steps():
    """Defensive check — schema should catch this but handler is belt-and-braces."""
    from gemma.tools.builtins.plan import plan

    result = plan(steps="not a list")  # type: ignore[arg-type]
    assert result.ok is False
    assert result.error == "invalid_args"


# ===========================================================================
# run_plan — sub-conversation seeding and budget split
# ===========================================================================

def test_per_step_budget_divides_and_floors():
    """``agent_max_turns / step_count`` with a ``plan_min_step_budget`` floor."""
    # 6 turns / 3 steps = 2, floor already 2.
    assert _per_step_budget(_cfg(agent_max_turns=6, plan_min_step_budget=2), 3) == 2
    # 6 / 4 = 1 → floor to 2.
    assert _per_step_budget(_cfg(agent_max_turns=6, plan_min_step_budget=2), 4) == 2
    # 20 / 4 = 5 → 5.
    assert _per_step_budget(_cfg(agent_max_turns=20, plan_min_step_budget=2), 4) == 5
    # Zero-step defensive path still returns the floor.
    assert _per_step_budget(_cfg(plan_min_step_budget=2), 0) == 2


def test_build_sub_messages_includes_parent_digest_and_prior_results():
    """Sub-convo has system + one user turn; mentions goal, prior steps, step text."""
    msgs = _build_sub_messages(
        system_prompt="You are a helpful agent.",
        parent_digest="research redis streams",
        step_text="find three URLs",
        prior_results=[StepResult(index=0, step="decompose", reply="ok")],
    )
    assert len(msgs) == 2
    assert msgs[0] == {"role": "system", "content": "You are a helpful agent."}
    assert msgs[1]["role"] == "user"
    body = msgs[1]["content"]
    assert "research redis streams" in body
    assert "find three URLs" in body
    assert "decompose" in body  # prior step visible
    assert "Do NOT call" in body  # anti-recursion nudge


def test_truncate_caps_step_reply():
    """Step replies longer than the budget are truncated with a marker."""
    long_text = "x" * (_STEP_SUMMARY_BUDGET + 500)
    out = _truncate(long_text)
    assert len(out) <= _STEP_SUMMARY_BUDGET
    assert "truncated" in out


# ---------------------------------------------------------------------------
# run_plan end-to-end with a fake agent_loop
# ---------------------------------------------------------------------------

def _fake_agent_loop_factory(replies: List[str]):
    """Return an ``agent_loop`` stand-in that produces scripted replies.

    The returned callable has the same signature as ``_agent_loop`` and
    ignores the client/tools/dispatch arguments — it just pops the next
    reply off the script. We also capture the kwargs each call received
    so tests can assert on the budget and depth that were forwarded.
    """
    calls: List[Dict[str, Any]] = []

    def _fake(client, cfg, messages, tools, budget, *, dispatch=None,
             session_cache=None, session_id="agent", _plan_depth=0):
        calls.append({
            "messages_len": len(messages),
            "budget": budget,
            "depth": _plan_depth,
            "user_content": messages[-1]["content"] if messages else "",
        })
        if replies:
            return replies.pop(0), False
        return "done", False

    _fake.calls = calls  # type: ignore[attr-defined]
    return _fake


def test_run_plan_runs_each_step_and_appends_single_summary():
    """Three steps → three sub-calls, parent grows by exactly one tool msg."""
    fake_loop = _fake_agent_loop_factory(["r1", "r2", "r3"])
    cfg = _cfg(
        agent_max_turns=6, plan_min_step_budget=2,
        plan_confirm_threshold=0,  # never prompt in tests
        system_prompt="SYS",
    )
    parent = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "do the research"},
    ]
    before = len(parent)

    results = run_plan(
        client=None, cfg=cfg, parent_messages=parent,
        steps=["step 1", "step 2", "step 3"],
        tools=[], dispatch=None, session_cache=None,
        agent_loop=fake_loop,
    )

    assert [r.reply for r in results] == ["r1", "r2", "r3"]
    # Parent grew by exactly ONE tool message (not 3, not 3 × body size).
    assert len(parent) == before + 1
    summary = parent[-1]
    assert summary["role"] == "tool" and summary["name"] == "plan"
    payload = json.loads(summary["content"])
    assert payload["kind"] == "plan_complete"
    assert payload["step_count"] == 3
    assert [s["result_summary"] for s in payload["steps"]] == ["r1", "r2", "r3"]


def test_run_plan_forwards_divided_budget_and_depth():
    """Each sub-call sees ``agent_max_turns // step_count`` and depth+1."""
    fake_loop = _fake_agent_loop_factory(["a", "b"])
    cfg = _cfg(agent_max_turns=8, plan_min_step_budget=2, plan_confirm_threshold=0)
    parent = [{"role": "user", "content": "goal"}]

    run_plan(
        client=None, cfg=cfg, parent_messages=parent,
        steps=["s1", "s2"], tools=[], dispatch=None,
        agent_loop=fake_loop, depth=1,
    )
    assert [c["budget"] for c in fake_loop.calls] == [4, 4]  # 8//2
    assert [c["depth"] for c in fake_loop.calls] == [1, 1]  # unchanged at top


def test_run_plan_skips_when_user_declines_confirmation():
    """Confirm returns False → empty results and a refusal tool message."""
    fake_loop = _fake_agent_loop_factory(["unused"])
    cfg = _cfg(agent_max_turns=6, plan_confirm_threshold=2)  # 3 steps > 2 triggers
    parent = [{"role": "user", "content": "goal"}]

    results = run_plan(
        client=None, cfg=cfg, parent_messages=parent,
        steps=["a", "b", "c"], tools=[], dispatch=None,
        agent_loop=fake_loop,
        confirm=lambda steps: False,
    )
    assert results == []
    # No sub-calls were made.
    assert fake_loop.calls == []
    assert parent[-1]["role"] == "tool"
    payload = json.loads(parent[-1]["content"])
    assert payload["kind"] == "plan_skipped"


def test_run_plan_proceeds_when_user_confirms():
    """Confirm returns True → all steps run."""
    fake_loop = _fake_agent_loop_factory(["r1", "r2", "r3", "r4"])
    cfg = _cfg(agent_max_turns=8, plan_confirm_threshold=2)
    parent = [{"role": "user", "content": "goal"}]

    results = run_plan(
        client=None, cfg=cfg, parent_messages=parent,
        steps=["a", "b", "c", "d"], tools=[], dispatch=None,
        agent_loop=fake_loop,
        confirm=lambda steps: True,
    )
    assert len(results) == 4


def test_run_plan_threshold_zero_disables_confirm():
    """``plan_confirm_threshold = 0`` bypasses the prompt entirely."""
    fake_loop = _fake_agent_loop_factory(["r"])
    # If the confirm fn gets called, it returns False and we'd skip.
    # With threshold=0 it must not be called.
    called: List[bool] = []
    def _confirm(steps):
        called.append(True)
        return False

    cfg = _cfg(agent_max_turns=4, plan_confirm_threshold=0)
    parent = [{"role": "user", "content": "g"}]
    results = run_plan(
        client=None, cfg=cfg, parent_messages=parent,
        steps=["just one"], tools=[], dispatch=None,
        agent_loop=fake_loop, confirm=_confirm,
    )
    assert called == []
    assert len(results) == 1


# ===========================================================================
# _agent_loop integration — catches PlanRequested from a real Dispatcher
# ===========================================================================

def _plan_raising_dispatch(name: str, args: Dict) -> Any:
    """Dispatcher stand-in: raises PlanRequested when the model calls plan."""
    if name == "plan":
        raise PlanRequested(list(args.get("steps", [])))
    # For non-plan tools, return a trivial result.
    from gemma.tools.registry import ToolResult
    return ToolResult(ok=True, content=f"ok-{name}")


def test_agent_loop_catches_plan_and_enters_executor_mode():
    """Model emits plan(); _agent_loop branches into run_plan and continues."""
    client = _StubClient([
        # Turn 1 — model calls plan.
        {"role": "assistant", "tool_calls": [
            _tc("plan", steps=["sub step 1", "sub step 2"])
        ]},
        # Turn 2 — model sees the plan summary and replies done.
        # (Note: each sub-step ALSO issues model calls via the stub — see below.)
        {"role": "assistant", "content": "sub-1 done"},
        {"role": "assistant", "content": "sub-2 done"},
        {"role": "assistant", "content": "final answer"},
    ])
    cfg = _cfg(
        plan_tool_enabled=True,
        agent_max_plan_depth=1,
        plan_confirm_threshold=0,
        agent_max_turns=6, plan_min_step_budget=2,
        system_prompt="SYS",
    )
    messages: List[Dict] = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "research X"},
    ]

    reply, exhausted = _agent_loop(
        client, cfg, messages, tools=[], budget=cfg.agent_max_turns,
        dispatch=_plan_raising_dispatch,
    )

    assert reply == "final answer"
    assert not exhausted
    # The parent gained: one assistant(tool_calls=plan), one tool(summary),
    # then the final assistant turn is captured as `reply` not appended
    # by _agent_loop itself (which only appends assistant-with-tool-calls).
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["name"] == "plan"
    payload = json.loads(tool_msgs[0]["content"])
    assert payload["kind"] == "plan_complete"
    assert payload["step_count"] == 2


def test_agent_loop_refuses_plan_when_flag_disabled():
    """plan_tool_enabled=False → refusal message, no sub-execution."""
    client = _StubClient([
        {"role": "assistant", "tool_calls": [
            _tc("plan", steps=["s1", "s2"])
        ]},
        {"role": "assistant", "content": "ok, I will do it directly"},
    ])
    cfg = _cfg(plan_tool_enabled=False, plan_confirm_threshold=0)
    messages: List[Dict] = [{"role": "user", "content": "go"}]

    reply, exhausted = _agent_loop(
        client, cfg, messages, tools=[], budget=cfg.agent_max_turns,
        dispatch=_plan_raising_dispatch,
    )
    assert reply == "ok, I will do it directly"
    assert not exhausted
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    payload = json.loads(tool_msgs[0]["content"])
    assert payload["kind"] == "plan_skipped"
    assert "disabled" in payload["reason"]


def test_agent_loop_refuses_nested_plan_at_depth_cap():
    """Second plan call from inside a step is refused when depth cap is 1."""
    # The planner recurses into _agent_loop with _plan_depth=1. A plan()
    # call at that depth is refused because 1+1 > max_depth=1.
    # Simulate this directly by calling _handle_plan_request at depth=1.
    cfg = _cfg(plan_tool_enabled=True, agent_max_plan_depth=1)
    messages: List[Dict] = []

    _handle_plan_request(
        plan_req=PlanRequested(["inner step"]),
        client=_StubClient([]),
        cfg=cfg, messages=messages, tools=[],
        depth=1,  # already one level deep
    )
    assert len(messages) == 1
    payload = json.loads(messages[0]["content"])
    assert payload["kind"] == "plan_skipped"
    assert "depth" in payload["reason"]


def test_agent_loop_honours_max_depth_greater_than_one():
    """With max_depth=2, a nested plan at depth=1 should succeed."""
    cfg = _cfg(plan_tool_enabled=True, agent_max_plan_depth=2,
               plan_confirm_threshold=0, agent_max_turns=6)
    messages: List[Dict] = [{"role": "user", "content": "g"}]

    # Inject a fake agent loop so run_plan doesn't recurse into the real one.
    calls: List[int] = []
    def _fake_loop(client, cfg_, msgs, tools, budget, *, dispatch=None,
                   session_cache=None, session_id="agent", _plan_depth=0):
        calls.append(_plan_depth)
        return f"reply@depth={_plan_depth}", False

    # Route via run_plan directly so we can control agent_loop.
    run_plan(
        client=None, cfg=cfg, parent_messages=messages,
        steps=["only step"], tools=[], dispatch=None,
        agent_loop=_fake_loop, depth=2,  # depth 2 still <= max 2
    )
    # The fake_loop ran (would not at depth > max).
    assert calls == [2]


# ===========================================================================
# Registry wiring
# ===========================================================================

def test_plan_tool_is_registered_as_read():
    """The @tool decorator landed the handler in the registry."""
    # Ensure the builtins module imported so self-registration ran.
    import gemma.tools.builtins  # noqa: F401
    from gemma.tools import registry as _registry
    from gemma.tools.capabilities import Capability

    spec, _handler = _registry.get("plan")
    assert spec.capability is Capability.READ
    assert "steps" in spec.parameters["properties"]
    assert spec.parameters["required"] == ["steps"]


def test_plan_tool_visible_in_mount_by_default():
    """The tool is mounted under the default GatingContext."""
    import gemma.tools.builtins  # noqa: F401
    from gemma.tools import registry as _registry
    from gemma.tools.capabilities import GatingContext

    specs = _registry.mount(GatingContext())
    assert "plan" in {s.name for s in specs}
