"""``plan`` — meta-tool that splits a long task into executor sub-steps (#19).

How the split works
-------------------
A naïve tool-use loop keeps both *planning* and *execution* in the same
growing message list. On long research-style tasks, every intermediate
page body and tool result stays in the log even after its information
has been absorbed, and the context balloons past the model's usable
window.

The ``plan`` meta-tool is the model's way to announce "I want to do
these N sub-tasks". The handler does NOT execute anything — it raises
the :class:`PlanRequested` sentinel exception, which propagates past
:class:`gemma.tools.dispatcher.Dispatcher` and is caught by
:func:`gemma.main._agent_loop`. The agent loop then hands control to
:func:`gemma.agent.planner.run_plan`, which runs each step in its own
short sub-conversation and folds only the summary back into the parent
log. Net effect: the parent log grows by one ``role=tool`` per step,
not by ``step_count × sub_tool_body_size``.

Feature flag
------------
The tool registers itself unconditionally at import time (the registry
is global). Advertising and enforcement are gated by
``cfg.plan_tool_enabled`` in :func:`gemma.main._agent_loop` — when
disabled, the tool is filtered out of the advertised schema list and
any direct attempt by the model is refused with a structured error.
"""

from __future__ import annotations

from typing import Any, List

from gemma.tools.capabilities import Capability
from gemma.tools.planning import PlanRequested
from gemma.tools.registry import ToolResult, ToolSpec, tool


#: Hard ceiling on the number of steps a single ``plan`` call may
#: contain. Larger plans tend to be the model hallucinating structure
#: rather than a real decomposition — cap keeps runaway planning in
#: check and bounds memory use by the planner runner.
_MAX_STEPS = 10


@tool(ToolSpec(
    name="plan",
    description=(
        "Decompose a long task into a short ordered list of sub-tasks. "
        "Call this ONCE at the start of complex research or multi-step work. "
        "The runtime will execute each step in its own sub-conversation and "
        "fold the results back — you do NOT run the steps yourself. "
        "Keep each step to a single sentence; 2–5 steps total is ideal."
    ),
    parameters={
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "description": (
                    "Ordered list of step descriptions, one sentence each. "
                    f"Up to {_MAX_STEPS} steps."
                ),
                "items": {"type": "string"},
            },
        },
        "required": ["steps"],
        "additionalProperties": False,
    },
    capability=Capability.READ,
    # The handler itself is effectively free — all the real cost is in
    # the sub-conversations the planner kicks off. A generous timeout
    # on the meta-call is fine.
    timeout_s=5,
    max_output_bytes=4 * 1024,
))
def plan(steps: Any) -> ToolResult:
    """Validate ``steps`` and raise :class:`PlanRequested` for the agent loop.

    Args:
        steps: Must be a non-empty list of strings. The schema validator
            already enforces the ``array`` type + string ``items``, so
            by the time we get here we trust the shape — but still
            reject degenerate cases (empty list, all-whitespace entries,
            over-long plan).

    Raises:
        PlanRequested: On success. This is the intended control-flow —
            the agent loop catches the exception and enters executor
            mode.

    Returns:
        A :class:`ToolResult` only on *validation failure*, so the
        model sees a structured error rather than a silent no-op.
        Successful plans never reach the return statement.
    """
    # Defensive — the schema should have caught this, but the meta-tool
    # is special enough that we double-check. Cheap.
    if not isinstance(steps, list) or not steps:
        return ToolResult(
            ok=False,
            error="invalid_args",
            content="plan: steps must be a non-empty list of strings.",
        )

    # Filter to non-empty trimmed strings; reject if that leaves zero.
    cleaned: List[str] = [
        str(s).strip() for s in steps if isinstance(s, str) and str(s).strip()
    ]
    if not cleaned:
        return ToolResult(
            ok=False,
            error="invalid_args",
            content="plan: every step must be a non-empty string.",
        )
    if len(cleaned) > _MAX_STEPS:
        return ToolResult(
            ok=False,
            error="invalid_args",
            content=(
                f"plan: too many steps ({len(cleaned)}). "
                f"Limit is {_MAX_STEPS}; consolidate related sub-tasks."
            ),
        )

    # Signal up to the agent loop. The dispatcher is wired to let this
    # sentinel propagate; see gemma/tools/dispatcher.py.
    raise PlanRequested(cleaned)


__all__ = ["plan", "_MAX_STEPS"]
