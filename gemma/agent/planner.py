"""Planner/executor runner for the ``plan`` meta-tool (item #19).

The ``plan`` tool's handler raises :class:`PlanRequested` with a list
of sub-tasks. :func:`gemma.main._agent_loop` catches that sentinel and
delegates to :func:`run_plan`, which:

1. (Optional) prompts the user to confirm the plan in TTY sessions
   when the step count exceeds ``cfg.plan_confirm_threshold``.
2. For each step, builds a short *sub-conversation* seeded with just
   the system prompt, a one-line digest of the parent conversation,
   and the step text as the user turn.
3. Invokes ``_agent_loop`` recursively on that sub-conversation with a
   divided budget. Depth is tracked so a sub-step cannot itself issue
   another ``plan(...)`` unless ``cfg.agent_max_plan_depth`` permits.
4. Collects the terminal assistant reply of each sub-conversation as
   the step's result.
5. Folds the per-step summaries back into the parent message list as
   a single ``role=tool name=plan`` entry. This is the critical
   context-efficiency win: the parent grows by one tool message per
   step, not by the cumulative body of every tool call the step made
   internally.

The implementation is intentionally small and dependency-free so it
can be unit-tested with the same stub ``_StubClient`` pattern used by
the existing agent loop tests.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from gemma.tools.planning import PlanRequested

if TYPE_CHECKING:  # pragma: no cover - type hint only
    from gemma.config import Config


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StepResult:
    """Outcome of one executor sub-conversation.

    Attributes:
        index:             Zero-based position of the step within the
                           plan.
        step:              The step description the model gave the
                           planner.
        reply:             The terminal assistant text from the step's
                           sub-conversation. Trimmed and truncated to
                           ``_STEP_SUMMARY_BUDGET`` characters so it
                           never dwarfs the parent context.
        budget_exhausted:  True iff the sub-conversation hit its turn
                           budget without producing a terminal reply.
                           The partial ``reply`` is still captured.
    """

    index: int
    step: str
    reply: str
    budget_exhausted: bool = False


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Per-step reply size written into the parent log. The whole point of
# the planner is to keep the parent from growing by the sum of sub-tool
# bodies — so trim hard. The model still saw the full sub-conversation
# while formulating the reply; we only truncate the forwarded *summary*.
_STEP_SUMMARY_BUDGET = 1500  # characters, ~= 400 tokens


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_plan(
    *,
    client: Any,
    cfg: "Config",
    parent_messages: List[Dict[str, Any]],
    steps: List[str],
    tools: List[Dict[str, Any]],
    dispatch: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
    session_cache: Optional[Any] = None,
    session_id: str = "agent",
    depth: int = 1,
    confirm: Optional[Callable[[List[str]], bool]] = None,
    agent_loop: Optional[Callable[..., Any]] = None,
) -> List[StepResult]:
    """Execute the plan's steps in sequence, each in its own sub-conversation.

    Side effect:
        ``parent_messages`` is mutated in place: a single
        ``role=tool name=plan`` entry is appended summarising every
        step. This lets the outer loop resume normally after the plan.

    Args:
        client:          Ollama-like object with ``.chat(...)``.
        cfg:             Active Config — drives budget splitting,
                         confirmation threshold, and depth policy.
        parent_messages: The outer conversation list. A single
                         summarising ``role=tool`` message is appended
                         to it before return.
        steps:           The step list from :class:`PlanRequested`.
        tools:           Tool schemas. We intentionally pass these
                         through unchanged so every step can still
                         call any non-``plan`` tool. (The caller
                         filters ``plan`` out when depth would
                         exceed ``agent_max_plan_depth``.)
        dispatch:        Tool-call dispatcher, reused across all
                         sub-conversations so audit and cache stay
                         coherent.
        session_cache:   Shared across steps — an expensive READ in
                         step 1 is free in step 3.
        session_id:      Threaded through audit records.
        depth:           Current planner nesting depth (1 for the
                         first level). Incremented when recursing.
        confirm:         Optional interactive hook ``(steps) -> bool``.
                         Defaults to a TTY-aware ``_prompt_confirm``
                         when ``plan_confirm_threshold`` is exceeded;
                         tests inject a lambda.
        agent_loop:      Injection point for the agent loop callable.
                         Defaults to the production
                         ``gemma.main._agent_loop``; tests pass a
                         tracking stub so we can assert the recursive
                         call without pulling in the full CLI.

    Returns:
        One :class:`StepResult` per executed step. On user refusal
        an empty list is returned and the parent log records a
        cancellation message.
    """
    # Lazy import — avoids a circular dependency between gemma.main
    # (which imports this module) and gemma.agent.planner.
    if agent_loop is None:  # pragma: no cover - production path
        from gemma.main import _agent_loop
        agent_loop = _agent_loop

    # ---- (Optional) confirmation ------------------------------------
    threshold = int(getattr(cfg, "plan_confirm_threshold", 3) or 0)
    if confirm is None:
        confirm = _default_confirm_prompt
    if threshold > 0 and len(steps) > threshold:
        approved = confirm(steps)
        if not approved:
            parent_messages.append(_make_refusal_message(
                reason="user declined to execute plan",
                steps=steps,
            ))
            return []

    # ---- Budget split ------------------------------------------------
    per_step_budget = _per_step_budget(cfg, len(steps))
    parent_digest = _digest_parent(parent_messages)
    system_prompt = getattr(cfg, "system_prompt", "")

    # ---- Sequential execution ---------------------------------------
    results: List[StepResult] = []
    for idx, step_text in enumerate(steps):
        sub_messages = _build_sub_messages(
            system_prompt=system_prompt,
            parent_digest=parent_digest,
            step_text=step_text,
            prior_results=results,
        )
        # Each step gets its OWN message list — bodies never bleed back
        # into the parent. ``_agent_loop`` is called recursively; the
        # ``_plan_depth`` kwarg lets a nested ``plan()`` call be refused
        # by policy.
        reply_text, exhausted = agent_loop(
            client, cfg, sub_messages, tools, per_step_budget,
            dispatch=dispatch,
            session_cache=session_cache,
            session_id=session_id,
            _plan_depth=depth,
        )
        results.append(StepResult(
            index=idx,
            step=step_text,
            reply=_truncate(reply_text),
            budget_exhausted=bool(exhausted),
        ))

    # ---- Summarise back into the parent -----------------------------
    parent_messages.append(_make_summary_message(results))
    return results


# ---------------------------------------------------------------------------
# Internals — small, pure, easy to unit-test
# ---------------------------------------------------------------------------

def _per_step_budget(cfg: "Config", step_count: int) -> int:
    """Divide ``cfg.agent_max_turns`` across steps, floor at the configured min.

    Lowering the per-step budget helps keep total wall time bounded,
    but each step still needs at least two turns: one to call a tool
    and one to reply. ``plan_min_step_budget`` enforces that floor.
    """
    if step_count <= 0:
        return max(1, int(getattr(cfg, "plan_min_step_budget", 2) or 2))
    total = int(getattr(cfg, "agent_max_turns", 8) or 8)
    floor = int(getattr(cfg, "plan_min_step_budget", 2) or 2)
    return max(floor, total // step_count)


def _digest_parent(parent_messages: List[Dict[str, Any]]) -> str:
    """Extract a short hand-off string from the parent conversation.

    V1 is deliberately naïve — use the most recent *user* prompt as
    the digest. It's what drove the model to plan in the first place,
    so each step benefits from seeing it. Future work may swap in a
    model-produced summary capped at ~300 tokens.
    """
    for msg in reversed(parent_messages):
        if msg.get("role") == "user" and msg.get("content"):
            return str(msg["content"])
    return ""


def _build_sub_messages(
    *,
    system_prompt: str,
    parent_digest: str,
    step_text: str,
    prior_results: List[StepResult],
) -> List[Dict[str, Any]]:
    """Seed a sub-conversation for one step.

    The shape is deliberately minimal:

    * ``system``    — global system prompt, unchanged.
    * ``user``      — two sections: the overall goal (parent digest)
                      and the step we want the model to perform now.
                      Prior steps' summaries are embedded so each
                      step knows what preceded it, without carrying
                      raw tool bodies.
    """
    context_block = parent_digest or "(no prior user prompt)"
    prior_block = _format_prior_results(prior_results) or "(this is the first step)"

    user_body = (
        f"Overall goal:\n{context_block}\n\n"
        f"Prior step results so far:\n{prior_block}\n\n"
        f"Your current step ({len(prior_results) + 1}): {step_text}\n"
        "Perform this step and respond with the findings. Do NOT call "
        "the `plan` tool again; just do the work."
    )

    msgs: List[Dict[str, Any]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_body})
    return msgs


def _format_prior_results(results: List[StepResult]) -> str:
    """Compact textual summary of already-completed steps."""
    if not results:
        return ""
    lines: List[str] = []
    for r in results:
        lines.append(f"- Step {r.index + 1} ({r.step!r}): {r.reply or '(no output)'}")
    return "\n".join(lines)


def _truncate(text: str) -> str:
    """Cap forwarded step text at :data:`_STEP_SUMMARY_BUDGET` chars."""
    text = (text or "").strip()
    if len(text) <= _STEP_SUMMARY_BUDGET:
        return text
    return text[: _STEP_SUMMARY_BUDGET - 20].rstrip() + " …[truncated]"


def _make_summary_message(results: List[StepResult]) -> Dict[str, Any]:
    """Build the single ``role=tool name=plan`` message to fold back.

    Content is JSON so the model can parse it on the next turn; it
    stays small because each step's reply is already truncated to
    :data:`_STEP_SUMMARY_BUDGET`.
    """
    payload = {
        "kind": "plan_complete",
        "step_count": len(results),
        "steps": [
            {
                "index": r.index,
                "step": r.step,
                "result_summary": r.reply,
                "budget_exhausted": r.budget_exhausted,
            }
            for r in results
        ],
    }
    return {
        "role": "tool",
        "name": "plan",
        "content": json.dumps(payload, ensure_ascii=False),
    }


def _make_refusal_message(*, reason: str, steps: List[str]) -> Dict[str, Any]:
    """Surface a plan-skipped state back to the model as a tool message."""
    payload = {
        "kind": "plan_skipped",
        "reason": reason,
        "step_count": len(steps),
    }
    return {
        "role": "tool",
        "name": "plan",
        "content": json.dumps(payload, ensure_ascii=False),
    }


def _default_confirm_prompt(steps: List[str]) -> bool:
    """Default y/N prompt for interactive TTY sessions.

    Non-TTY returns True — the caller is expected to set
    ``plan_confirm_threshold = 0`` in pipeline profiles to avoid
    blocking, but we choose the permissive default so a missing
    configuration doesn't deadlock an unattended run.
    """
    if not sys.stdout.isatty():
        return True
    print("\nThe agent proposes the following plan:")
    for i, step in enumerate(steps, start=1):
        print(f"  {i}. {step}")
    try:
        answer = input("Execute this plan? [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


__all__ = ["run_plan", "StepResult", "PlanRequested"]
