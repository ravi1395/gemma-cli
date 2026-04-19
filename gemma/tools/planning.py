"""Sentinel type used by the ``plan`` meta-tool (item #19).

Lives in its own leaf module to keep the import graph tidy:

* ``gemma.tools.dispatcher`` imports :class:`PlanRequested` so it can
  let the sentinel propagate instead of wrapping it as a generic
  ``handler_exception``.
* ``gemma.tools.builtins.plan`` imports :class:`PlanRequested` and
  raises it from the ``plan`` tool handler.
* ``gemma.agent.planner`` imports :class:`PlanRequested` and
  ``gemma.main._agent_loop`` catches it to switch into executor mode.

Keeping it as a zero-dependency module means none of the above pull in
each other transitively.
"""

from __future__ import annotations

from typing import Iterable, List


class PlanRequested(Exception):
    """Raised by the ``plan`` tool handler to signal a planner/executor split.

    The model calls ``plan(["step 1", "step 2", ...])`` to hand the
    dispatcher a list of sub-tasks. The ``plan`` handler does not
    execute the steps itself — it just captures them and raises this
    exception, which propagates past the dispatcher up to the agent
    loop, where :func:`gemma.agent.planner.run_plan` takes over.

    Attributes:
        steps: The step descriptions as a stable list. Whatever the
            model passed is normalised to ``List[str]`` here so every
            consumer downstream can rely on that shape.
    """

    def __init__(self, steps: Iterable[str]) -> None:
        # Normalise + freeze a copy so later mutation of the caller's
        # list cannot change what the planner executes.
        self.steps: List[str] = [str(s).strip() for s in steps]
        super().__init__(
            f"PlanRequested({len(self.steps)} steps): "
            + "; ".join(self.steps[:3])
            + ("..." if len(self.steps) > 3 else "")
        )
