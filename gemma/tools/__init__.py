"""Public API for the gemma tool-use subsystem.

This package implements the capability-gated, audit-logged function-call
layer that lets Gemma invoke a small, vetted set of tools. Importing
the package has two effects:

1. Exposes the stable API — :class:`ToolSpec`, :class:`Capability`,
   :class:`ToolResult`, and the :func:`tool` decorator — from one
   place, so feature code never reaches into internal modules.
2. Triggers the side-effectful *auto-registration* of every built-in
   tool by importing ``gemma.tools.builtins``. Each builtin module
   applies ``@tool(...)`` at import time, populating the registry.

Import order is significant: :mod:`capabilities`, :mod:`registry`, and
:mod:`audit` must be importable *before* any builtin runs its
decorator, which is why they are all imported here first.
"""

from __future__ import annotations

# Re-export stable types so feature code can ``from gemma.tools import ...``.
# Note: do *not* re-export the ``registry()`` snapshot helper here —
# doing so would shadow the ``gemma.tools.registry`` submodule attribute,
# breaking ``from gemma.tools import registry`` for the submodule.
# Callers who need the snapshot can use ``from gemma.tools.registry import
# registry`` explicitly.
from gemma.tools.capabilities import Capability  # noqa: F401
from gemma.tools.registry import (  # noqa: F401
    ToolResult,
    ToolSpec,
    get,
    mount,
    register,
    tool,
)

# Eagerly import the built-in tools so their @tool decorators run. This
# is done last, after all the machinery above is in place.
from gemma.tools import builtins  # noqa: F401, E402
