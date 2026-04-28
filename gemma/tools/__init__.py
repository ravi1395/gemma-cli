"""Public API for the gemma tool-use subsystem.

This package implements the capability-gated, audit-logged function-call
layer that lets Gemma invoke a small, vetted set of tools.

Importing this package is now cheap. The built-in tools used to be
auto-imported here, which dragged ~125 ms of optional dependencies
(trafilatura's lxml stack, etc.) into every CLI start, even
``gemma --help``. Builtins are now loaded lazily on the first call to
``get`` / ``mount`` / ``all_specs`` / ``registry`` (see
:func:`gemma.tools.registry.bootstrap_builtins`). Callers that want the
registry pre-warmed can invoke ``bootstrap_builtins()`` explicitly.

Public surface
--------------
* :class:`ToolSpec`, :class:`ToolResult` — declarative metadata + outcome
* :class:`Capability` — the capability enum used by the gating layer
* :func:`tool` — decorator that registers a handler against a spec
* :func:`get`, :func:`mount`, :func:`register` — registry operations
* :func:`bootstrap_builtins` — explicit one-shot for the lazy bootstrap

Note: callers using ``from gemma.tools import registry`` continue to get
the submodule (not the snapshot helper) because we deliberately don't
re-export ``registry()``.
"""

from __future__ import annotations

from gemma.tools.capabilities import Capability  # noqa: F401
from gemma.tools.registry import (  # noqa: F401
    ToolResult,
    ToolSpec,
    bootstrap_builtins,
    get,
    mount,
    register,
    tool,
)
