"""Tool registry, spec container, and the ``@tool`` decorator.

The registry is a process-global mapping of ``name -> (ToolSpec,
handler)``. Tools register themselves at import time via the
:func:`tool` decorator. The dispatcher consults the registry at
mount time to decide which tools to advertise to the model, and at
call time to look up the handler for an incoming tool call.

Design choices
--------------

* **One registry per process, not per dispatcher.** Tools are
  essentially static code; having the registry be a module-level dict
  (rather than an instance hung off a dispatcher) makes testing much
  simpler — tests construct a :class:`GatingContext`, call
  :func:`mount`, and assert on the filtered list.
* **Mount is a filter, not a registration step.** Everything
  ``@tool``-decorated is in the registry unconditionally. Mounting is
  what decides which of those to *expose* under the current CLI flags.
  That separation keeps ``@tool`` code pure: a decorator can't care
  about runtime state it has no access to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from gemma.tools.capabilities import Capability, GatingContext, gate


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

#: Signature every tool handler must implement.
#:
#: The handler receives pre-validated kwargs (schema already enforced
#: by the dispatcher) and returns a :class:`ToolResult`. Handlers must
#: be idempotent under identical input where practical — the dispatcher
#: may retry on transient failures in a future version.
ToolHandler = Callable[..., "ToolResult"]


@dataclass(frozen=True)
class ToolSpec:
    """Declarative metadata describing a tool to the model and the dispatcher.

    Attributes:
        name: Stable identifier used in both the model's tool-call
            syntax and the audit log. Must be a valid Python identifier
            (enforced) so it can also appear unambiguously in error
            messages.
        description: One-line, model-readable summary. Shown to Gemma
            when advertising tools.
        parameters: OpenAI-style JSON schema for the tool's arguments.
        capability: Which :class:`Capability` this tool embodies.
        requires_confirm: Hint to the dispatcher. Some READ tools that
            still touch sensitive state may set this to True; most do
            not. WRITE/ARCHIVE always prompt regardless of this flag.
        timeout_s: Default wall-clock timeout for subprocess tools.
            Pure-Python tools ignore it.
        max_output_bytes: Default output cap for subprocess tools.
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    capability: Capability
    requires_confirm: bool = False
    timeout_s: int = 30
    max_output_bytes: int = 256 * 1024

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(
                f"tool name {self.name!r} is not a valid identifier"
            )
        if "type" not in self.parameters:
            # OpenAI schemas must have a top-level "type": "object"; we
            # catch the common mistake here rather than letting Ollama
            # reject it at call time.
            raise ValueError(
                f"tool {self.name!r} parameters schema missing 'type' key"
            )


@dataclass(frozen=True)
class ToolResult:
    """Structured outcome returned by every tool handler.

    Attributes:
        ok: Success flag. When False, :attr:`error` is set and
            :attr:`content` may or may not be partial.
        content: The payload returned to the model. Always a string
            (the Ollama tool-message format expects text). Structured
            results should be JSON-serialised here by the handler.
        error: Short machine-readable error code when ``ok=False``
            (e.g. ``"timeout"``, ``"denied"``, ``"invalid_args"``).
        metadata: Free-form diagnostics the dispatcher may fold into
            the audit record. Kept separate from ``content`` because
            the model shouldn't see raw sha256s or durations.
    """

    ok: bool
    content: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registry storage
# ---------------------------------------------------------------------------

# Intentionally module-level: tools self-register via ``@tool`` at import
# time, which is only meaningful against a singleton.
_registry: Dict[str, Tuple[ToolSpec, ToolHandler]] = {}


def register(spec: ToolSpec, handler: ToolHandler) -> None:
    """Insert or replace ``spec``/``handler`` in the registry.

    Raises:
        ValueError: If a different handler is already registered under
            the same name (deterministic: prevents accidental shadowing
            when two modules register the same name).
    """
    existing = _registry.get(spec.name)
    if existing is not None and existing[1] is not handler:
        raise ValueError(
            f"tool {spec.name!r} is already registered by "
            f"{existing[1].__module__}.{existing[1].__qualname__}"
        )
    _registry[spec.name] = (spec, handler)


def _unregister(name: str) -> None:
    """Remove a tool from the registry.

    Not public API — tests and hot-reload helpers only. The leading
    underscore is load-bearing: removing a tool at runtime is almost
    always a bug in anything but a test.
    """
    _registry.pop(name, None)


def get(name: str) -> Tuple[ToolSpec, ToolHandler]:
    """Return the ``(spec, handler)`` for ``name``.

    Raises:
        KeyError: Unknown tool. The dispatcher translates this into a
            structured error returned to the model.
    """
    if name not in _registry:
        raise KeyError(name)
    return _registry[name]


def registry() -> Dict[str, Tuple[ToolSpec, ToolHandler]]:
    """Return a shallow copy of the full registry.

    Copy so callers iterating the result can't see a later
    modification mid-iteration if a tool is registered concurrently
    (unlikely but cheap to prevent).
    """
    return dict(_registry)


# ---------------------------------------------------------------------------
# @tool decorator
# ---------------------------------------------------------------------------

def tool(spec: ToolSpec) -> Callable[[ToolHandler], ToolHandler]:
    """Decorator that registers a handler under ``spec``.

    Usage::

        @tool(ToolSpec(
            name="read_file",
            description="Return the contents of a file.",
            parameters={"type": "object", ...},
            capability=Capability.READ,
        ))
        def read_file(path: str) -> ToolResult:
            ...

    The decorated function is returned unchanged so it can still be
    called directly from Python (handy for unit tests).
    """
    def decorator(fn: ToolHandler) -> ToolHandler:
        register(spec, fn)
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Mount
# ---------------------------------------------------------------------------

def mount(ctx: GatingContext) -> List[ToolSpec]:
    """Return the list of :class:`ToolSpec` visible to the model under ``ctx``.

    Filtering is purely by capability admission — tools whose
    capability is refused by :func:`capabilities.gate` are not
    advertised. Per-call gating (interactive y/N for WRITE tools) still
    runs inside the dispatcher; mount only controls *visibility*.

    Sort is alphabetical so the advertised list is deterministic,
    which keeps the model's view stable across runs and makes
    snapshot-style tests trivial.
    """
    visible: List[ToolSpec] = []
    for name in sorted(_registry):
        spec, _handler = _registry[name]
        if gate(spec.capability, ctx).allowed:
            visible.append(spec)
    return visible


def all_specs() -> List[ToolSpec]:
    """Return every registered :class:`ToolSpec`, regardless of gating.

    Used by ``gemma tools list`` to show both mounted and unmounted
    tools so users can see *why* a tool is unavailable.
    """
    return [spec for spec, _ in (_registry[n] for n in sorted(_registry))]


def as_openai_tools(specs: Iterable[ToolSpec]) -> List[Dict[str, Any]]:
    """Serialise specs to Ollama's OpenAI-compatible ``tools=[…]`` form.

    Ollama 0.4+ accepts the format::

        {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}

    This helper is shape-only — no capability filtering here; callers
    should pass the output of :func:`mount`.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": s.name,
                "description": s.description,
                "parameters": s.parameters,
            },
        }
        for s in specs
    ]
