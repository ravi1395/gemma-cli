"""Dispatch model-issued tool calls into the registered handlers.

The :class:`Dispatcher` is the single entry point that glues together
everything else in ``gemma.tools``:

* consults the :mod:`registry` to look up each tool,
* validates incoming args against the declared JSON schema,
* runs the :mod:`capabilities` gate to decide admission,
* prompts the user (or auto-approves) for WRITE/ARCHIVE tools,
* invokes the handler,
* appends an :mod:`audit` record,
* enforces the per-turn call budget.

The class is deliberately boring. All interesting decisions live in
the helper modules so the dispatcher itself can be covered by a few
high-level integration tests.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from gemma.tools import audit as _audit
from gemma.tools import registry as _registry
from gemma.tools.capabilities import Capability, GatingContext, gate
from gemma.tools.registry import ToolResult, ToolSpec


#: Callback the dispatcher uses to obtain interactive y/N confirmation.
#: Returning True means "proceed". The CLI provides a Rich-based
#: implementation; tests pass a lambda.
ConfirmCallback = Callable[[ToolSpec, Dict[str, Any]], bool]


@dataclass
class Dispatcher:
    """Mediates one conversational turn's worth of tool calls.

    Attributes:
        ctx: Gating context describing the CLI flags in force.
        session_id: Opaque identifier threaded into every audit record.
        confirm: Callback invoked for WRITE/ARCHIVE tools. Defaults to
            "deny everything" — safe default; real callers must
            provide a real prompt.
        budget: Max number of tool calls per turn. Once exhausted the
            dispatcher returns a budget-exhausted message and the
            caller should stop looping.
    """

    ctx: GatingContext
    session_id: str = "default"
    confirm: ConfirmCallback = field(
        default_factory=lambda: (lambda spec, args: False)
    )
    budget: int = 8

    # Mutable counters — dataclass default_factory gives each instance
    # its own so two concurrent dispatchers don't clobber each other.
    calls_made: int = 0

    # --------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------

    def mount_specs(self) -> List[ToolSpec]:
        """Return the tool specs advertised to the model under ``ctx``."""
        return _registry.mount(self.ctx)

    def advertised_schemas(self) -> List[Dict[str, Any]]:
        """Return the Ollama ``tools=[…]`` payload for the current mount."""
        return _registry.as_openai_tools(self.mount_specs())

    def dispatch(self, name: str, args: Dict[str, Any]) -> ToolResult:
        """Handle one tool call end-to-end.

        Always returns a :class:`ToolResult`: exceptions from the
        handler become ``ok=False`` results so the model can react
        conversationally rather than the CLI crashing out.
        """
        # --- Budget check (before anything else so runaway loops stop early).
        if self.calls_made >= self.budget:
            return self._refuse(
                name, args, reason="budget exhausted",
                error="budget_exhausted",
            )
        self.calls_made += 1

        # --- Lookup.
        try:
            spec, handler = _registry.get(name)
        except KeyError:
            return self._refuse(
                name, args, reason=f"unknown tool {name!r}",
                error="unknown_tool",
            )

        # --- Admission gate.
        decision = gate(spec.capability, self.ctx)
        if not decision.allowed:
            return self._refuse(
                name, args, reason=decision.reason,
                error="denied", spec=spec,
            )

        # --- Schema validation.
        valid, err = _validate_against_schema(args, spec.parameters)
        if not valid:
            return self._refuse(
                name, args, reason=err, error="invalid_args", spec=spec,
            )

        # --- Interactive confirmation for WRITE/ARCHIVE.
        approved_by = "auto"
        if decision.requires_confirm:
            if not self.confirm(spec, args):
                return self._refuse(
                    name, args,
                    reason="user declined confirmation",
                    error="declined",
                    spec=spec,
                    approved_by="refused",
                )
            approved_by = "user"
        elif spec.capability in (Capability.WRITE, Capability.ARCHIVE):
            # allow_writes + auto_approve_writes path.
            approved_by = "flag"

        # --- Execute.
        started = time.monotonic()
        try:
            result = handler(**args)
        except Exception as exc:  # pragma: no cover - defensive
            # Handlers shouldn't raise, but if one does, never let it
            # crash the CLI mid-turn.
            duration_ms = int((time.monotonic() - started) * 1000)
            _audit.append(_audit.make_record(
                tool=name,
                capability=spec.capability.value,
                args=args,
                session_id=self.session_id,
                exit_code=1,
                duration_ms=duration_ms,
                approved_by=approved_by,
                refusal_reason=f"handler raised {type(exc).__name__}: {exc}",
            ))
            return ToolResult(
                ok=False,
                error="handler_exception",
                content=f"tool {name} raised {type(exc).__name__}: {exc}",
            )

        duration_ms = int((time.monotonic() - started) * 1000)

        # --- Audit.
        _audit.append(_audit.make_record(
            tool=name,
            capability=spec.capability.value,
            args=args,
            session_id=self.session_id,
            exit_code=0 if result.ok else 1,
            duration_ms=duration_ms,
            paths_touched=[
                _audit.PathDigest(**p)
                for p in result.metadata.get("paths_touched", [])
            ],
            approved_by=approved_by,
            refusal_reason=None if result.ok else result.error,
        ))

        return result

    # --------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------

    def _refuse(
        self,
        name: str,
        args: Dict[str, Any],
        *,
        reason: str,
        error: str,
        spec: Optional[ToolSpec] = None,
        approved_by: str = "refused",
    ) -> ToolResult:
        """Audit a refusal and return a structured error ToolResult.

        Refusals *are* audited so a post-hoc reviewer can see what the
        model tried to do and what stopped it — often more interesting
        than successful calls.
        """
        capability = spec.capability.value if spec is not None else "unknown"
        _audit.append(_audit.make_record(
            tool=name,
            capability=capability,
            args=args,
            session_id=self.session_id,
            exit_code=1,
            duration_ms=0,
            approved_by=approved_by,
            refusal_reason=reason,
        ))
        return ToolResult(ok=False, error=error, content=reason)


# ---------------------------------------------------------------------------
# JSON-schema validation
# ---------------------------------------------------------------------------
#
# We intentionally ship a tiny hand-rolled validator rather than pull in
# ``jsonschema``. Our schemas are small and use a predictable subset of
# Draft-7 (object/string/integer/boolean/array of primitives). A full
# validator would cost an extra dependency for no gain.

_JSON_TYPE_MAP: Dict[str, Tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}


def _validate_against_schema(
    value: Any, schema: Dict[str, Any]
) -> Tuple[bool, str]:
    """Minimal JSON-schema check.

    Supports: ``type`` (scalar or array-of-types), ``required``,
    ``properties`` (recursively), ``additionalProperties``, ``items``
    (for arrays), and ``enum``. Unknown keywords are ignored rather
    than erroring — a schema that goes beyond our subset will be
    under-enforced but not rejected outright.
    """
    expected = schema.get("type")
    if expected is not None and not _type_matches(value, expected):
        return False, f"expected type {expected!r}, got {type(value).__name__}"

    if "enum" in schema and value not in schema["enum"]:
        return False, f"value {value!r} not in enum {schema['enum']}"

    if expected == "object":
        assert isinstance(value, dict)  # narrowed by _type_matches
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                return False, f"missing required property {key!r}"

        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, sub in value.items():
            if key in properties:
                ok, err = _validate_against_schema(sub, properties[key])
                if not ok:
                    return False, f"property {key!r}: {err}"
            elif additional is False:
                return False, f"unexpected property {key!r}"

    if expected == "array" and "items" in schema:
        assert isinstance(value, list)
        for i, item in enumerate(value):
            ok, err = _validate_against_schema(item, schema["items"])
            if not ok:
                return False, f"item [{i}]: {err}"

    return True, ""


def _type_matches(value: Any, expected: Any) -> bool:
    """Return True if ``value`` matches the schema ``type`` keyword.

    ``expected`` may be a single type string or a list of them.
    Booleans are tricky in Python: ``isinstance(True, int)`` is True,
    so we explicitly reject bools for the ``integer``/``number``
    cases to match JSON-schema semantics.
    """
    names = expected if isinstance(expected, list) else [expected]
    for name in names:
        allowed = _JSON_TYPE_MAP.get(name, ())
        if allowed and isinstance(value, allowed):
            if name in ("integer", "number") and isinstance(value, bool):
                continue
            return True
    return False
