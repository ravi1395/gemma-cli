"""Tests for the tool registry, ``@tool`` decorator, and mount filtering.

The registry is process-global; we use a small helper to register a
stub tool, exercise the API, then unregister so other tests are
unaffected.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from gemma.tools import registry as _r
from gemma.tools.capabilities import Capability, GatingContext
from gemma.tools.registry import ToolResult, ToolSpec


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

@contextmanager
def _temp_tool(spec: ToolSpec, handler):
    """Register a tool for the duration of a test, then clean up."""
    _r.register(spec, handler)
    try:
        yield
    finally:
        _r._unregister(spec.name)


def _ok_handler(**kwargs):  # pragma: no cover - trivial
    return ToolResult(ok=True, content=str(kwargs))


# ---------------------------------------------------------------------------
# Spec validation
# ---------------------------------------------------------------------------

def test_spec_rejects_non_identifier_names():
    with pytest.raises(ValueError, match="identifier"):
        ToolSpec(
            name="not a name",
            description="x",
            parameters={"type": "object"},
            capability=Capability.READ,
        )


def test_spec_requires_schema_type():
    with pytest.raises(ValueError, match="type"):
        ToolSpec(
            name="ok",
            description="x",
            parameters={"properties": {}},
            capability=Capability.READ,
        )


# ---------------------------------------------------------------------------
# Register / get / re-register
# ---------------------------------------------------------------------------

def test_register_then_get_returns_same_handler():
    spec = ToolSpec(
        name="reg_test_1",
        description="x",
        parameters={"type": "object"},
        capability=Capability.READ,
    )
    with _temp_tool(spec, _ok_handler):
        got_spec, got_handler = _r.get("reg_test_1")
        assert got_spec is spec
        assert got_handler is _ok_handler


def test_double_register_same_handler_is_idempotent():
    spec = ToolSpec(
        name="reg_test_2",
        description="x",
        parameters={"type": "object"},
        capability=Capability.READ,
    )
    with _temp_tool(spec, _ok_handler):
        # Re-registering the *same* handler under the same name must not raise.
        _r.register(spec, _ok_handler)


def test_double_register_different_handler_raises():
    spec = ToolSpec(
        name="reg_test_3",
        description="x",
        parameters={"type": "object"},
        capability=Capability.READ,
    )
    other = lambda **kw: ToolResult(ok=True)  # noqa: E731
    with _temp_tool(spec, _ok_handler):
        with pytest.raises(ValueError, match="already registered"):
            _r.register(spec, other)


def test_get_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        _r.get("__definitely_not_a_tool__")


# ---------------------------------------------------------------------------
# Mount filtering
# ---------------------------------------------------------------------------

def test_mount_only_shows_read_by_default():
    """With default GatingContext, WRITE/ARCHIVE tools must be hidden."""
    mounted = _r.mount(GatingContext())
    names = {s.name for s in mounted}
    # Built-in READ tools should be there.
    assert "read_file" in names
    assert "list_dir" in names
    # WRITE/ARCHIVE tools should not be.
    assert "write_file" not in names
    assert "archive_path" not in names


def test_mount_shows_write_when_allowed():
    ctx = GatingContext(allow_writes=True, is_tty=True)
    mounted = _r.mount(ctx)
    names = {s.name for s in mounted}
    assert "write_file" in names
    assert "archive_path" in names


def test_mount_hides_network_when_disabled():
    ctx = GatingContext(allow_network=False)
    mounted = _r.mount(ctx)
    names = {s.name for s in mounted}
    assert "http_get" not in names


def test_mount_order_is_alphabetical():
    mounted = _r.mount(GatingContext())
    names = [s.name for s in mounted]
    assert names == sorted(names)


# ---------------------------------------------------------------------------
# OpenAI schema serialisation
# ---------------------------------------------------------------------------

def test_as_openai_tools_shape():
    mounted = _r.mount(GatingContext())
    payload = _r.as_openai_tools(mounted[:1])
    assert payload[0]["type"] == "function"
    fn = payload[0]["function"]
    assert "name" in fn
    assert "description" in fn
    assert "parameters" in fn
