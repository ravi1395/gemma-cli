"""Tests for :class:`Dispatcher` — budget, validation, gating, audit.

Each test either uses a built-in tool (preferred — exercises the real
registry) or registers a stub tool for the duration of the test.
Audit records are redirected to a tmp file via :func:`audit.set_log_path`.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Dict

import pytest

from gemma.tools import audit as _audit
from gemma.tools import registry as _r
from gemma.tools.capabilities import Capability, GatingContext
from gemma.tools.dispatcher import Dispatcher
from gemma.tools.registry import ToolResult, ToolSpec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_audit(tmp_path: Path):
    """Redirect the audit log to a tmp path for this test."""
    log = tmp_path / "audit.jsonl"
    _audit.set_log_path(log)
    yield log
    _audit.set_log_path(None)


@contextmanager
def _temp_tool(spec: ToolSpec, handler):
    _r.register(spec, handler)
    try:
        yield
    finally:
        _r._unregister(spec.name)


def _read_audit(path: Path):
    """Return parsed audit records from a tmp log."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Unknown tool
# ---------------------------------------------------------------------------

def test_unknown_tool_refused_and_audited(tmp_audit):
    disp = Dispatcher(ctx=GatingContext(), session_id="t")
    result = disp.dispatch("no_such_tool", {"x": 1})

    assert result.ok is False
    assert result.error == "unknown_tool"

    records = _read_audit(tmp_audit)
    assert len(records) == 1
    assert records[0]["tool"] == "no_such_tool"
    assert records[0]["approved_by"] == "refused"


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_missing_required_arg_rejected(tmp_audit):
    # read_file requires `path`.
    disp = Dispatcher(ctx=GatingContext(), session_id="t")
    result = disp.dispatch("read_file", {})
    assert result.ok is False
    assert result.error == "invalid_args"
    assert "path" in result.content


def test_wrong_arg_type_rejected(tmp_audit):
    disp = Dispatcher(ctx=GatingContext(), session_id="t")
    result = disp.dispatch("read_file", {"path": 42})
    assert result.ok is False
    assert result.error == "invalid_args"


def test_extra_arg_rejected_when_additional_properties_false(tmp_audit):
    disp = Dispatcher(ctx=GatingContext(), session_id="t")
    result = disp.dispatch("read_file", {"path": "x", "extra": "y"})
    assert result.ok is False
    assert result.error == "invalid_args"


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

def test_per_turn_budget_enforced(tmp_audit):
    spec = ToolSpec(
        name="budget_probe",
        description="x",
        parameters={"type": "object"},
        capability=Capability.READ,
    )

    def handler(**kw):
        return ToolResult(ok=True, content="hi")

    with _temp_tool(spec, handler):
        disp = Dispatcher(ctx=GatingContext(), session_id="t", budget=2)
        assert disp.dispatch("budget_probe", {}).ok is True
        assert disp.dispatch("budget_probe", {}).ok is True
        result = disp.dispatch("budget_probe", {})
        assert result.ok is False
        assert result.error == "budget_exhausted"


# ---------------------------------------------------------------------------
# Capability gating
# ---------------------------------------------------------------------------

def test_write_tool_refused_without_flag(tmp_audit, tmp_path):
    disp = Dispatcher(
        ctx=GatingContext(allow_writes=False, is_tty=True),
        session_id="t",
    )
    result = disp.dispatch("write_file", {"path": str(tmp_path / "x"), "content": "hi"})
    assert result.ok is False
    assert result.error == "denied"


def test_write_tool_prompts_and_declines(tmp_audit, tmp_path):
    decisions = []

    def confirm(spec, args):
        decisions.append(spec.name)
        return False  # user says no

    disp = Dispatcher(
        ctx=GatingContext(allow_writes=True, is_tty=True),
        confirm=confirm,
        session_id="t",
    )
    target = tmp_path / "x.txt"
    result = disp.dispatch("write_file", {"path": str(target), "content": "hi"})
    assert result.ok is False
    assert result.error == "declined"
    assert decisions == ["write_file"]
    assert not target.exists()


def test_write_tool_runs_when_user_confirms(tmp_audit, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    disp = Dispatcher(
        ctx=GatingContext(allow_writes=True, is_tty=True),
        confirm=lambda spec, args: True,
        session_id="t",
    )
    result = disp.dispatch("write_file", {"path": "new.txt", "content": "hi"})
    assert result.ok is True
    assert (tmp_path / "new.txt").read_text() == "hi"

    records = _read_audit(tmp_audit)
    assert records[-1]["approved_by"] == "user"
    assert records[-1]["tool"] == "write_file"


def test_handler_exception_becomes_failed_result(tmp_audit):
    spec = ToolSpec(
        name="explodes",
        description="x",
        parameters={"type": "object"},
        capability=Capability.READ,
    )

    def handler(**kw):
        raise RuntimeError("boom")

    with _temp_tool(spec, handler):
        disp = Dispatcher(ctx=GatingContext(), session_id="t")
        result = disp.dispatch("explodes", {})
        assert result.ok is False
        assert result.error == "handler_exception"
        assert "boom" in result.content
