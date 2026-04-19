"""Tests for the capability gating policy.

The policy is deliberately small and easy to cover exhaustively:
every cell of the Capability x GatingContext decision matrix has a
test here. If this file drifts from the implementation, new tools may
silently inherit the wrong default — keep it comprehensive.
"""

from __future__ import annotations

import pytest

from gemma.tools.capabilities import Capability, GatingContext, gate


# ---------------------------------------------------------------------------
# READ capability — always allowed, never prompts
# ---------------------------------------------------------------------------

def test_read_always_allowed_no_prompt():
    ctx = GatingContext(allow_writes=False, allow_network=False, is_tty=False)
    d = gate(Capability.READ, ctx)
    assert d.allowed is True
    assert d.requires_confirm is False


# ---------------------------------------------------------------------------
# NETWORK capability
# ---------------------------------------------------------------------------

def test_network_allowed_by_default():
    d = gate(Capability.NETWORK, GatingContext(allow_network=True))
    assert d.allowed is True
    assert d.requires_confirm is False


def test_network_refused_when_disabled():
    d = gate(Capability.NETWORK, GatingContext(allow_network=False))
    assert d.allowed is False
    assert "network" in d.reason.lower()


# ---------------------------------------------------------------------------
# WRITE / ARCHIVE capabilities
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cap", [Capability.WRITE, Capability.ARCHIVE])
def test_write_refused_without_flag(cap):
    ctx = GatingContext(allow_writes=False, is_tty=True)
    d = gate(cap, ctx)
    assert d.allowed is False
    assert "allow-writes" in d.reason


@pytest.mark.parametrize("cap", [Capability.WRITE, Capability.ARCHIVE])
def test_write_allowed_with_flag_but_requires_confirm_in_tty(cap):
    ctx = GatingContext(allow_writes=True, is_tty=True)
    d = gate(cap, ctx)
    assert d.allowed is True
    assert d.requires_confirm is True


@pytest.mark.parametrize("cap", [Capability.WRITE, Capability.ARCHIVE])
def test_write_refused_in_non_tty_without_auto_approve(cap):
    ctx = GatingContext(allow_writes=True, is_tty=False, auto_approve_writes=False)
    d = gate(cap, ctx)
    assert d.allowed is False
    assert "auto-approve" in d.reason


@pytest.mark.parametrize("cap", [Capability.WRITE, Capability.ARCHIVE])
def test_write_allowed_in_non_tty_with_auto_approve(cap):
    ctx = GatingContext(allow_writes=True, is_tty=False, auto_approve_writes=True)
    d = gate(cap, ctx)
    assert d.allowed is True
    # No interactive prompt possible in non-TTY — auto-approve subsumes it.
    assert d.requires_confirm is False
