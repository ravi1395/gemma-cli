"""Capability enum and gating policy for the tool registry.

Every tool declares exactly one :class:`Capability`. The dispatcher
uses that capability to decide whether a given call is allowed under
the current CLI flags and — for write-side-effects — whether an
interactive confirmation is required.

Keeping the gating policy in one tiny module means the answer to
"can Gemma do X right now?" lives in one place, not scattered across
N tools. Adding a new capability (say, ``EXEC`` for running arbitrary
scripts) means writing one enum member + one row in :func:`gate`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Capability(str, Enum):
    """Side-effect class a tool belongs to.

    Values are deliberately strings so they serialise cleanly into the
    audit log and into ``gemma tools list`` output. The ``str, Enum``
    base lets you pass a capability where a plain string is expected
    (e.g. JSON dumps) without extra casts.
    """

    #: Pure read: file content, directory listings, linter/test output,
    #: HTTP GET against an allowlisted host. No observable side-effects
    #: on the local filesystem.
    READ = "read"

    #: Outbound network access. Separate from READ so the user can
    #: disable network independently of file reads (e.g. on a flight).
    NETWORK = "network"

    #: Create or overwrite a file. Never deletes — the never-delete
    #: rule is enforced by *not registering any tool that unlinks*.
    WRITE = "write"

    #: Move a file into the archive directory via
    #: :func:`gemma.safety.archive`. Semantically a "soft delete": the
    #: file is relocated, never unlinked.
    ARCHIVE = "archive"


# ---------------------------------------------------------------------------
# Gating policy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GatingContext:
    """Snapshot of the CLI flags that influence tool-call admission.

    Attributes:
        allow_writes: User passed ``--allow-writes``. Required to mount
            any WRITE or ARCHIVE tool.
        allow_network: User passed ``--allow-network``. Required to
            mount NETWORK tools. Default policy is allow — network
            tools are *mounted by default* but can be disabled with
            ``--no-network``.
        is_tty: stdin/stdout is interactive. Required for WRITE/ARCHIVE
            calls unless ``auto_approve_writes`` is set.
        auto_approve_writes: User passed ``--auto-approve-writes``
            (only meaningful in non-TTY mode — pipelines).
    """

    allow_writes: bool = False
    allow_network: bool = True
    is_tty: bool = True
    auto_approve_writes: bool = False


@dataclass(frozen=True)
class GateDecision:
    """Result of :func:`gate`.

    Attributes:
        allowed: The call may proceed if True; it must be refused if
            False.
        requires_confirm: When True, the dispatcher must prompt the
            user for y/N before executing. Never true for READ /
            NETWORK, always true for WRITE / ARCHIVE unless
            ``auto_approve_writes`` is set.
        reason: Human-readable explanation — always set on denial,
            optional on acceptance.
    """

    allowed: bool
    requires_confirm: bool
    reason: str = ""


def gate(capability: Capability, ctx: GatingContext) -> GateDecision:
    """Decide whether a call with ``capability`` is admissible under ``ctx``.

    Pure function — no globals, no side effects — so tests can cover
    every combination cheaply.

    Policy summary:

    ============  =================  ==========================================
    Capability    Default mount      Per-call rule
    ============  =================  ==========================================
    READ          yes                always allowed, never prompts
    NETWORK       yes                allowed iff ``allow_network``
    WRITE         no                 requires ``allow_writes`` + y/N
                                     (or auto-approve in non-TTY)
    ARCHIVE       no                 same as WRITE
    ============  =================  ==========================================
    """
    if capability is Capability.READ:
        return GateDecision(allowed=True, requires_confirm=False)

    if capability is Capability.NETWORK:
        if not ctx.allow_network:
            return GateDecision(
                allowed=False, requires_confirm=False,
                reason="network tools disabled (pass --allow-network)",
            )
        return GateDecision(allowed=True, requires_confirm=False)

    if capability in (Capability.WRITE, Capability.ARCHIVE):
        if not ctx.allow_writes:
            return GateDecision(
                allowed=False, requires_confirm=False,
                reason="write-capability tools require --allow-writes",
            )
        if not ctx.is_tty and not ctx.auto_approve_writes:
            return GateDecision(
                allowed=False, requires_confirm=False,
                reason=(
                    "non-interactive session: pass --auto-approve-writes to "
                    "run write-capability tools without a prompt"
                ),
            )
        # TTY present (or auto-approve set). The dispatcher must still
        # confirm interactively unless the caller pre-approved.
        return GateDecision(
            allowed=True,
            requires_confirm=not ctx.auto_approve_writes,
        )

    # Defensive — unknown capability should be impossible thanks to the
    # Enum, but fail-closed just in case.
    return GateDecision(
        allowed=False, requires_confirm=False,
        reason=f"unknown capability {capability!r}",
    )
