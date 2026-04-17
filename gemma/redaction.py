"""Secret redaction for conversation turns before long-term persistence.

Motivation
----------
Developers paste logs, stack traces, and configuration snippets into the CLI.
Any of those inputs might contain credentials (AWS keys, GitHub tokens, JWTs,
private-key blocks, ``.env`` lines). Once a turn is condensed into a Redis
MemoryRecord it can persist for days, so we scrub secrets out of turns
*before* they hit the store.

Scope
-----
This is not a DLP system. It targets a small set of common, high-confidence
patterns that are both cheap to match and genuinely dangerous if leaked.
False negatives are possible; false positives should be very rare because
every pattern requires a structural marker (known prefix, envelope, or
explicit key name).

API
---
``redact(text) -> (clean_text, findings)``
  * ``clean_text``: the input with every matched secret replaced by a
    ``[REDACTED:TYPE]`` marker. The surrounding context is preserved so the
    model (and anyone auditing memory) can still see *that* a secret existed,
    just not *what* it was.
  * ``findings``: a list of :class:`RedactionFinding` describing what was
    replaced. Only used for logging / testing -- the matched strings are
    never persisted to Redis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Pattern, Tuple


# ---------------------------------------------------------------------------
# Finding dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RedactionFinding:
    """A single secret pattern match in a piece of text.

    Attributes:
        type:  A stable label such as ``"AWS_ACCESS_KEY"`` identifying the
               pattern that matched. Used in tests and logging only.
        match: The original matched substring. Useful for local debugging --
               NEVER log or persist this downstream.
    """

    type: str
    match: str


# ---------------------------------------------------------------------------
# Pattern spec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _PatternSpec:
    """Internal description of a single redaction pattern."""

    type: str
    regex: Pattern[str]
    # Callable that builds the replacement string given the match.
    # Most patterns just return the literal "[REDACTED:TYPE]", but a few
    # preserve structural context (e.g. "Bearer [REDACTED:BEARER_TOKEN]"
    # keeps the word "Bearer" so auditors still see the header shape).
    replacement: Callable[[re.Match], str]


def _static(tag: str) -> Callable[[re.Match], str]:
    """Build a replacement function that always returns the given static marker."""
    marker = f"[REDACTED:{tag}]"
    def _repl(_m: re.Match) -> str:
        return marker
    return _repl


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------
#
# Order matters: more structural / envelope-like patterns run first so they
# consume their contents (e.g. a PRIVATE KEY block is swallowed whole before
# a JWT inside it could match on its own).

_PATTERNS: List[_PatternSpec] = [
    # -----------------------------------------------------------------
    # Envelope patterns (multiline, greedy within the envelope markers)
    # -----------------------------------------------------------------
    _PatternSpec(
        type="PRIVATE_KEY",
        # Matches RSA / OPENSSH / EC / DSA / bare "PRIVATE KEY" blocks.
        # DOTALL lets `.` span newlines so we capture the whole block.
        regex=re.compile(
            r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"
            r".*?"
            r"-----END (?:[A-Z0-9 ]+ )?PRIVATE KEY-----",
            re.DOTALL,
        ),
        replacement=_static("PRIVATE_KEY"),
    ),

    # -----------------------------------------------------------------
    # Authorization headers -- keep the "Bearer" word for context
    # -----------------------------------------------------------------
    _PatternSpec(
        type="BEARER_TOKEN",
        # "Bearer " followed by a typical token char set. 20+ chars to avoid
        # false-positives on short, non-credential identifiers.
        regex=re.compile(
            r"(?i)\bbearer\s+[A-Za-z0-9\-_=+\./]{20,}\b",
        ),
        replacement=lambda _m: "Bearer [REDACTED:BEARER_TOKEN]",
    ),

    # -----------------------------------------------------------------
    # .env-shaped lines whose key name implies secrecy
    # -----------------------------------------------------------------
    _PatternSpec(
        type="ENV_SECRET",
        # Captures KEY=VALUE when KEY contains one of the sensitive words.
        # The key name is preserved; the value is replaced.
        regex=re.compile(
            r"(?m)^(?P<key>[A-Z][A-Z0-9_]*"
            r"(?:TOKEN|SECRET|PASSWORD|PASSWD|PWD|APIKEY|API_KEY|ACCESS_KEY|PRIVATE_KEY)"
            r"[A-Z0-9_]*)"
            r"\s*=\s*"
            r"(?P<val>[^\s#]+)",
        ),
        replacement=lambda m: f"{m.group('key')}=[REDACTED:ENV_SECRET]",
    ),

    # -----------------------------------------------------------------
    # AWS credentials
    # -----------------------------------------------------------------
    _PatternSpec(
        type="AWS_ACCESS_KEY",
        # AKIA (long-lived) and ASIA (session) access key IDs are 20 chars.
        regex=re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
        replacement=_static("AWS_ACCESS_KEY"),
    ),

    # -----------------------------------------------------------------
    # GitHub / GitLab personal access tokens
    # -----------------------------------------------------------------
    _PatternSpec(
        type="GITHUB_TOKEN",
        # ghp_ (personal), gho_ (oauth), ghs_ (server-to-server),
        # ghu_ (user-to-server), ghr_ (refresh). All use 36+ base62 chars.
        regex=re.compile(r"\bgh[psoru]_[A-Za-z0-9]{36,}\b"),
        replacement=_static("GITHUB_TOKEN"),
    ),
    _PatternSpec(
        type="GITLAB_TOKEN",
        # glpat-<20+ url-safe chars>
        regex=re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}\b"),
        replacement=_static("GITLAB_TOKEN"),
    ),

    # -----------------------------------------------------------------
    # JWTs (three base64url segments separated by dots)
    # -----------------------------------------------------------------
    _PatternSpec(
        type="JWT",
        # Header starts with eyJ (the base64 of '{"'). Body must also start
        # with eyJ in standard JWTs. Third segment is the signature.
        regex=re.compile(
            r"\beyJ[A-Za-z0-9_\-]+\.eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b"
        ),
        replacement=_static("JWT"),
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def redact(text: str) -> Tuple[str, List[RedactionFinding]]:
    """Scrub secrets from ``text`` and report what was found.

    Args:
        text: Arbitrary user- or model-generated content.

    Returns:
        A tuple ``(clean_text, findings)`` where ``clean_text`` is safe to
        persist and ``findings`` lists every replacement performed.

    The function is pure: no I/O, no logging, no state. Callers that want to
    log findings should do so with the ``type`` field only -- the ``match``
    field intentionally contains the raw secret.
    """
    if not text:
        return text, []

    findings: List[RedactionFinding] = []
    clean = text

    for spec in _PATTERNS:
        # A closure over `spec` and `findings`, applied via re.sub.
        def _replace(m: re.Match, _spec: _PatternSpec = spec) -> str:
            findings.append(RedactionFinding(type=_spec.type, match=m.group(0)))
            return _spec.replacement(m)

        clean = spec.regex.sub(_replace, clean)

    return clean, findings


def contains_secret(text: str) -> bool:
    """Convenience predicate. True iff ``text`` contains any known secret pattern."""
    if not text:
        return False
    return any(spec.regex.search(text) for spec in _PATTERNS)
