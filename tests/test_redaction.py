"""Tests for gemma.redaction.

Covers each supported pattern with a representative sample, confirms that
non-secret text passes through unchanged, and verifies the findings list
reports the correct type and original match.
"""

from __future__ import annotations

from gemma.redaction import contains_secret, redact


# ---------------------------------------------------------------------------
# Empty / no-op cases
# ---------------------------------------------------------------------------

def test_empty_string_returns_empty():
    clean, findings = redact("")
    assert clean == ""
    assert findings == []


def test_plain_text_unchanged():
    text = "Hello world, no secrets here. Just a normal sentence."
    clean, findings = redact(text)
    assert clean == text
    assert findings == []


def test_contains_secret_false_on_plain():
    assert contains_secret("nothing to see") is False
    assert contains_secret("") is False


# ---------------------------------------------------------------------------
# AWS access keys
# ---------------------------------------------------------------------------

def test_aws_access_key_akia():
    text = "My key is AKIAIOSFODNN7EXAMPLE in the logs."
    clean, findings = redact(text)
    assert "AKIAIOSFODNN7EXAMPLE" not in clean
    assert "[REDACTED:AWS_ACCESS_KEY]" in clean
    assert len(findings) == 1
    assert findings[0].type == "AWS_ACCESS_KEY"
    assert findings[0].match == "AKIAIOSFODNN7EXAMPLE"


def test_aws_access_key_asia():
    text = "session key ASIAIOSFODNN7EXAMPLE here"
    clean, findings = redact(text)
    assert "[REDACTED:AWS_ACCESS_KEY]" in clean
    assert findings[0].type == "AWS_ACCESS_KEY"


# ---------------------------------------------------------------------------
# GitHub / GitLab tokens
# ---------------------------------------------------------------------------

def test_github_personal_token():
    tok = "ghp_" + "a" * 36
    clean, findings = redact(f"token={tok} end")
    assert tok not in clean
    assert "[REDACTED:GITHUB_TOKEN]" in clean
    assert findings[0].type == "GITHUB_TOKEN"


def test_github_server_token():
    tok = "ghs_" + "X" * 40
    clean, findings = redact(tok)
    assert "[REDACTED:GITHUB_TOKEN]" in clean
    assert findings[0].type == "GITHUB_TOKEN"


def test_gitlab_token():
    tok = "glpat-" + "abcdefGHIJKL1234_-xy"
    clean, findings = redact(f"export GL={tok}")
    assert tok not in clean
    assert "[REDACTED:GITLAB_TOKEN]" in clean
    # Note: env-secret pattern may match on "GL=" too if name contained token words;
    # here the key is just GL so only the gitlab token pattern fires.
    assert any(f.type == "GITLAB_TOKEN" for f in findings)


# ---------------------------------------------------------------------------
# Bearer tokens
# ---------------------------------------------------------------------------

def test_bearer_token_preserves_word():
    text = "Authorization: Bearer abcdefghijklmnopqrstuvwxyz1234567890"
    clean, findings = redact(text)
    assert "Bearer [REDACTED:BEARER_TOKEN]" in clean
    assert "abcdefghij" not in clean
    assert findings[0].type == "BEARER_TOKEN"


def test_bearer_case_insensitive():
    text = "bearer abcdefghijklmnopqrstuvwxyz12345"
    clean, _ = redact(text)
    assert "[REDACTED:BEARER_TOKEN]" in clean


def test_bearer_short_token_not_matched():
    # Too-short token (<20 chars) should not trigger redaction
    text = "Bearer short"
    clean, findings = redact(text)
    assert clean == text
    assert findings == []


# ---------------------------------------------------------------------------
# JWTs
# ---------------------------------------------------------------------------

def test_jwt_three_segments():
    jwt = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4ifQ"
        ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    text = f"token: {jwt} (expires soon)"
    clean, findings = redact(text)
    assert jwt not in clean
    assert "[REDACTED:JWT]" in clean
    assert findings[0].type == "JWT"


# ---------------------------------------------------------------------------
# Private key envelopes
# ---------------------------------------------------------------------------

def test_private_key_block():
    key = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEpAIBAAKCAQEAtotallyfakekeyblobmaterialhere\n"
        "moremoremoremoremoremoremoremoremoremoremore\n"
        "-----END RSA PRIVATE KEY-----"
    )
    text = f"before\n{key}\nafter"
    clean, findings = redact(text)
    assert "fakekey" not in clean
    assert "[REDACTED:PRIVATE_KEY]" in clean
    assert "before" in clean and "after" in clean
    assert findings[0].type == "PRIVATE_KEY"


def test_openssh_private_key_block():
    key = (
        "-----BEGIN OPENSSH PRIVATE KEY-----\n"
        "b3BlbnNzaC1rZXktdjEAAAAABG5vbmU=\n"
        "-----END OPENSSH PRIVATE KEY-----"
    )
    clean, findings = redact(key)
    assert "[REDACTED:PRIVATE_KEY]" in clean
    assert findings[0].type == "PRIVATE_KEY"


def test_bare_private_key_block():
    key = (
        "-----BEGIN PRIVATE KEY-----\n"
        "MIIBVgIBADANBgkqhkiG9w0BAQEFAASCAUAwggE8\n"
        "-----END PRIVATE KEY-----"
    )
    clean, _ = redact(key)
    assert "[REDACTED:PRIVATE_KEY]" in clean


# ---------------------------------------------------------------------------
# .env-shaped lines
# ---------------------------------------------------------------------------

def test_env_secret_line():
    text = "API_TOKEN=supersecret123abcdef\nOTHER=value"
    clean, findings = redact(text)
    assert "supersecret123abcdef" not in clean
    assert "API_TOKEN=[REDACTED:ENV_SECRET]" in clean
    assert "OTHER=value" in clean
    assert any(f.type == "ENV_SECRET" for f in findings)


def test_env_password_line():
    text = "DB_PASSWORD=hunter2hunter2"
    clean, findings = redact(text)
    assert "hunter2hunter2" not in clean
    assert "DB_PASSWORD=[REDACTED:ENV_SECRET]" in clean
    assert findings[0].type == "ENV_SECRET"


def test_env_non_secret_key_untouched():
    text = "USERNAME=alice\nDEBUG=1"
    clean, findings = redact(text)
    assert clean == text
    assert findings == []


# ---------------------------------------------------------------------------
# Multiple secrets + ordering
# ---------------------------------------------------------------------------

def test_multiple_secrets_in_one_text():
    text = (
        "aws=AKIAIOSFODNN7EXAMPLE\n"
        "gh=ghp_" + "z" * 36 + "\n"
        "Authorization: Bearer abcdefghijklmnopqrstuvwxyz0123"
    )
    clean, findings = redact(text)
    types = {f.type for f in findings}
    assert {"AWS_ACCESS_KEY", "GITHUB_TOKEN", "BEARER_TOKEN"}.issubset(types)
    assert "AKIAIOSFODNN7EXAMPLE" not in clean
    assert "ghp_" not in clean
    assert "[REDACTED:AWS_ACCESS_KEY]" in clean
    assert "[REDACTED:GITHUB_TOKEN]" in clean
    assert "Bearer [REDACTED:BEARER_TOKEN]" in clean


def test_contains_secret_true():
    assert contains_secret("key AKIAIOSFODNN7EXAMPLE here") is True
    assert contains_secret("Bearer abcdefghijklmnopqrstuvwxyz1234") is True


# ---------------------------------------------------------------------------
# Finding metadata
# ---------------------------------------------------------------------------

def test_finding_preserves_original_match():
    secret = "AKIAIOSFODNN7EXAMPLE"
    _, findings = redact(f"key={secret}")
    assert findings[0].match == secret
    assert findings[0].type == "AWS_ACCESS_KEY"
