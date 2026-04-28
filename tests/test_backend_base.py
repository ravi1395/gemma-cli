"""Unit tests for :mod:`gemma.backends.base`.

Covers ``parse_keep_alive_seconds`` — the keep-alive ↔ TTL mapping that
sits between the Ollama duration-string world and the LM Studio
seconds-int world. Getting this right is the only piece of arithmetic
shared by both backends, so a regression here breaks both.
"""

from __future__ import annotations

import pytest

from gemma.backends.base import parse_keep_alive_seconds


@pytest.mark.parametrize(
    "value, expected",
    [
        # Forever / no expiry sentinels
        (None, None),
        ("-1", None),
        ("forever", None),
        ("none", None),
        ("", None),
        (-1, None),
        # Evict immediately
        ("0", 0),
        ("0s", 0),
        (0, 0),
        # Suffixed durations
        ("30s", 30),
        ("2m", 120),
        ("30m", 1800),
        ("2h", 7200),
        ("1d", 86400),
        # Bare integer string
        ("45", 45),
        # Plain int / float
        (60, 60),
        (60.7, 60),
    ],
)
def test_parse_keep_alive_seconds_known_values(value, expected):
    assert parse_keep_alive_seconds(value) == expected


def test_parse_keep_alive_rejects_garbage():
    with pytest.raises(ValueError):
        parse_keep_alive_seconds("nonsense")
