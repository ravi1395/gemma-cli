"""Unit tests for the JSON-file session history."""

import json

import pytest

from gemma.config import Config
from gemma.history import SessionHistory


@pytest.fixture
def cfg(tmp_path) -> Config:
    """A Config pointing at a temp history file."""
    return Config(history_file=str(tmp_path / "history.json"))


def test_load_empty_when_file_missing(cfg):
    history = SessionHistory(cfg)
    assert history.load() == []


def test_append_and_save_round_trip(cfg):
    history = SessionHistory(cfg)
    history.append("system", "sys prompt")
    history.append("user", "hi")
    history.append("assistant", "hello")
    history.save()

    # Re-open and confirm persistence
    reloaded = SessionHistory(cfg).load()
    assert reloaded == [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_show_returns_copy_not_reference(cfg):
    history = SessionHistory(cfg)
    history.append("user", "a")
    view = history.show()
    view.append({"role": "user", "content": "MUTATION"})
    assert history.turns[-1]["content"] == "a"


def test_clear_removes_file(cfg):
    history = SessionHistory(cfg)
    history.append("user", "x")
    history.save()
    assert cfg.resolved_history_path().exists()

    history.clear()
    assert not cfg.resolved_history_path().exists()
    assert history.turns == []


def test_corrupt_file_falls_back_to_empty(cfg):
    path = cfg.resolved_history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json")

    history = SessionHistory(cfg)
    assert history.load() == []


def test_lazy_load_on_turns_access(cfg):
    # Pre-populate the file
    path = cfg.resolved_history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{"role": "user", "content": "pre"}]))

    history = SessionHistory(cfg)
    # Accessing turns should auto-load
    assert history.turns == [{"role": "user", "content": "pre"}]
