"""Unit tests for ``gemma model {pull,list,use,info}``.

The ``lms`` CLI and the ``lmstudio`` SDK are mocked so these tests run
without touching a real LM Studio install. Coverage focuses on the
user-facing surface: exit codes, the TOML upsert in ``use``, and the
graceful error paths when the runtime is missing.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from gemma.commands.model import _upsert_toml_field
from gemma.main import app


runner = CliRunner()


# ---------------------------------------------------------------------------
# pull
# ---------------------------------------------------------------------------


def test_pull_errors_when_lms_missing(monkeypatch):
    """Without ``lms`` on PATH, pull exits 1 and prints install hint."""
    monkeypatch.setattr("shutil.which", lambda name: None)
    result = runner.invoke(app, ["model", "pull", "owner/repo"])
    assert result.exit_code == 1
    assert "lms" in result.output
    assert "install-cli" in result.output


def test_pull_invokes_lms_get(monkeypatch):
    """When ``lms`` is present we shell out to ``lms get <repo>``."""
    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/lms")
    fake_run = MagicMock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr("subprocess.run", fake_run)

    result = runner.invoke(app, ["model", "pull", "mlx-community/foo"])
    assert result.exit_code == 0, result.output
    fake_run.assert_called_once_with(["lms", "get", "mlx-community/foo"])


def test_pull_propagates_lms_failure(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/lms")
    monkeypatch.setattr(
        "subprocess.run",
        MagicMock(return_value=SimpleNamespace(returncode=2)),
    )
    result = runner.invoke(app, ["model", "pull", "owner/repo"])
    assert result.exit_code == 2


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def test_list_loaded_models_renders_table(monkeypatch):
    fake_models = [
        SimpleNamespace(model_key="mlx-community/foo", type="llm", size_bytes=4_200_000_000),
        SimpleNamespace(model_key="bar/embed", type="embedding", size_bytes=300_000_000),
    ]
    with patch(
        "lmstudio.list_loaded_models",
        return_value=fake_models,
    ):
        result = runner.invoke(app, ["model", "list", "--loaded"])
    assert result.exit_code == 0, result.output
    assert "mlx-community/foo" in result.output
    assert "bar/embed" in result.output


def test_list_handles_no_models(monkeypatch):
    with patch("lmstudio.list_loaded_models", return_value=[]):
        result = runner.invoke(app, ["model", "list"])
    assert result.exit_code == 0
    assert "No loaded models" in result.output


def test_list_handles_runtime_unavailable(monkeypatch):
    with patch(
        "lmstudio.list_loaded_models",
        side_effect=RuntimeError("connection refused"),
    ):
        result = runner.invoke(app, ["model", "list"])
    assert result.exit_code == 1
    assert "LM Studio" in result.output


# ---------------------------------------------------------------------------
# use
# ---------------------------------------------------------------------------


def test_use_creates_profile_with_model_field(tmp_path, monkeypatch):
    """``model use <key> -p <name>`` writes a fresh profile."""
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    result = runner.invoke(
        app,
        ["model", "use", "mlx-community/x", "-p", "myprofile"],
    )
    assert result.exit_code == 0, result.output
    profile = tmp_path / ".config" / "gemma" / "profiles" / "myprofile.toml"
    assert profile.exists()
    text = profile.read_text()
    assert 'model = "mlx-community/x"' in text


def test_use_overwrites_existing_model_field(tmp_path, monkeypatch):
    """When the profile already pins a model, ``use`` replaces in place."""
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    profile = tmp_path / ".config" / "gemma" / "profiles" / "p.toml"
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text('model = "old/model"\ntemperature = 0.5\n')

    result = runner.invoke(app, ["model", "use", "new/model", "-p", "p"])
    assert result.exit_code == 0, result.output
    text = profile.read_text()
    assert 'model = "new/model"' in text
    assert 'model = "old/model"' not in text
    # Other fields are preserved.
    assert "temperature = 0.5" in text


# ---------------------------------------------------------------------------
# upsert helper
# ---------------------------------------------------------------------------


def test_upsert_inserts_when_absent():
    out = _upsert_toml_field("temperature = 0.5\n", "model", "x/y")
    assert 'model = "x/y"' in out
    assert "temperature = 0.5" in out


def test_upsert_replaces_when_present():
    out = _upsert_toml_field('model = "old"\nx = 1\n', "model", "new")
    assert 'model = "new"' in out
    assert 'model = "old"' not in out
    assert "x = 1" in out


def test_upsert_handles_empty_input():
    out = _upsert_toml_field("", "model", "v")
    assert out.strip() == 'model = "v"'


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


def test_info_prints_resolved_config():
    result = runner.invoke(app, ["model", "info"])
    assert result.exit_code == 0, result.output
    assert "backend" in result.output
    assert "chat model" in result.output
    assert "embedding model" in result.output
