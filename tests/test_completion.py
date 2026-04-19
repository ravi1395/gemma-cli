"""Tests for :mod:`gemma.completion` and :mod:`gemma.commands.completion`.

Covers:

* script generation via Click's completer (happy paths + unsupported shells)
* :func:`plan_install` for fresh-file, append, replace-existing cases
* idempotent re-install: running install twice leaves a single fenced block
* archive-on-rewrite: the prior rc content is preserved before any write
* ``--dry-run`` writes nothing
* fish completion uses a standalone file, not a fenced block
* :func:`inspect_installation` warns when zsh ``compinit`` is missing
* :func:`profile_completer` lists profile stems and is crash-safe
* The Typer CLI surface (install / print / status / uninstall) exits cleanly
"""

from __future__ import annotations

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from gemma import completion as _completion
from gemma import platform as _platform
from gemma.commands import completion as _cli


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def runner() -> CliRunner:
    # Newer Typer versions dropped ``mix_stderr`` (stderr is always
    # captured separately). Passing the kwarg fails, so we construct
    # with no args and rely on ``result.stdout`` / ``result.stderr``.
    return CliRunner()


@pytest.fixture
def tiny_app():
    """A minimal Typer app with the same wiring as the real gemma CLI.

    Using a stub here avoids depending on the full ``gemma.main`` import
    graph (which pulls in Redis / Ollama) and keeps the tests self-
    contained.
    """
    app = typer.Typer(help="Stub for completion tests.")

    @app.command()
    def hello(name: str = "world") -> None:  # pragma: no cover - stub
        typer.echo(f"hi {name}")

    return app


@pytest.fixture
def tiny_cli(tiny_app):
    """The Click command tree for the stub app — what :func:`generate_script` expects."""
    return typer.main.get_command(tiny_app)


# ---------------------------------------------------------------------------
# Script generation
# ---------------------------------------------------------------------------

class TestGenerateScript:
    def test_bash_script_looks_sane(self, tiny_cli):
        s = _completion.generate_script(
            _platform.Shell.BASH, prog_name="gemma", cli_command=tiny_cli,
        )
        assert "_gemma_completion" in s
        assert "_GEMMA_COMPLETE" in s

    def test_zsh_script_has_compdef(self, tiny_cli):
        s = _completion.generate_script(
            _platform.Shell.ZSH, prog_name="gemma", cli_command=tiny_cli,
        )
        assert "compdef" in s

    def test_fish_script_uses_complete_builtin(self, tiny_cli):
        s = _completion.generate_script(
            _platform.Shell.FISH, prog_name="gemma", cli_command=tiny_cli,
        )
        assert s.startswith("complete ")

    def test_unknown_shell_rejected(self, tiny_cli):
        with pytest.raises(ValueError):
            _completion.generate_script(
                _platform.Shell.UNKNOWN, cli_command=tiny_cli,
            )

    def test_powershell_rejected_with_hint(self, tiny_cli):
        with pytest.raises(ValueError, match="PowerShell"):
            _completion.generate_script(
                _platform.Shell.POWERSHELL, cli_command=tiny_cli,
            )


# ---------------------------------------------------------------------------
# Plan install
# ---------------------------------------------------------------------------

class TestPlanInstall:
    def test_fresh_rc_creates_block(self, tmp_path):
        rc = tmp_path / ".bashrc"
        plan = _completion.plan_install(
            _platform.Shell.BASH, rc_path=rc, script="echo hi\n",
        )
        assert plan.action == "create"
        assert not plan.existing_block
        assert _completion.FENCE_START in plan.new_content
        assert _completion.FENCE_END in plan.new_content

    def test_existing_rc_without_block_appends(self, tmp_path):
        rc = tmp_path / ".bashrc"
        rc.write_text("export PS1='> '\n")
        plan = _completion.plan_install(
            _platform.Shell.BASH, rc_path=rc, script="echo hi\n",
        )
        assert plan.action == "append"
        assert "export PS1" in plan.new_content
        assert _completion.FENCE_START in plan.new_content

    def test_existing_block_is_replaced(self, tmp_path):
        rc = tmp_path / ".zshrc"
        original = (
            "autoload -Uz compinit && compinit\n"
            + _completion.FENCE_START + "\n"
            + "# old content\n"
            + _completion.FENCE_END + "\n"
            + "alias ll='ls -la'\n"
        )
        rc.write_text(original)
        plan = _completion.plan_install(
            _platform.Shell.ZSH, rc_path=rc, script="new-block\n",
        )
        assert plan.action == "replace"
        assert plan.existing_block
        # Surrounding content preserved verbatim.
        assert "autoload -Uz compinit" in plan.new_content
        assert "alias ll='ls -la'" in plan.new_content
        # Old content gone, new content present.
        assert "# old content" not in plan.new_content
        assert "new-block" in plan.new_content
        # Exactly one block present.
        assert plan.new_content.count(_completion.FENCE_START) == 1

    def test_fish_uses_standalone_file(self, tmp_path):
        rc = tmp_path / "fish" / "completions" / "gemma.fish"
        plan = _completion.plan_install(
            _platform.Shell.FISH, rc_path=rc, script="complete --command gemma ...\n",
        )
        assert plan.action == "create"
        # Fish file has no fence — entire file is the script.
        assert _completion.FENCE_START not in plan.new_content
        assert plan.new_content.endswith("\n")

    def test_fish_same_script_is_noop(self, tmp_path):
        rc = tmp_path / "gemma.fish"
        script = "complete --command gemma ...\n"
        rc.write_text(script)
        plan = _completion.plan_install(
            _platform.Shell.FISH, rc_path=rc, script=script,
        )
        assert plan.action == "noop"


# ---------------------------------------------------------------------------
# Install / uninstall execution
# ---------------------------------------------------------------------------

class TestInstall:
    def test_install_creates_fresh_file(self, tmp_path):
        rc = tmp_path / ".bashrc"
        plan = _completion.install(
            _platform.Shell.BASH, rc_path=rc, script="echo hi\n",
        )
        assert plan.action == "create"
        assert rc.exists()
        assert _completion.FENCE_START in rc.read_text()

    def test_install_is_idempotent(self, tmp_path):
        rc = tmp_path / ".bashrc"
        _completion.install(
            _platform.Shell.BASH, rc_path=rc, script="echo v1\n",
        )
        _completion.install(
            _platform.Shell.BASH, rc_path=rc, script="echo v1\n",
        )
        content = rc.read_text()
        # Exactly one block, even after two installs of the same script.
        assert content.count(_completion.FENCE_START) == 1

    def test_install_replaces_existing_block(self, tmp_path):
        rc = tmp_path / ".bashrc"
        _completion.install(
            _platform.Shell.BASH, rc_path=rc, script="echo old\n",
        )
        _completion.install(
            _platform.Shell.BASH, rc_path=rc, script="echo new\n",
        )
        content = rc.read_text()
        assert "echo old" not in content
        assert "echo new" in content

    def test_install_archives_previous_rc(self, tmp_path):
        rc = tmp_path / ".bashrc"
        rc.write_text("original content\n")
        _completion.install(
            _platform.Shell.BASH, rc_path=rc, script="echo hi\n",
        )
        # The archive lands in <parent>/archive/<ts>/.bashrc.
        archive_root = tmp_path / "archive"
        assert archive_root.is_dir()
        archived = list(archive_root.rglob(".bashrc"))
        assert len(archived) == 1
        assert archived[0].read_text() == "original content\n"

    def test_dry_run_writes_nothing(self, tmp_path):
        rc = tmp_path / ".bashrc"
        plan = _completion.install(
            _platform.Shell.BASH, rc_path=rc,
            script="echo hi\n", dry_run=True,
        )
        assert plan.action == "create"
        assert not rc.exists()

    def test_atomic_write_cleans_up_tmp(self, tmp_path):
        rc = tmp_path / ".bashrc"
        _completion.install(
            _platform.Shell.BASH, rc_path=rc, script="echo hi\n",
        )
        # No leftover .gemma-tmp siblings.
        assert not any(p.name.endswith(".gemma-tmp") for p in tmp_path.iterdir())


class TestUninstall:
    def test_uninstall_removes_block_and_archives(self, tmp_path):
        rc = tmp_path / ".bashrc"
        _completion.install(
            _platform.Shell.BASH, rc_path=rc, script="echo hi\n",
        )
        content_after_install = rc.read_text()
        plan = _completion.uninstall(_platform.Shell.BASH, rc_path=rc)
        assert plan.action == "replace"
        new = rc.read_text()
        assert _completion.FENCE_START not in new

        # Pre-uninstall content is preserved as an inline archive copy
        # (or, for bashrc outside a gemma config dir, as a regular archive).
        archived = list(tmp_path.rglob(".bashrc*"))
        archived_contents = [p.read_text() for p in archived if p != rc]
        assert any(content_after_install in c for c in archived_contents)

    def test_uninstall_noop_when_no_block(self, tmp_path):
        rc = tmp_path / ".bashrc"
        rc.write_text("unrelated content\n")
        plan = _completion.uninstall(_platform.Shell.BASH, rc_path=rc)
        assert plan.action == "noop"
        assert rc.read_text() == "unrelated content\n"

    def test_uninstall_missing_rc_is_noop(self, tmp_path):
        rc = tmp_path / ".bashrc"
        plan = _completion.uninstall(_platform.Shell.BASH, rc_path=rc)
        assert plan.action == "noop"


# ---------------------------------------------------------------------------
# Installation inspection
# ---------------------------------------------------------------------------

class TestInspectInstallation:
    def test_reports_missing_rc(self, tmp_path):
        status = _completion.inspect_installation(
            _platform.Shell.BASH, rc_path=tmp_path / ".bashrc",
        )
        assert status.rc_exists is False
        assert status.block_present is False

    def test_reports_installed_block(self, tmp_path):
        rc = tmp_path / ".bashrc"
        _completion.install(
            _platform.Shell.BASH, rc_path=rc, script="echo hi\n",
        )
        status = _completion.inspect_installation(
            _platform.Shell.BASH, rc_path=rc,
        )
        assert status.rc_exists is True
        assert status.block_present is True
        assert status.warning is None

    def test_zsh_warns_when_compinit_missing(self, tmp_path):
        rc = tmp_path / ".zshrc"
        rc.write_text(
            _completion.FENCE_START + "\n"
            + "compdef _gemma gemma\n"
            + _completion.FENCE_END + "\n"
        )
        status = _completion.inspect_installation(
            _platform.Shell.ZSH, rc_path=rc,
        )
        assert status.block_present is True
        assert status.warning is not None
        assert "compinit" in status.warning

    def test_zsh_no_warning_when_compinit_present(self, tmp_path):
        rc = tmp_path / ".zshrc"
        rc.write_text(
            "autoload -Uz compinit && compinit\n"
            + _completion.FENCE_START + "\n"
            + "compdef _gemma gemma\n"
            + _completion.FENCE_END + "\n"
        )
        status = _completion.inspect_installation(
            _platform.Shell.ZSH, rc_path=rc,
        )
        assert status.warning is None


# ---------------------------------------------------------------------------
# Profile completer
# ---------------------------------------------------------------------------

class TestProfileCompleter:
    def test_returns_empty_when_dir_missing(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        # Force Path.home() to respect the new HOME.
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        assert _completion.profile_completer("") == []

    def test_lists_profile_stems(self, monkeypatch, tmp_path):
        profiles = tmp_path / ".config" / "gemma" / "profiles"
        profiles.mkdir(parents=True)
        (profiles / "work.toml").touch()
        (profiles / "personal.toml").touch()
        (profiles / "not-a-profile.txt").touch()
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        result = _completion.profile_completer("")
        assert set(result) == {"work", "personal"}

    def test_filters_by_incomplete_prefix(self, monkeypatch, tmp_path):
        profiles = tmp_path / ".config" / "gemma" / "profiles"
        profiles.mkdir(parents=True)
        (profiles / "work.toml").touch()
        (profiles / "personal.toml").touch()
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        assert _completion.profile_completer("wo") == ["work"]

    def test_swallows_unexpected_errors(self, monkeypatch):
        def _boom():
            raise RuntimeError("filesystem on fire")
        monkeypatch.setattr(Path, "home", _boom)
        # Must not raise — completion callbacks that throw break the shell.
        assert _completion.profile_completer("") == []


# ---------------------------------------------------------------------------
# Typer CLI surface
# ---------------------------------------------------------------------------

def _make_cli_app():
    """Build a Typer app that only exposes the completion subcommands."""
    app = typer.Typer()
    sub = typer.Typer()
    sub.command("install")(_cli.install_command)
    sub.command("print")(_cli.print_command)
    sub.command("status")(_cli.status_command)
    sub.command("uninstall")(_cli.uninstall_command)
    app.add_typer(sub, name="completion")
    return app


class TestCli:
    def test_install_dry_run_prints_content(self, runner, tmp_path, monkeypatch):
        # Point rc_file_for at our tmp path so we don't touch $HOME.
        monkeypatch.setattr(
            _platform, "rc_file_for",
            lambda shell, os_=None: tmp_path / ".bashrc",
        )
        result = runner.invoke(
            _make_cli_app(),
            ["completion", "install", "--shell", "bash", "--dry-run"],
        )
        assert result.exit_code == 0, result.stderr
        assert _completion.FENCE_START in result.stdout
        assert not (tmp_path / ".bashrc").exists()

    def test_print_emits_script(self, runner):
        result = runner.invoke(
            _make_cli_app(),
            ["completion", "print", "--shell", "bash"],
        )
        assert result.exit_code == 0, result.stderr
        assert "_GEMMA_COMPLETE" in result.stdout

    def test_print_rejects_unknown_shell(self, runner):
        result = runner.invoke(
            _make_cli_app(),
            ["completion", "print", "--shell", "tcsh"],
        )
        assert result.exit_code != 0

    def test_status_reports_not_installed(self, runner, tmp_path, monkeypatch):
        monkeypatch.setattr(
            _platform, "rc_file_for",
            lambda shell, os_=None: tmp_path / ".bashrc",
        )
        monkeypatch.setattr(_platform, "detect_shell", lambda: _platform.Shell.BASH)
        result = runner.invoke(
            _make_cli_app(),
            ["completion", "status"],
        )
        assert result.exit_code == 0, result.stderr
        assert "block installed" in result.stdout
