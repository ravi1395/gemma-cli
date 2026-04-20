"""Tests for ``gemma sh``, ``gemma why``, and ``gemma install-shell``.

All tests use ``typer.testing.CliRunner`` and mock ``client.chat`` so no
Ollama server is required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from gemma.main import app


runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_chat(messages, cfg, stream=True):
    """Stub that yields a single ("content", ...) tuple."""
    yield ("content", "find . -name '*.py' -mtime -1")


def _fake_chat_explain(messages, cfg, stream=True):
    """Stub that yields a simple explanation string."""
    yield ("content", "This finds Python files modified in the last day.")


def _fake_chat_why(messages, cfg, stream=True):
    """Stub that yields a why explanation."""
    yield ("content", "The command failed because the path does not exist.")


# ---------------------------------------------------------------------------
# gemma sh
# ---------------------------------------------------------------------------

class TestShCommand:
    def test_no_exec_prints_command(self):
        """`--no-exec` should print the model output to stdout without prompting."""
        with patch("gemma.commands.shell.client_chat", side_effect=_fake_chat):
            result = runner.invoke(app, ["sh", "--no-exec", "find python files modified today"])
        assert result.exit_code == 0
        assert "find" in result.output

    def test_no_exec_strips_fences(self):
        """Model output wrapped in fences must be stripped."""
        def _fenced(messages, cfg, stream=True):
            yield ("content", "```bash\nfind . -name '*.py'\n```")

        with patch("gemma.commands.shell.client_chat", side_effect=_fenced):
            result = runner.invoke(app, ["sh", "--no-exec", "find python files"])
        assert result.exit_code == 0
        assert "```" not in result.output
        assert "find" in result.output

    def test_no_exec_with_explain_flag(self):
        """``--explain`` is forwarded to the model prompt; output passes through."""
        def _with_comment(messages, cfg, stream=True):
            yield ("content", "# finds recently modified Python files\nfind . -name '*.py' -mtime -1")

        with patch("gemma.commands.shell.client_chat", side_effect=_with_comment):
            result = runner.invoke(app, ["sh", "--no-exec", "--explain", "find python files"])
        assert result.exit_code == 0
        # Comment line and command should both appear
        assert "#" in result.output
        assert "find" in result.output

    def test_empty_model_response_exits_nonzero(self):
        def _empty(messages, cfg, stream=True):
            yield ("content", "   ")

        with patch("gemma.commands.shell.client_chat", side_effect=_empty):
            result = runner.invoke(app, ["sh", "--no-exec", "do something"])
        assert result.exit_code != 0

    def test_model_error_exits_nonzero(self):
        def _boom(messages, cfg, stream=True):
            raise RuntimeError("connection refused")
            yield  # make it a generator

        with patch("gemma.commands.shell.client_chat", side_effect=_boom):
            result = runner.invoke(app, ["sh", "--no-exec", "--no-cache", "list files"])
        assert result.exit_code != 0

    def test_missing_prompt_argument_exits_nonzero(self):
        result = runner.invoke(app, ["sh"])
        assert result.exit_code != 0

    def test_shell_flag_accepted(self):
        """``--shell zsh`` is accepted and passed through without error."""
        with patch("gemma.commands.shell.client_chat", side_effect=_fake_chat):
            result = runner.invoke(app, ["sh", "--no-exec", "--shell", "zsh", "list files"])
        assert result.exit_code == 0

    def test_keep_alive_flag_accepted(self):
        """``--keep-alive`` is accepted and applied to config."""
        captured = {}

        def _capture(messages, cfg, stream=True):
            captured["keep_alive"] = cfg.ollama_keep_alive
            yield ("content", "ls -la")

        with patch("gemma.commands.shell.client_chat", side_effect=_capture):
            result = runner.invoke(app, ["sh", "--no-exec", "--keep-alive", "2h", "--no-cache", "list files"])
        assert result.exit_code == 0
        assert captured.get("keep_alive") == "2h"


# ---------------------------------------------------------------------------
# Safety blocklist
# ---------------------------------------------------------------------------

class TestShSafety:
    def test_dangerous_rm_rf_blocked(self):
        """Interactive: ``rm -rf /`` must be blocked even after user says y."""
        def _dangerous(messages, cfg, stream=True):
            yield ("content", "rm -rf /")

        with patch("gemma.commands.shell.client_chat", side_effect=_dangerous):
            # Simulate TTY by NOT using --no-exec; CliRunner fakes isatty=False
            # so the pipe-mode path runs and prints the command.
            # To test the safety guard we call the underlying helper directly.
            from gemma.commands.shell import _is_dangerous
            assert _is_dangerous("rm -rf /") is True

    def test_safe_commands_not_blocked(self):
        from gemma.commands.shell import _is_dangerous
        assert _is_dangerous("find . -name '*.py'") is False
        assert _is_dangerous("git log --oneline -10") is False
        assert _is_dangerous("ls -la /tmp") is False

    def test_fork_bomb_blocked(self):
        from gemma.commands.shell import _is_dangerous
        assert _is_dangerous(":(){:|:&};:") is True

    def test_mkfs_blocked(self):
        from gemma.commands.shell import _is_dangerous
        assert _is_dangerous("mkfs.ext4 /dev/sdb") is True


# ---------------------------------------------------------------------------
# gemma why
# ---------------------------------------------------------------------------

class TestWhyCommand:
    def test_reads_last_file_and_explains(self, tmp_path):
        last_file = tmp_path / "last_cmd"
        last_file.write_text("1\tgit push origin main\n")

        with patch("gemma.commands.shell.client_chat", side_effect=_fake_chat_why):
            result = runner.invoke(
                app, ["why", "--last-file", str(last_file)]
            )
        assert result.exit_code == 0
        assert "failed" in result.output.lower() or "path" in result.output.lower()

    def test_exit_code_zero_prints_success_message(self, tmp_path):
        last_file = tmp_path / "last_cmd"
        last_file.write_text("0\tls -la\n")

        result = runner.invoke(app, ["why", "--last-file", str(last_file)])
        assert result.exit_code == 0
        assert "succeeded" in result.output.lower() or "exit 0" in result.output.lower()

    def test_missing_file_exits_nonzero(self, tmp_path):
        result = runner.invoke(
            app, ["why", "--last-file", str(tmp_path / "nonexistent")]
        )
        assert result.exit_code != 0

    def test_empty_file_exits_nonzero(self, tmp_path):
        last_file = tmp_path / "last_cmd"
        last_file.write_text("")

        result = runner.invoke(app, ["why", "--last-file", str(last_file)])
        assert result.exit_code != 0

    def test_model_error_exits_nonzero(self, tmp_path):
        last_file = tmp_path / "last_cmd"
        last_file.write_text("127\tnot-a-real-command\n")

        def _boom(messages, cfg, stream=True):
            raise RuntimeError("model unavailable")
            yield

        with patch("gemma.commands.shell.client_chat", side_effect=_boom):
            result = runner.invoke(app, ["why", "--last-file", str(last_file)])
        assert result.exit_code != 0

    def test_with_stderr_field(self, tmp_path):
        """Three-field record (exit\tcmd\tstderr) is parsed correctly."""
        last_file = tmp_path / "last_cmd"
        last_file.write_text("1\tpython app.py\tModuleNotFoundError: No module named 'typer'\n")

        with patch("gemma.commands.shell.client_chat", side_effect=_fake_chat_why):
            result = runner.invoke(app, ["why", "--last-file", str(last_file)])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# gemma install-shell
# ---------------------------------------------------------------------------

class TestInstallShellCommand:
    def test_bash_snippet_printed(self):
        result = runner.invoke(app, ["install-shell", "--shell", "bash"])
        assert result.exit_code == 0
        assert "GEMMA_LAST_FILE" in result.output
        assert "PROMPT_COMMAND" in result.output
        assert "_gemma_post" in result.output

    def test_zsh_snippet_printed(self):
        result = runner.invoke(app, ["install-shell", "--shell", "zsh"])
        assert result.exit_code == 0
        assert "GEMMA_LAST_FILE" in result.output
        assert "add-zsh-hook" in result.output
        assert "_gemma_precmd" in result.output

    def test_unsupported_shell_exits_nonzero(self):
        result = runner.invoke(app, ["install-shell", "--shell", "fish"])
        assert result.exit_code != 0

    def test_append_creates_backup(self, tmp_path):
        rc = tmp_path / ".bashrc"
        rc.write_text("# existing content\n")

        result = runner.invoke(
            app, ["install-shell", "--shell", "bash", "--append", str(rc)]
        )
        assert result.exit_code == 0
        # Backup is the original file name with ".gemma-backup" appended
        backup = rc.parent / (rc.name + ".gemma-backup")
        assert backup.exists(), f"Expected backup at {backup}"
        # Snippet must appear at the end of the rc file
        appended = rc.read_text()
        assert "GEMMA_LAST_FILE" in appended
        assert "# existing content" in appended

    def test_append_to_nonexistent_file(self, tmp_path):
        """Appending to a new file should work without error."""
        rc = tmp_path / ".zshrc"
        result = runner.invoke(
            app, ["install-shell", "--shell", "zsh", "--append", str(rc)]
        )
        assert result.exit_code == 0
        assert rc.exists()
        assert "GEMMA_LAST_FILE" in rc.read_text()

    def test_snippet_is_multi_line(self):
        """The printed snippet must contain multiple actual newline characters.

        (Guards against accidentally collapsing the snippet to a single line.)
        The snippet legitimately contains shell printf escape sequences like
        ``\\n`` and ``\\t``, so we only check for real newlines here.
        """
        result = runner.invoke(app, ["install-shell", "--shell", "bash"])
        assert result.output.count("\n") > 5, "Snippet should span multiple lines"
