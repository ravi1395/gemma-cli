"""Tests for ``gemma explain``.

Uses ``typer.testing.CliRunner`` and mocks ``client.chat`` so no Ollama
server is required.  The three main input modes (stdin, file, --cmd/--error)
and the error paths are all exercised.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from gemma.main import app


runner = CliRunner()


# ---------------------------------------------------------------------------
# Shared stub
# ---------------------------------------------------------------------------

def _fake_chat(messages, cfg, stream=True):
    """Returns a canned explanation string."""
    yield ("content", "This is a short explanation of the input.")


# ---------------------------------------------------------------------------
# --cmd mode
# ---------------------------------------------------------------------------

class TestExplainCmd:
    def test_cmd_mode_happy_path(self):
        with patch("gemma.commands.explain.client_chat", side_effect=_fake_chat):
            result = runner.invoke(
                app,
                ["explain", "--cmd", "find . -name '*.py' -mtime -1", "--no-stream"],
            )
        assert result.exit_code == 0
        assert "explanation" in result.output.lower()

    def test_cmd_mode_forwarded_in_user_message(self):
        """The --cmd value must appear in the message sent to the model."""
        captured: list[str] = []

        def _capture(messages, cfg, stream=True):
            captured.extend(m["content"] for m in messages)
            yield ("content", "ok")

        with patch("gemma.commands.explain.client_chat", side_effect=_capture):
            runner.invoke(
                app,
                ["explain", "--cmd", "ls -la /tmp", "--no-stream"],
            )

        joined = " ".join(captured)
        assert "ls -la /tmp" in joined

    def test_cmd_mode_does_not_execute_command(self, tmp_path):
        """Passing --cmd must never execute the command on the host."""
        sentinel = tmp_path / "was_created"
        cmd = f"touch {sentinel}"

        with patch("gemma.commands.explain.client_chat", side_effect=_fake_chat):
            runner.invoke(app, ["explain", "--cmd", cmd, "--no-stream"])

        assert not sentinel.exists(), "--cmd must not execute the command"


# ---------------------------------------------------------------------------
# --error mode
# ---------------------------------------------------------------------------

class TestExplainError:
    def test_error_mode_happy_path(self):
        with patch("gemma.commands.explain.client_chat", side_effect=_fake_chat):
            result = runner.invoke(
                app,
                ["explain", "--error", "ModuleNotFoundError: No module named 'typer'", "--no-stream"],
            )
        assert result.exit_code == 0

    def test_error_string_in_user_message(self):
        captured: list[str] = []

        def _capture(messages, cfg, stream=True):
            captured.extend(m["content"] for m in messages)
            yield ("content", "ok")

        with patch("gemma.commands.explain.client_chat", side_effect=_capture):
            runner.invoke(
                app,
                ["explain", "--error", "SegmentationFault", "--no-stream"],
            )

        assert "SegmentationFault" in " ".join(captured)


# ---------------------------------------------------------------------------
# File mode
# ---------------------------------------------------------------------------

class TestExplainFile:
    def test_file_mode_reads_file(self, tmp_path):
        f = tmp_path / "errors.log"
        f.write_text("ERROR: connection refused\n" * 5)

        with patch("gemma.commands.explain.client_chat", side_effect=_fake_chat):
            result = runner.invoke(
                app, ["explain", str(f), "--no-stream"]
            )
        assert result.exit_code == 0

    def test_file_mode_lines_flag(self, tmp_path):
        """--lines N should limit how many lines are read."""
        f = tmp_path / "big.log"
        f.write_text("\n".join(f"line {i}" for i in range(100)))

        captured_messages: list[str] = []

        def _capture(messages, cfg, stream=True):
            captured_messages.extend(m["content"] for m in messages)
            yield ("content", "ok")

        with patch("gemma.commands.explain.client_chat", side_effect=_capture):
            runner.invoke(app, ["explain", str(f), "--lines", "3", "--no-stream"])

        joined = " ".join(captured_messages)
        # Lines 0–2 should appear; line 3 should not
        assert "line 0" in joined
        assert "line 2" in joined
        assert "line 3" not in joined

    def test_missing_file_exits_nonzero(self):
        result = runner.invoke(app, ["explain", "/nonexistent/path/file.txt"])
        assert result.exit_code != 0

    def test_empty_file_exits_nonzero(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("")
        result = runner.invoke(app, ["explain", str(f)])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Stdin mode
# ---------------------------------------------------------------------------

class TestExplainStdin:
    def test_stdin_pipe_mode(self):
        with patch("gemma.commands.explain.client_chat", side_effect=_fake_chat):
            result = runner.invoke(
                app,
                ["explain", "--no-stream"],
                input="stack trace goes here",
            )
        assert result.exit_code == 0

    def test_stdin_content_forwarded(self):
        captured: list[str] = []

        def _capture(messages, cfg, stream=True):
            captured.extend(m["content"] for m in messages)
            yield ("content", "ok")

        with patch("gemma.commands.explain.client_chat", side_effect=_capture):
            runner.invoke(
                app,
                ["explain", "--no-stream"],
                input="ERROR: disk full",
            )

        assert "ERROR: disk full" in " ".join(captured)


# ---------------------------------------------------------------------------
# No-input error path
# ---------------------------------------------------------------------------

class TestExplainNoInput:
    def test_no_input_no_tty_exits_nonzero(self):
        """Invoking explain with no args, no stdin, and no flags must exit non-zero."""
        result = runner.invoke(app, ["explain"])
        # CliRunner sets stdin to empty bytes; explain should detect no usable input
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Priority: file > --cmd
# ---------------------------------------------------------------------------

class TestExplainPriority:
    def test_file_takes_priority_over_cmd(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")

        captured: list[str] = []

        def _capture(messages, cfg, stream=True):
            captured.extend(m["content"] for m in messages)
            yield ("content", "ok")

        with patch("gemma.commands.explain.client_chat", side_effect=_capture):
            runner.invoke(
                app,
                ["explain", str(f), "--cmd", "ls -la", "--no-stream"],
            )

        joined = " ".join(captured)
        # File content should be in the message, not the --cmd value
        assert "x = 1" in joined
        assert "ls -la" not in joined


# ---------------------------------------------------------------------------
# Model / config flags
# ---------------------------------------------------------------------------

class TestExplainFlags:
    def test_keep_alive_forwarded(self):
        captured: dict = {}

        def _capture(messages, cfg, stream=True):
            captured["keep_alive"] = cfg.ollama_keep_alive
            yield ("content", "ok")

        with patch("gemma.commands.explain.client_chat", side_effect=_capture):
            runner.invoke(
                app,
                ["explain", "--cmd", "ls", "--keep-alive", "1h", "--no-stream"],
            )

        assert captured.get("keep_alive") == "1h"

    def test_model_flag_forwarded(self):
        captured: dict = {}

        def _capture(messages, cfg, stream=True):
            captured["model"] = cfg.model
            yield ("content", "ok")

        with patch("gemma.commands.explain.client_chat", side_effect=_capture):
            runner.invoke(
                app,
                ["explain", "--cmd", "ls", "--model", "gemma3:4b", "--no-stream"],
            )

        assert captured.get("model") == "gemma3:4b"

    def test_model_error_exits_nonzero(self):
        def _boom(messages, cfg, stream=True):
            raise RuntimeError("model unavailable")
            yield

        with patch("gemma.commands.explain.client_chat", side_effect=_boom):
            result = runner.invoke(
                app,
                ["explain", "--cmd", "ls", "--no-stream"],
            )
        assert result.exit_code != 0
