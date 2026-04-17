"""Tests for ``gemma commit`` and ``gemma diff``.

Each test that exercises real git operations uses a throwaway repository
created inside ``tmp_path`` via ``subprocess``.  The model call is always
mocked with a ``client_chat`` stub so no Ollama server is required.

``monkeypatch.chdir`` is used to set the working directory for the duration
of each test, which ensures the ``git`` subprocess calls inside the commands
see the right repository without mutating the global cwd permanently.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from gemma.main import app


runner = CliRunner()


# ---------------------------------------------------------------------------
# Shared git-repo fixtures
# ---------------------------------------------------------------------------

def _git(*args: str, cwd: Path) -> None:
    """Run a git command in *cwd*, raising on failure."""
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture()
def git_repo(tmp_path: Path) -> Path:
    """A git repo with one initial commit and one **staged** new file.

    Layout after setup:
        README.md  — committed
        hello.py   — staged (not yet committed)
    """
    _git("init", cwd=tmp_path)
    _git("config", "user.email", "test@example.com", cwd=tmp_path)
    _git("config", "user.name", "Test User", cwd=tmp_path)

    readme = tmp_path / "README.md"
    readme.write_text("# Test repo\n")
    _git("add", "README.md", cwd=tmp_path)
    _git("commit", "-m", "init", cwd=tmp_path)

    # Stage a new file so `git diff --cached` has content.
    new_file = tmp_path / "hello.py"
    new_file.write_text("def greet(name: str) -> str:\n    return f'Hello, {name}!'\n")
    _git("add", "hello.py", cwd=tmp_path)

    return tmp_path


@pytest.fixture()
def git_repo_unstaged(tmp_path: Path) -> Path:
    """A git repo with one initial commit and one **unstaged** modification.

    Layout after setup:
        README.md — committed, then modified (not staged)
    """
    _git("init", cwd=tmp_path)
    _git("config", "user.email", "test@example.com", cwd=tmp_path)
    _git("config", "user.name", "Test User", cwd=tmp_path)

    readme = tmp_path / "README.md"
    readme.write_text("# Test repo\n")
    _git("add", "README.md", cwd=tmp_path)
    _git("commit", "-m", "init", cwd=tmp_path)

    # Modify without staging.
    readme.write_text("# Test repo\n\nUpdated content here.\n")

    return tmp_path


# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------

def _fake_commit_msg(messages, cfg, stream=True):
    """Returns a canned conventional-commit message."""
    yield (
        "content",
        "feat(greet): add greeting function\n\n"
        "Adds greet() which returns a personalised greeting string.",
    )


def _fake_diff_summary(messages, cfg, stream=True):
    """Returns a canned per-file diff summary."""
    yield ("content", "hello.py — adds a greet() helper function.")


# ---------------------------------------------------------------------------
# gemma commit — happy paths
# ---------------------------------------------------------------------------

class TestCommitCommand:
    def test_prints_generated_message(self, git_repo, monkeypatch):
        """`gemma commit` prints the model's message for review."""
        monkeypatch.chdir(git_repo)
        with patch("gemma.commands.git.client_chat", side_effect=_fake_commit_msg):
            result = runner.invoke(app, ["commit"])
        assert result.exit_code == 0
        assert "feat" in result.output

    def test_apply_creates_git_commit(self, git_repo, monkeypatch):
        """`--apply` must actually run git commit and create a new entry in the log."""
        monkeypatch.chdir(git_repo)
        with patch("gemma.commands.git.client_chat", side_effect=_fake_commit_msg):
            result = runner.invoke(app, ["commit", "--apply"])
        assert result.exit_code == 0

        log = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            cwd=git_repo, capture_output=True, text=True,
        )
        # The subject line from the stub starts with "feat(greet):"
        assert "feat" in log.stdout

    def test_no_apply_shows_tip(self, git_repo, monkeypatch):
        """Without --apply the output must mention --apply so the user knows how to proceed."""
        monkeypatch.chdir(git_repo)
        with patch("gemma.commands.git.client_chat", side_effect=_fake_commit_msg):
            result = runner.invoke(app, ["commit"])
        assert result.exit_code == 0
        assert "--apply" in result.output

    def test_type_flag_forwarded_to_model(self, git_repo, monkeypatch):
        """--type must appear in the user message sent to the model."""
        captured: list[str] = []

        def _capture(messages, cfg, stream=True):
            captured.extend(m["content"] for m in messages)
            yield ("content", "chore: update greet helper")

        monkeypatch.chdir(git_repo)
        with patch("gemma.commands.git.client_chat", side_effect=_capture):
            result = runner.invoke(app, ["commit", "--type", "chore"])
        assert result.exit_code == 0
        assert any("chore" in msg for msg in captured)

    def test_keep_alive_forwarded_to_config(self, git_repo, monkeypatch):
        """--keep-alive must be stored in cfg.ollama_keep_alive."""
        captured: dict = {}

        def _capture(messages, cfg, stream=True):
            captured["keep_alive"] = cfg.ollama_keep_alive
            yield ("content", "fix: something")

        monkeypatch.chdir(git_repo)
        with patch("gemma.commands.git.client_chat", side_effect=_capture):
            runner.invoke(app, ["commit", "--keep-alive", "1h"])
        assert captured.get("keep_alive") == "1h"


# ---------------------------------------------------------------------------
# gemma commit — error paths
# ---------------------------------------------------------------------------

class TestCommitErrors:
    def test_not_a_git_repo_exits_nonzero(self, tmp_path, monkeypatch):
        """Outside a repo the command must exit non-zero without touching the model."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["commit"])
        assert result.exit_code != 0

    def test_nothing_staged_exits_nonzero(self, tmp_path, monkeypatch):
        """An empty staging area must produce a clear error and exit non-zero."""
        # Repo with an initial commit but nothing staged.
        _git("init", cwd=tmp_path)
        _git("config", "user.email", "test@example.com", cwd=tmp_path)
        _git("config", "user.name", "Test User", cwd=tmp_path)
        readme = tmp_path / "README.md"
        readme.write_text("# Test\n")
        _git("add", "README.md", cwd=tmp_path)
        _git("commit", "-m", "init", cwd=tmp_path)

        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["commit"])
        assert result.exit_code != 0

    def test_model_error_exits_nonzero(self, git_repo, monkeypatch):
        def _boom(messages, cfg, stream=True):
            raise RuntimeError("model unavailable")
            yield  # make it a generator

        monkeypatch.chdir(git_repo)
        with patch("gemma.commands.git.client_chat", side_effect=_boom):
            result = runner.invoke(app, ["commit"])
        assert result.exit_code != 0

    def test_empty_model_response_exits_nonzero(self, git_repo, monkeypatch):
        def _empty(messages, cfg, stream=True):
            yield ("content", "   ")

        monkeypatch.chdir(git_repo)
        with patch("gemma.commands.git.client_chat", side_effect=_empty):
            result = runner.invoke(app, ["commit"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# gemma diff — happy paths
# ---------------------------------------------------------------------------

class TestDiffCommand:
    def test_unstaged_diff_summarized(self, git_repo_unstaged, monkeypatch):
        """`gemma diff` summarizes unstaged working-tree changes."""
        monkeypatch.chdir(git_repo_unstaged)
        with patch("gemma.commands.git.client_chat", side_effect=_fake_diff_summary):
            result = runner.invoke(app, ["diff"])
        assert result.exit_code == 0
        assert result.output.strip()  # non-empty output

    def test_staged_flag_summarizes_staging_area(self, git_repo, monkeypatch):
        """`--staged` should diff the staging area and produce a summary."""
        monkeypatch.chdir(git_repo)
        with patch("gemma.commands.git.client_chat", side_effect=_fake_diff_summary):
            result = runner.invoke(app, ["diff", "--staged"])
        assert result.exit_code == 0

    def test_overall_flag_uses_prose_prompt(self, git_repo_unstaged, monkeypatch):
        """``--overall`` must send a system prompt that asks for prose, not per-file lines."""
        captured: list[str] = []

        def _capture(messages, cfg, stream=True):
            captured.extend(m["content"] for m in messages)
            yield ("content", "This diff updates the README with additional content.")

        monkeypatch.chdir(git_repo_unstaged)
        with patch("gemma.commands.git.client_chat", side_effect=_capture):
            result = runner.invoke(app, ["diff", "--overall"])
        assert result.exit_code == 0
        # The overall system prompt contains "prose" to signal a paragraph mode.
        sys_msgs = [m for m in captured if "prose" in m.lower()]
        assert sys_msgs, "Overall mode must use the prose system prompt"

    def test_refspec_argument_accepted(self, git_repo, monkeypatch):
        """A positional refspec (e.g. HEAD~1) is passed through to git diff."""
        # Commit the staged file so HEAD~1 is valid.
        _git("commit", "-m", "add hello.py", cwd=git_repo)

        monkeypatch.chdir(git_repo)
        with patch("gemma.commands.git.client_chat", side_effect=_fake_diff_summary):
            result = runner.invoke(app, ["diff", "HEAD~1"])
        assert result.exit_code == 0

    def test_no_changes_prints_info_and_exits_zero(self, tmp_path, monkeypatch):
        """An empty diff should produce an informational message and exit 0."""
        _git("init", cwd=tmp_path)
        _git("config", "user.email", "test@example.com", cwd=tmp_path)
        _git("config", "user.name", "Test User", cwd=tmp_path)
        readme = tmp_path / "README.md"
        readme.write_text("# Test\n")
        _git("add", "README.md", cwd=tmp_path)
        _git("commit", "-m", "init", cwd=tmp_path)

        monkeypatch.chdir(tmp_path)
        # No unstaged or staged changes at this point.
        result = runner.invoke(app, ["diff"])
        assert result.exit_code == 0
        assert "no changes" in result.output.lower()


# ---------------------------------------------------------------------------
# gemma diff — error paths
# ---------------------------------------------------------------------------

class TestDiffErrors:
    def test_not_a_git_repo_exits_nonzero(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["diff"])
        assert result.exit_code != 0

    def test_model_error_exits_nonzero(self, git_repo_unstaged, monkeypatch):
        def _boom(messages, cfg, stream=True):
            raise RuntimeError("model unavailable")
            yield

        monkeypatch.chdir(git_repo_unstaged)
        with patch("gemma.commands.git.client_chat", side_effect=_boom):
            result = runner.invoke(app, ["diff"])
        assert result.exit_code != 0

    def test_empty_model_response_exits_nonzero(self, git_repo_unstaged, monkeypatch):
        def _empty(messages, cfg, stream=True):
            yield ("content", "   ")

        monkeypatch.chdir(git_repo_unstaged)
        with patch("gemma.commands.git.client_chat", side_effect=_empty):
            result = runner.invoke(app, ["diff"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Helpers — unit tests for internal functions
# ---------------------------------------------------------------------------

class TestTruncateDiff:
    def test_short_diff_not_truncated(self):
        from gemma.commands.git import _truncate_diff
        short = "a" * 100
        result, was_truncated = _truncate_diff(short, max_bytes=200)
        assert result == short
        assert was_truncated is False

    def test_long_diff_truncated(self):
        from gemma.commands.git import _truncate_diff
        long_diff = "x" * 30_000
        result, was_truncated = _truncate_diff(long_diff, max_bytes=20_000)
        assert was_truncated is True
        assert len(result.encode("utf-8")) <= 20_000

    def test_exact_boundary_not_truncated(self):
        from gemma.commands.git import _truncate_diff
        exact = "y" * 20_000
        result, was_truncated = _truncate_diff(exact, max_bytes=20_000)
        assert was_truncated is False
        assert result == exact
