"""Chat session persistence.

Phase 1: JSON-file-based single-session history at ~/.gemma_history.json.
Phase 6 will replace this with an adapter pattern (Redis primary, JSON fallback),
but the public interface (load/append/save/clear/show) remains stable.
"""

import json
from pathlib import Path
from typing import Optional

from gemma.config import Config


class SessionHistory:
    """Load/save/clear the chat history as a JSON file."""

    def __init__(self, config: Config):
        self._config = config
        self._path: Path = config.resolved_history_path()
        self._turns: list[dict] = []
        self._loaded: bool = False

    # --- Public API ---

    def load(self) -> list[dict]:
        """Load turns from disk. Returns an empty list if the file is absent."""
        if self._path.exists():
            try:
                with self._path.open("r", encoding="utf-8") as f:
                    self._turns = json.load(f)
            except (json.JSONDecodeError, OSError):
                # Corrupt or unreadable file -- start fresh rather than crashing
                self._turns = []
        else:
            self._turns = []
        self._loaded = True
        return self._turns

    def append(self, role: str, content: str) -> None:
        """Add a turn to the in-memory list. Call save() to persist."""
        if not self._loaded:
            self.load()
        self._turns.append({"role": role, "content": content})

    def save(self) -> None:
        """Write the current turns to disk, creating parent dirs if needed."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as f:
            json.dump(self._turns, f, indent=2, ensure_ascii=False)

    def clear(self) -> None:
        """Reset the in-memory turns and remove the JSON file if present."""
        self._turns = []
        if self._path.exists():
            self._path.unlink()

    def show(self) -> list[dict]:
        """Return the current turns (loading from disk if needed)."""
        if not self._loaded:
            self.load()
        return list(self._turns)

    @property
    def turns(self) -> list[dict]:
        """Direct access to the current turn list (loads lazily)."""
        if not self._loaded:
            self.load()
        return self._turns
