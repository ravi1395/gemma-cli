"""Final prompt assembly from retrieved memories + raw recent turns.

Produces a list of Ollama chat messages in this shape:
  [system_prompt + memory_block]
  recent_turn_1
  ...
  recent_turn_n

The memory block is injected into the system message (not as a separate
message) to keep the role alternation clean and so Gemma treats the
memories as background context rather than prior conversation.
"""

from __future__ import annotations

from typing import Iterable

from gemma.config import Config
from gemma.memory.models import ConversationTurn, MemoryRecord


class ContextAssembler:
    """Build the final chat message list for a single user turn."""

    def __init__(self, config: Config):
        self._config = config

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def build_messages(
        self,
        system_prompt: str,
        relevant_memories: list[MemoryRecord],
        recent_turns: list[ConversationTurn],
    ) -> list[dict[str, str]]:
        """Assemble messages. Memory block is folded into the system message."""
        system_content = system_prompt
        if relevant_memories:
            system_content = system_content + "\n\n" + self._format_memory_block(
                relevant_memories
            )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content}
        ]
        for t in recent_turns:
            # Skip stored system turns; we already supplied a fresh one above.
            if t.role == "system":
                continue
            messages.append(t.to_message())
        return messages

    # ------------------------------------------------------------------
    # Budget trimming
    # ------------------------------------------------------------------

    def trim_to_budget(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> list[dict[str, str]]:
        """Trim messages to fit within an approximate token budget.

        Strategy (least to most disruptive):
          1. Drop lowest-importance memories from the system block
             (best-effort: we only know the sort order, not the raw records
              at this stage, so we peel from the end of the memory block).
          2. Drop oldest non-system turns, never the last 2.
        The final system message + at least two turns are always preserved.
        """
        if self.estimate_token_count(messages) <= max_tokens:
            return messages

        # Step 1: iteratively shrink the memory block in the system message
        sys_msg = messages[0]
        sys_content = sys_msg.get("content", "")
        header, _, block = sys_content.partition("\n\nRelevant memories")
        if block:
            # Rebuild block line-by-line, dropping from the end until we fit
            lines = ("Relevant memories" + block).splitlines()
            while lines and self.estimate_token_count(
                [{"role": "system", "content": header + "\n\n" + "\n".join(lines)}]
                + messages[1:]
            ) > max_tokens:
                lines.pop()
            if lines:
                sys_msg["content"] = header + "\n\n" + "\n".join(lines)
            else:
                sys_msg["content"] = header

        if self.estimate_token_count(messages) <= max_tokens:
            return messages

        # Step 2: drop oldest turns, preserving the tail of the conversation
        # Never drop: system message (idx 0), last 2 turns.
        while len(messages) > 3 and self.estimate_token_count(messages) > max_tokens:
            # Remove messages[1] (oldest non-system turn)
            del messages[1]
        return messages

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_memory_block(self, memories: list[MemoryRecord]) -> str:
        """Render memories as a bulleted block sorted by importance desc."""
        ordered = sorted(
            memories,
            key=lambda m: (-int(m.importance), -float(m.last_accessed)),
        )
        lines = [
            "Relevant memories from previous conversations "
            "(use these naturally, do not repeat them verbatim unless asked):"
        ]
        for m in ordered:
            lines.append(
                f"- [importance {m.importance}] {m.content} "
                f"({m.category.value})"
            )
        return "\n".join(lines)

    @staticmethod
    def estimate_token_count(messages: Iterable[dict[str, str]]) -> int:
        """Rough token estimate: 1 token per ~3.5 English characters.

        Cheap and deterministic; avoids pulling a full tokenizer dependency
        just for budget checks. Good enough for trim decisions.
        """
        msg_list = list(messages)
        total_chars = 0
        for m in msg_list:
            total_chars += len(m.get("content", "")) + len(m.get("role", ""))
        # Add a small per-message overhead for role-switch tokens
        return int(total_chars / 3.5) + len(msg_list)
