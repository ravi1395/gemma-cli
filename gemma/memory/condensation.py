"""Self-summarization pipeline.

Uses Gemma itself to compress overflow conversation turns into structured
MemoryRecords. This is where Gemma "processes its own memories" before they
go to Redis.

Design notes:
  - A 4B model will make mistakes; the prompt is deliberately simple with a
    single few-shot example and a fixed category enum.
  - Output is expected to be a JSON array; fallback parsing handles the common
    failure modes (markdown fences, leading/trailing prose).
  - Reconsolidation merges an existing list of memories into a shorter list --
    the "recursive" part of recursive summarization.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

import ollama

from gemma.config import Config
from gemma.memory.models import (
    ConversationTurn,
    MemoryCategory,
    MemoryRecord,
)


# Valid category values, rendered into the prompt and used to constrain parsing
_CATEGORY_VALUES = [c.value for c in MemoryCategory]
_CATEGORY_LIST_STR = ", ".join(_CATEGORY_VALUES)


_EXTRACTION_TEMPLATE = """You are a memory extraction system. Read the conversation below and extract key facts as structured JSON.

Rules:
- Extract only facts worth remembering for future conversations
- Each fact must be a single, self-contained statement
- Assign a category from: {categories}
- Assign importance 1-5 (5 = critical identity/preference, 1 = trivial)
- If a fact contradicts a known memory, mark it as category "correction" with importance 5
- Output a valid JSON array only, no other text, no markdown fences

Known memories (may be contradicted by new conversation):
{known_block}

Conversation to extract from:
{conversation}

Example output:
[
  {{"content": "User prefers Python over JavaScript", "category": "user_preference", "importance": 4}},
  {{"content": "User is building a CLI tool called gemma-cli", "category": "task_state", "importance": 5}},
  {{"content": "The project uses Typer for CLI framework", "category": "factual_context", "importance": 3}}
]

Extract memories now:
"""


_RECONSOLIDATION_TEMPLATE = """You are a memory consolidation system. You will receive a list of previously extracted memories. Merge duplicates, remove outdated information, and produce a shorter, more concise list.

Rules:
- Combine related facts into single statements when possible
- Drop facts that are no longer relevant (superseded or trivial)
- Preserve all corrections and high-importance facts (importance 4-5) exactly
- Output a valid JSON array only, no other text, no markdown fences

Memories to consolidate:
{memories}

Consolidated output:
"""


class CondensationPipeline:
    """Turn conversation turns into structured MemoryRecords via Gemma."""

    def __init__(self, config: Config, client: Optional[Any] = None):
        self._config = config
        self._client = client or ollama.Client(host=config.ollama_host)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def condense_turns(
        self,
        turns: list[ConversationTurn],
        existing_memories: Optional[list[MemoryRecord]] = None,
        session_id: str = "",
    ) -> list[MemoryRecord]:
        """Extract memories from a batch of conversation turns."""
        if not turns:
            return []

        prompt = self._build_extraction_prompt(turns, existing_memories or [])
        raw = self._call_model(prompt)
        parsed = self._parse_extraction_response(raw)
        return self._to_memory_records(parsed, turns, session_id)

    def reconsolidate(
        self,
        memories: list[MemoryRecord],
    ) -> list[MemoryRecord]:
        """Merge a memory list into a shorter, deduplicated list."""
        if len(memories) <= 1:
            return memories

        payload = [
            {
                "content": m.content,
                "category": m.category.value,
                "importance": m.importance,
            }
            for m in memories
        ]
        prompt = _RECONSOLIDATION_TEMPLATE.format(
            memories=json.dumps(payload, indent=2)
        )
        raw = self._call_model(prompt)
        parsed = self._parse_extraction_response(raw)
        if not parsed:
            # If Gemma returned nothing usable, keep the originals
            return memories

        session_id = memories[0].session_id if memories else ""
        turn_range = memories[0].turn_range if memories else ""
        return self._to_memory_records(parsed, [], session_id, turn_range=turn_range)

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def _build_extraction_prompt(
        self,
        turns: list[ConversationTurn],
        existing: list[MemoryRecord],
    ) -> str:
        conversation = "\n".join(
            f"{t.role}: {t.content}" for t in turns
        )
        if existing:
            known_block = "\n".join(
                f"- [{m.category.value}] {m.content}" for m in existing[:20]
            )
        else:
            known_block = "(none)"
        return _EXTRACTION_TEMPLATE.format(
            categories=_CATEGORY_LIST_STR,
            known_block=known_block,
            conversation=conversation,
        )

    # ------------------------------------------------------------------
    # Model invocation + parsing
    # ------------------------------------------------------------------

    def _call_model(self, prompt: str) -> str:
        """Run a blocking completion against Gemma."""
        response = self._client.chat(
            model=self._config.model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options={"temperature": 0.2},  # lower temp -> more structured output
        )
        try:
            return response["message"]["content"]
        except (KeyError, TypeError):
            return ""

    def _parse_extraction_response(self, raw: str) -> list[dict]:
        """Robustly parse Gemma's JSON output.

        Handles:
          - Clean JSON array
          - JSON inside a ```json ... ``` fence
          - JSON with leading/trailing prose
          - Complete failure (returns [])
        """
        if not raw:
            return []

        # Strip markdown fences
        fenced = re.search(
            r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL | re.IGNORECASE
        )
        if fenced:
            candidate = fenced.group(1)
        else:
            # Find the first top-level JSON array by brace matching
            candidate = self._first_json_array(raw)
            if candidate is None:
                return []

        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            return []

        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, dict)]

    @staticmethod
    def _first_json_array(text: str) -> Optional[str]:
        """Return the substring from the first [ to its matching ]."""
        start = text.find("[")
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    # ------------------------------------------------------------------
    # Record building
    # ------------------------------------------------------------------

    @staticmethod
    def _to_memory_records(
        parsed: list[dict],
        turns: list[ConversationTurn],
        session_id: str,
        turn_range: str = "",
    ) -> list[MemoryRecord]:
        if turns:
            turn_nums = [t.turn_number for t in turns]
            turn_range = turn_range or f"{min(turn_nums)}-{max(turn_nums)}"
            summary_source = turns[0].content[:80]
        else:
            summary_source = ""

        records: list[MemoryRecord] = []
        for item in parsed:
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            category = MemoryCategory.parse(str(item.get("category", "")))
            importance = int(item.get("importance", 2) or 2)
            records.append(
                MemoryRecord(
                    content=content,
                    category=category,
                    importance=importance,
                    session_id=session_id,
                    turn_range=turn_range,
                    source_summary=summary_source,
                )
            )
        return records
