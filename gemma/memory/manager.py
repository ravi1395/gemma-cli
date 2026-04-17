"""The orchestrator that ties the memory system together.

MemoryManager is the single entry point used by the CLI:
  - record_turn()          — log a user or assistant turn
  - get_context_messages() — build the prompt for the next Gemma call
  - get_stats()            — introspection for `gemma history stats`

Internally it coordinates MemoryStore, CondensationPipeline, MemoryRetriever,
ContextAssembler, and an Embedder. It also supports a "degraded" mode where
Redis is unreachable — in that case we fall back to an in-memory turn list
and skip condensation entirely so the CLI still works.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Optional

from gemma.config import Config
from gemma.embeddings import Embedder
from gemma.memory.condensation import CondensationPipeline
from gemma.memory.context import ContextAssembler
from gemma.memory.models import ConversationTurn, MemoryRecord
from gemma.memory.retrieval import MemoryRetriever
from gemma.memory.store import MemoryStore
from gemma.redaction import redact


logger = logging.getLogger(__name__)


class MemoryManager:
    """Coordinate the whole memory pipeline."""

    def __init__(
        self,
        config: Config,
        *,
        store: Optional[MemoryStore] = None,
        embedder: Optional[Embedder] = None,
        pipeline: Optional[CondensationPipeline] = None,
        retriever: Optional[MemoryRetriever] = None,
        assembler: Optional[ContextAssembler] = None,
        session_id: Optional[str] = None,
    ):
        self._config = config
        self._store = store or MemoryStore(config)
        self._embedder = embedder or Embedder(
            model=config.embedding_model,
            host=config.ollama_host,
            keep_alive=config.ollama_keep_alive,
        )
        self._pipeline = pipeline or CondensationPipeline(config)
        self._assembler = assembler or ContextAssembler(config)
        self._retriever = retriever or MemoryRetriever(
            self._store, self._embedder, config
        )
        self._session_id: str = session_id or uuid.uuid4().hex
        self._degraded: bool = False
        self._turn_counter: int = 0
        self._fallback_turns: list[ConversationTurn] = []
        self._condensation_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """Attempt to connect to Redis. Returns True if memory is fully online."""
        if not self._config.memory_enabled:
            self._degraded = True
            return False
        ok = self._store.connect()
        self._degraded = not ok
        if self._degraded:
            logger.warning(
                "Memory system unavailable (Redis not reachable). Running without memory."
            )
        return ok

    @property
    def available(self) -> bool:
        return self._config.memory_enabled and not self._degraded and self._store.available

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def degraded(self) -> bool:
        return self._degraded

    # ------------------------------------------------------------------
    # Turn recording + async condensation
    # ------------------------------------------------------------------

    def record_turn(self, role: str, content: str) -> None:
        """Log a turn. Triggers background condensation on overflow.

        The content is scrubbed for known secret patterns (AWS keys, GitHub
        tokens, JWTs, PRIVATE KEY blocks, .env-shaped lines, Bearer tokens)
        BEFORE the turn is constructed, so nothing sensitive is written to
        the in-memory fallback list, the Redis sliding window, or any later
        condensed memory derived from it.
        """
        self._turn_counter += 1
        clean_content, findings = redact(content)
        if findings:
            logger.info(
                "Redacted %d secret(s) from %s turn: types=%s",
                len(findings),
                role,
                sorted({f.type for f in findings}),
            )
        turn = ConversationTurn(
            role=role, content=clean_content, turn_number=self._turn_counter
        )

        if not self.available:
            # In degraded mode, keep a bounded in-memory list
            self._fallback_turns.append(turn)
            if len(self._fallback_turns) > self._config.sliding_window_size * 2:
                # Drop the oldest entries to avoid unbounded growth
                self._fallback_turns = self._fallback_turns[
                    -self._config.sliding_window_size :
                ]
            return

        self._store.push_turn(self._session_id, turn)

        # If we've exceeded the window, pull overflow and hand it off for
        # condensation. Do this synchronously OR in a background thread
        # depending on config so callers aren't blocked on slow inference.
        if self._store.get_turn_count(self._session_id) > self._config.sliding_window_size:
            overflow = self._store.get_overflow_turns(
                self._session_id, self._config.sliding_window_size
            )
            if overflow:
                self._schedule_condensation(overflow)

    def _schedule_condensation(self, overflow: list[ConversationTurn]) -> None:
        if self._config.condensation_async:
            t = threading.Thread(
                target=self._condense_and_store,
                args=(overflow,),
                name="gemma-condensation",
                daemon=True,
            )
            t.start()
        else:
            self._condense_and_store(overflow)

    def _condense_and_store(self, overflow: list[ConversationTurn]) -> None:
        """Run condensation and persist new memories. Safe to call in a thread."""
        with self._condensation_lock:
            try:
                existing = self._retriever.find_relevant(
                    query=self._summarize_for_context(overflow),
                    top_k=10,
                    min_similarity=self._config.memory_min_similarity,
                )
                existing_records = [r for r, _ in existing]

                new_records = self._pipeline.condense_turns(
                    overflow, existing_records, session_id=self._session_id
                )
                if not new_records:
                    return

                for rec in new_records:
                    # Check for near-duplicates to supersede
                    self._maybe_supersede(rec)
                    # Embed and persist
                    try:
                        vec = self._embedder.embed(rec.content)
                    except Exception:
                        vec = None
                    self._store.save_memory(rec)
                    if vec is not None and vec.size > 0:
                        self._store.save_embedding(rec.memory_id, vec)

                # If we've accumulated a lot, trigger reconsolidation
                if self._store.count_active_memories() > self._config.memory_max_count:
                    self._maybe_reconsolidate()

            except Exception:
                logger.exception("Condensation failed (non-fatal, skipping batch)")

    def _maybe_supersede(self, new_rec: MemoryRecord) -> None:
        """If the new memory near-duplicates an existing one, supersede it."""
        try:
            conflicts = self._retriever.find_conflicting(new_rec.content)
        except Exception:
            return
        for old_rec, _score in conflicts:
            if old_rec.memory_id == new_rec.memory_id:
                continue
            # Only supersede if the new one is at least as important
            if new_rec.importance >= old_rec.importance:
                self._store.supersede_memory(old_rec.memory_id, new_rec.memory_id)

    def _maybe_reconsolidate(self) -> None:
        """Merge the full memory set into a shorter list (the 'recursive' step)."""
        try:
            all_active = self._store.get_all_active_memories()
            if len(all_active) <= self._config.memory_max_count:
                return
            merged = self._pipeline.reconsolidate(all_active)
            if not merged:
                return
            # Retire the old ones, persist the new ones.
            for old in all_active:
                self._store.supersede_memory(old.memory_id, "__reconsolidated__")
            for new_rec in merged:
                try:
                    vec = self._embedder.embed(new_rec.content)
                except Exception:
                    vec = None
                self._store.save_memory(new_rec)
                if vec is not None and vec.size > 0:
                    self._store.save_embedding(new_rec.memory_id, vec)
        except Exception:
            logger.exception("Reconsolidation failed")

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def get_context_messages(
        self,
        current_query: str,
        system_prompt: Optional[str] = None,
    ) -> list[dict[str, str]]:
        """Assemble the final message list for the next Gemma call."""
        sys_prompt = system_prompt or self._config.system_prompt

        if not self.available:
            return self._assembler.build_messages(
                system_prompt=sys_prompt,
                relevant_memories=[],
                recent_turns=self._fallback_turns,
            )

        try:
            hits = self._retriever.find_relevant(current_query)
        except Exception:
            hits = []
        memories = [rec for rec, _ in hits]

        recent = self._store.get_recent_turns(
            self._session_id, self._config.sliding_window_size
        )
        messages = self._assembler.build_messages(
            system_prompt=sys_prompt,
            relevant_memories=memories,
            recent_turns=recent,
        )
        # Leave 25% headroom for the response
        budget = max(512, int(self._config.context_window * 0.75))
        return self._assembler.trim_to_budget(messages, budget)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        if not self.available:
            return {
                "available": False,
                "session_id": self._session_id,
                "fallback_turns": len(self._fallback_turns),
            }
        count = self._store.count_active_memories()
        top = self._store.get_top_memories(n=count or 1)
        by_category: dict[str, int] = {}
        by_importance: dict[int, int] = {}
        for rec in top:
            by_category[rec.category.value] = by_category.get(rec.category.value, 0) + 1
            by_importance[rec.importance] = by_importance.get(rec.importance, 0) + 1
        return {
            "available": True,
            "session_id": self._session_id,
            "active_memories": count,
            "by_category": by_category,
            "by_importance": by_importance,
            "window_turns": self._store.get_turn_count(self._session_id),
        }

    def list_memories(self, limit: int = 50) -> list[MemoryRecord]:
        if not self.available:
            return []
        return self._store.get_top_memories(n=limit)

    def clear_session(self) -> None:
        """Wipe the sliding window for this session. Keeps condensed memories."""
        self._fallback_turns = []
        self._turn_counter = 0
        if self.available:
            self._store.clear_session(self._session_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _summarize_for_context(turns: list[ConversationTurn]) -> str:
        """Cheap concatenation used only for retrieval-during-condensation."""
        return " ".join(t.content for t in turns if t.role != "system")[:500]
