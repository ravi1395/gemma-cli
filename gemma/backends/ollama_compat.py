"""Ollama-shape compatibility adapter for the agent loop.

The tool-use agent loop in :func:`gemma.main._agent_loop` predates the
backend abstraction. Its protocol is Ollama-specific: it calls
``client.chat(model=..., messages=..., tools=..., think=...,
keep_alive=..., options={"temperature": ...})`` non-streaming and reads
back ``response["message"]["content"]`` plus ``message["tool_calls"]``.

Rewriting the loop to consume our backend's ``(kind, text)`` tuple
stream is a separate, larger change. In the meantime, this adapter
lets the loop run unchanged against any backend by:

  * Routing the call through the chosen backend's ``chat()``.
  * Synthesising the Ollama dict shape the loop expects.

Tool-calling note
-----------------
The first cut emits ``tool_calls = []`` for the LM Studio backend.
LM Studio surfaces tool use via a different SDK path (``model.act()``)
so genuine tool dispatch on LM Studio is a follow-up. The Ollama
backend continues to use its native client object and supports tool
use exactly as before — selecting ``backend = "ollama"`` is the
escape hatch for tool-heavy workflows until the LM Studio integration
lands.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from gemma.backends import get_backend

if TYPE_CHECKING:
    from gemma.config import Config


class OllamaShapeAdapter:
    """Wrap any LLMBackend in the duck type the agent loop expects.

    Construction is cheap — defer to ``get_backend(config)`` and store
    the config so per-call overrides (model / temperature / keep_alive)
    can be applied without re-resolving the backend.
    """

    def __init__(self, config: "Config") -> None:
        self._config = config
        self._backend = get_backend(config)

    def chat(
        self,
        *,
        model: str,
        messages: list[dict],
        tools: list | None = None,
        think: bool = False,
        keep_alive: str | None = None,
        options: dict | None = None,
        stream: bool = False,
    ) -> dict:
        """Mirror ``ollama.Client.chat`` non-streaming response shape.

        Per-call kwargs override the matching fields on the cached
        ``Config`` so callers don't have to mutate the source object.
        Streaming is not supported here — the agent loop only ever
        calls non-streaming.
        """
        if stream:
            raise NotImplementedError(
                "OllamaShapeAdapter only supports stream=False; the agent "
                "loop never asks for streaming."
            )

        cfg = self._config
        if model and model != cfg.model:
            cfg = replace(cfg, model=model)
        if keep_alive:
            cfg = replace(cfg, ollama_keep_alive=keep_alive)
        if options and "temperature" in options:
            cfg = replace(cfg, temperature=float(options["temperature"]))
        cfg = replace(cfg, thinking_mode=bool(think))

        backend = get_backend(cfg) if cfg is not self._config else self._backend

        content_parts: list[str] = []
        thinking_parts: list[str] = []
        metrics = {"prompt_eval_count": 0, "eval_count": 0}
        for kind, text in backend.chat(messages, cfg, stream=False):
            if kind == "content":
                content_parts.append(text)
            elif kind == "think":
                thinking_parts.append(text)
            elif kind == "metrics":
                try:
                    metrics = json.loads(text)
                except (TypeError, ValueError):
                    pass

        # ``tool_calls`` left empty: the LM Studio backend does not yet
        # surface tool dispatch through this adapter. The agent loop
        # treats an empty tool_calls list as "model produced a final
        # reply", so the conversation terminates cleanly. ``_`` arg is
        # accepted but not used so the call signature matches Ollama's.
        _ = tools
        return {
            "message": {
                "role": "assistant",
                "content": "".join(content_parts),
                "thinking": "".join(thinking_parts),
                "tool_calls": [],
            },
            "prompt_eval_count": metrics.get("prompt_eval_count", 0),
            "eval_count": metrics.get("eval_count", 0),
        }
