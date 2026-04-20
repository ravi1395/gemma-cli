"""Ollama API wrapper.

Thin layer on top of the `ollama` Python client. Exposes two entry points:
  - ask(): single-shot query with optional streaming
  - chat(): multi-turn chat from a pre-built message list

Both functions yield (chunk_type, text) tuples where chunk_type is either
"think" (extended reasoning tokens), "content" (response tokens), or
"metrics" (a JSON string with token counts, yielded once at the end of each
response). Callers that only care about the final response can filter for
chunk_type == "content".
"""

import json
from typing import Generator, Optional

import ollama

from gemma.config import Config


def chat(
    messages: list[dict],
    config: Config,
    stream: bool = True,
) -> Generator[tuple[str, str], None, None]:
    """Send a pre-built message list to Ollama and yield (chunk_type, text) tuples.

    Args:
        messages: List of {"role": ..., "content": ...} dicts.
        config: Runtime configuration (provides model, temperature, host,
                thinking_mode).
        stream: If True, yield chunks as they arrive. If False, yield the
                full response as a single tuple.
    Yields:
        ("think", text) for extended-reasoning tokens (thinking_mode=True only).
        ("content", text) for response tokens.
        ("metrics", json_str) once at the end with prompt/completion token counts.
    """
    client = ollama.Client(host=config.ollama_host)
    response = client.chat(
        model=config.model,
        messages=messages,
        stream=stream,
        think=config.thinking_mode,
        # keep_alive tells Ollama how long to hold the model in RAM after this
        # call. Warm re-use across invocations is worth ~1-2s of TTFT.
        keep_alive=config.ollama_keep_alive,
        options={"temperature": config.temperature},
    )
    if stream:
        last_chunk: object = {}
        for chunk in response:
            last_chunk = chunk
            thinking = (chunk["message"].get("thinking") or "")
            content = (chunk["message"].get("content") or "")
            if thinking:
                yield ("think", thinking)
            if content:
                yield ("content", content)
        # Last streaming chunk carries Ollama's token-count summary fields.
        yield ("metrics", json.dumps(_extract_metrics(last_chunk)))
    else:
        thinking = (response["message"].get("thinking") or "")
        content = (response["message"].get("content") or "")
        if thinking:
            yield ("think", thinking)
        yield ("content", content)
        yield ("metrics", json.dumps(_extract_metrics(response)))


def _extract_metrics(chunk: object) -> dict:
    """Pull prompt/completion token counts from an Ollama response chunk.

    Handles both dict-style (raw API) and Pydantic-object style (SDK).

    Args:
        chunk: The last chunk from a streaming response, or the full non-streaming response.

    Returns:
        Dict with 'prompt_eval_count' and 'eval_count' (ints, default 0).
    """
    def _get(key: str) -> int:
        if isinstance(chunk, dict):
            return int(chunk.get(key) or 0)
        return int(getattr(chunk, key, 0) or 0)

    return {
        "prompt_eval_count": _get("prompt_eval_count"),
        "eval_count": _get("eval_count"),
    }


def ask(
    prompt: str,
    config: Config,
    system_prompt: Optional[str] = None,
    stream: bool = True,
) -> Generator[tuple[str, str], None, None]:
    """Single-shot query with no history.

    Args:
        prompt: The user query.
        config: Runtime configuration.
        system_prompt: Override the default system prompt for this call.
        stream: Whether to stream chunks.
    """
    messages = [
        {"role": "system", "content": system_prompt or config.system_prompt},
        {"role": "user", "content": prompt},
    ]
    yield from chat(messages, config, stream=stream)
