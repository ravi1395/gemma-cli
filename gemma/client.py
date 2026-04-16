"""Ollama API wrapper.

Thin layer on top of the `ollama` Python client. Exposes two entry points:
  - ask(): single-shot query with optional streaming
  - chat(): multi-turn chat from a pre-built message list

Both functions yield (chunk_type, text) tuples where chunk_type is either
"think" (extended reasoning tokens) or "content" (response tokens). Thinking
chunks are only produced when config.thinking_mode is True and the model
supports it. Callers that only care about the final response can simply filter
for chunk_type == "content".
"""

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
    """
    client = ollama.Client(host=config.ollama_host)
    response = client.chat(
        model=config.model,
        messages=messages,
        stream=stream,
        think=config.thinking_mode,
        options={"temperature": config.temperature},
    )
    if stream:
        for chunk in response:
            thinking = (chunk["message"].get("thinking") or "")
            content = (chunk["message"].get("content") or "")
            if thinking:
                yield ("think", thinking)
            if content:
                yield ("content", content)
    else:
        thinking = (response["message"].get("thinking") or "")
        content = (response["message"].get("content") or "")
        if thinking:
            yield ("think", thinking)
        yield ("content", content)


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
