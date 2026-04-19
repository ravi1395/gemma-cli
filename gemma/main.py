"""Typer CLI entry point for gemma-cli.

Subcommands
-----------
Core:
  ask             Single-shot query (with optional memory)
  chat            Interactive REPL with persistent memory + sliding window
  pipe            Stateless: read input from stdin (no memory)
  history         Inspect session, list condensed memories, show stats

Memory power-user:
  remember        Seed a fact directly into the memory store (bypasses condensation)
  forget          Remove a memory (soft delete, keeps audit trail in Redis)
  pin             Set importance=5 on a memory so it never expires
  context         Preview what memory context would be injected for a query

Terminal-assistant:
  sh              Translate natural language to a shell command
  explain         Explain text, a file, a command, or an error
  commit          Generate a conventional-commit message from staged changes
  diff            Plain-English summary of git diff output
  why             Explain why the last shell command failed
  install-shell   Print/append the shell hook snippet for ``why``

All terminal-assistant commands are stateless by default (no memory read/write)
and behave correctly when stdout is not a TTY (pipe-safe).
"""

from __future__ import annotations

import dataclasses
import json
import sys
from typing import Any, Callable, Dict, List, Optional

import typer
from rich.console import Console
from rich.table import Table

from gemma.cache import ResponseCache
from gemma.client import chat as client_chat
from gemma.commands.clipboard import status_command as clipboard_status_command
from gemma.commands.completion import (
    install_command as completion_install_command,
    print_command as completion_print_command,
    status_command as completion_status_command,
    uninstall_command as completion_uninstall_command,
)
from gemma.commands.explain import explain_command
from gemma.commands.git import commit_command, diff_command
from gemma.commands.memory import (
    context_command,
    forget_command,
    pin_command,
    remember_command,
)
from gemma.commands.shell import (
    install_shell_command,
    sh_command,
    why_command,
)
from gemma.commands.rag import (
    index_command as rag_index_command,
    query_command as rag_query_command,
    reset_command as rag_reset_command,
    status_command as rag_status_command,
)
from gemma.commands.tools import (
    audit_command as tools_audit_command,
    list_command as tools_list_command,
    run_command as tools_run_command,
)
from gemma.completion import profile_completer
from gemma.config import Config
from gemma.history import SessionHistory
from gemma.memory import MemoryManager
from gemma.output import OutputMode, render_response
from gemma.session import GemmaSession


app = typer.Typer(help="Local CLI for Google's Gemma 4 model via Ollama.")
history_app = typer.Typer(help="Manage history, memories, and stats.")
app.add_typer(history_app, name="history")

# Clipboard subcommands (``gemma clipboard status`` today; room to grow).
clipboard_app = typer.Typer(help="Clipboard diagnostics and helpers.")
clipboard_app.command("status")(clipboard_status_command)
app.add_typer(clipboard_app, name="clipboard")

# Shell completion subcommands — install/print/status/uninstall.
# Kept in their own Typer group so the top-level help stays uncluttered.
completion_app = typer.Typer(help="Shell tab-completion installer and helpers.")
completion_app.command("install")(completion_install_command)
completion_app.command("print")(completion_print_command)
completion_app.command("status")(completion_status_command)
completion_app.command("uninstall")(completion_uninstall_command)
app.add_typer(completion_app, name="completion")

# Tool-use subsystem. ``gemma tools list`` shows every registered tool;
# ``gemma tools run <name>`` invokes one directly (same dispatcher the
# model uses). ``gemma tools audit`` tails the append-only JSONL log.
tools_app = typer.Typer(help="Sandboxed tool registry (experimental).")
tools_app.command("list")(tools_list_command)
tools_app.command("run")(tools_run_command)
tools_app.command("audit")(tools_audit_command)
app.add_typer(tools_app, name="tools")

# Local RAG over the workspace. ``gemma rag index`` builds or refreshes
# the namespace-scoped vector store in Redis; ``query`` asks a question;
# ``status`` prints what is currently indexed; ``reset`` wipes the
# namespace (useful when switching embedding models).
rag_app = typer.Typer(help="Local RAG over the workspace (experimental).")
rag_app.command("index")(rag_index_command)
rag_app.command("query")(rag_query_command)
rag_app.command("status")(rag_status_command)
rag_app.command("reset")(rag_reset_command)
app.add_typer(rag_app, name="rag")

# Active profile set by --profile flag in the top-level callback.
# None means "use Config dataclass defaults".
_active_profile: Optional[Config] = None

# Terminal-assistant commands (stateless, developer-workflow focused)
app.command("sh")(sh_command)
app.command("explain")(explain_command)
app.command("commit")(commit_command)
app.command("diff")(diff_command)
app.command("why")(why_command)
app.command("install-shell")(install_shell_command)

# Memory power-user commands
app.command("remember")(remember_command)
app.command("forget")(forget_command)
app.command("pin")(pin_command)
app.command("context")(context_command)

console = Console()


# -----------------------------------------------------------------------------
# Top-level callback — global flags (e.g. --profile) applied before subcommands
# -----------------------------------------------------------------------------

@app.callback()
def main_callback(
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        "-p",
        help=(
            "Load a named config profile from "
            "~/.config/gemma/profiles/<name>.toml before running a subcommand."
        ),
        # Dynamic completion: offers profile stems from
        # ``~/.config/gemma/profiles/*.toml``. Silent on errors so a
        # broken profiles dir never degrades the shell experience.
        autocompletion=profile_completer,
    ),
) -> None:
    """gemma – local Gemma 4 CLI via Ollama."""
    global _active_profile
    if profile:
        try:
            _active_profile = Config.load_profile(profile)
        except FileNotFoundError as exc:
            console.print(f"[red]error: {exc}[/red]")
            raise typer.Exit(code=1)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _make_config(
    model: Optional[str] = None,
    system: Optional[str] = None,
    memory_enabled: bool = True,
    thinking_mode: bool = False,
    keep_alive: Optional[str] = None,
) -> Config:
    """Build a Config, applying optional profile then explicit CLI overrides.

    Resolution order (highest wins):
      1. Explicit CLI flag (e.g. --model)
      2. Profile TOML field (set via --profile flag)
      3. Config dataclass default
    """
    # Start from a shallow copy of the active profile or bare defaults.
    cfg = dataclasses.replace(_active_profile) if _active_profile is not None else Config()
    if model:
        cfg.model = model
    if system:
        cfg.system_prompt = system
    cfg.memory_enabled = memory_enabled
    cfg.thinking_mode = thinking_mode
    if keep_alive:
        cfg.ollama_keep_alive = keep_alive
    return cfg


def _resolve_output_mode(
    json_flag: bool,
    only: Optional[str],
    code: bool,
) -> tuple[OutputMode, Optional[str]]:
    """Validate that at most one scripting output flag is set and return the mode.

    Args:
        json_flag: True if --json was passed.
        only:      Field name if --only was passed, else None.
        code:      True if --code was passed.

    Returns:
        A (OutputMode, field) pair where field is only set for ONLY mode.
    """
    set_count = sum([json_flag, only is not None, code])
    if set_count > 1:
        console.print("[red]error: --json, --only, and --code are mutually exclusive[/red]")
        raise typer.Exit(code=1)
    if json_flag:
        return OutputMode.JSON, None
    if only is not None:
        return OutputMode.ONLY, only
    if code:
        return OutputMode.CODE, None
    return OutputMode.RICH, None


# -----------------------------------------------------------------------------
# Agent-loop helpers
# -----------------------------------------------------------------------------

def _extract_tool_call(call: Any) -> tuple[str, Dict[str, Any]]:
    """Extract (name, args) from an Ollama tool-call object or dict.

    Handles two serialisation formats that appear in the wild:

    * **Ollama SDK format** — ``{"function": {"name": ..., "arguments": {...}}}``
      This is what a real ``ollama.Client.chat()`` response yields.
    * **Flat format** — ``{"name": ..., "arguments": {...}}``
      Used by bench-test stub clients so the loop can be exercised
      without a live Ollama server.

    Args:
        call: A single element from ``message["tool_calls"]``.

    Returns:
        ``(tool_name, args_dict)`` tuple, always strings/dicts even when
        the model returns arguments as a JSON-encoded string.
    """
    if isinstance(call, dict):
        if "function" in call:
            fn = call["function"]
            name = fn.get("name", "") if isinstance(fn, dict) else getattr(fn, "name", "")
            args = fn.get("arguments", {}) if isinstance(fn, dict) else getattr(fn, "arguments", {})
        else:
            name = call.get("name", "")
            args = call.get("arguments", {})
    else:
        # Ollama SDK Pydantic object — access via attributes.
        fn = getattr(call, "function", None)
        if fn is not None:
            name = getattr(fn, "name", "")
            args = getattr(fn, "arguments", {})
        else:
            name = getattr(call, "name", "")
            args = getattr(call, "arguments", {})

    # Some models serialise arguments as a JSON string rather than an object.
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, ValueError):
            args = {}

    return str(name), (args if isinstance(args, dict) else {})


def _agent_loop(
    client: Any,
    cfg: "Config",
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    budget: int,
    *,
    dispatch: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
    session_cache: Optional[Any] = None,
    session_id: str = "agent",
) -> tuple[str, bool]:
    """OpenAI-style tool-use loop for ``gemma ask``.

    Calls the model, dispatches any ``tool_calls``, appends
    ``role=tool`` results, and loops until the model replies without
    tool calls or ``budget`` turns are exhausted.

    Args:
        client:       Object with a ``.chat(model, messages, tools, ...)``
                      method (``ollama.Client`` or a test stub).
        cfg:          Active Config — provides model, temperature, etc.
        messages:     Message list, modified in place as turns accumulate.
        tools:        Tool schemas in Ollama/OpenAI format.
        budget:       Maximum number of model-call turns.
        dispatch:     Callable ``(name, args) → ToolResult | str``.  When
                      ``None``, tool calls are acknowledged but skipped.
        session_cache: ``AgentSessionCache`` for within-session memoization.
                      ``None`` means caching is disabled.
        session_id:   Audit record identifier.

    Returns:
        ``(final_reply_text, budget_exhausted)`` where
        ``budget_exhausted`` is True when all ``budget`` turns were
        consumed without a terminal reply.
    """
    from gemma.tools import audit as _audit

    last_content = ""

    for _turn in range(budget):
        # Non-streaming call so we see tool_calls in the complete response.
        raw = client.chat(
            model=cfg.model,
            messages=messages,
            tools=tools or [],
            think=cfg.thinking_mode,
            keep_alive=cfg.ollama_keep_alive,
            options={"temperature": cfg.temperature},
        )

        # Unify dict-style and Ollama SDK object access.
        msg = raw["message"] if isinstance(raw, dict) else raw.message
        if isinstance(msg, dict):
            last_content = msg.get("content", "") or ""
            raw_calls = msg.get("tool_calls") or []
        else:
            last_content = getattr(msg, "content", "") or ""
            raw_calls = getattr(msg, "tool_calls", None) or []

        if not raw_calls:
            # Model returned a plain reply — the loop is done.
            break

        # Record the assistant turn so the model sees its own requests.
        messages.append({
            "role": "assistant",
            "content": last_content,
            "tool_calls": list(raw_calls),
        })

        for call in raw_calls:
            # Normalise to a plain dict so _extract_tool_call is consistent.
            if not isinstance(call, dict):
                call = getattr(call, "__dict__", {}) or {"name": str(call)}
            tool_name, tool_args = _extract_tool_call(call)
            if not tool_name:
                continue

            # --- Session-cache fast path (READ-only tools) ---
            cache_hit = False
            tool_result_str = ""

            if session_cache is not None:
                cached_val = session_cache.get(tool_name, tool_args)
                if cached_val is not None:
                    cache_hit = True
                    tool_result_str = cached_val
                    _audit.append(_audit.make_record(
                        tool=tool_name,
                        capability="read",
                        args=tool_args,
                        session_id=session_id,
                        exit_code=0,
                        duration_ms=0,
                        approved_by="auto",
                        cached=True,
                    ))

            if not cache_hit:
                if dispatch is not None:
                    raw_result = dispatch(tool_name, tool_args)
                    # Real Dispatcher returns ToolResult; stubs return str.
                    tool_result_str = (
                        raw_result.content if hasattr(raw_result, "content")
                        else str(raw_result)
                    )
                else:
                    tool_result_str = f"(agent: tool {tool_name!r} not dispatched)"

                # Populate cache when this tool's capability is READ.
                if session_cache is not None:
                    try:
                        from gemma.tools import registry as _reg
                        spec, _ = _reg.get(tool_name)
                        if session_cache.is_cacheable(spec.capability.value):
                            session_cache.put(tool_name, tool_args, tool_result_str)
                    except KeyError:
                        pass  # Unregistered tool (e.g. bench stub) — skip.

            messages.append({"role": "tool", "name": tool_name, "content": tool_result_str})
    else:
        # Exhausted all budget turns without a terminal reply.
        return last_content, True

    return last_content, False


# -----------------------------------------------------------------------------
# ask
# -----------------------------------------------------------------------------

@app.command()
def ask(
    prompt: str = typer.Argument(..., help="The prompt to send to Gemma."),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    system: Optional[str] = typer.Option(None, "--system", "-s"),
    no_stream: bool = typer.Option(False, "--no-stream"),
    no_memory: bool = typer.Option(False, "--no-memory", help="Skip memory retrieval."),
    think: bool = typer.Option(False, "--think", help="Enable Gemma 4 extended thinking mode."),
    keep_alive: Optional[str] = typer.Option(
        None, "--keep-alive",
        help="Ollama model-residency duration (e.g. '30m', '2h', '-1' to pin, '0' to evict).",
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit a JSON object (for piping to jq)."),
    only: Optional[str] = typer.Option(None, "--only", help="Print a single JSON field (content, model, elapsed_ms, cache_hit)."),
    code: bool = typer.Option(False, "--code", help="Extract and emit only fenced code blocks."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the response cache."),
    cache_only: bool = typer.Option(
        False, "--cache-only",
        help="Error if no cache hit (requires --no-stream).",
    ),
    agent: bool = typer.Option(
        True, "--agent/--no-agent",
        help=(
            "Enable the tool-use agent loop (default: on). "
            "Pass --no-agent to skip tool dispatch and send a plain one-shot query."
        ),
    ),
) -> None:
    """Send a one-off prompt. Retrieves relevant memories unless --no-memory."""
    mode, field = _resolve_output_mode(json_output, only, code)
    cfg = _make_config(
        model=model, system=system, memory_enabled=not no_memory,
        thinking_mode=think, keep_alive=keep_alive,
    )

    with GemmaSession(cfg) as session:
        # Memory — only initialise when enabled; session.memory is lazy.
        memory = None
        if cfg.memory_enabled:
            mgr = session.memory
            if mgr.degraded:
                console.print(
                    "[yellow]memory: Redis unreachable, running without long-term memory[/yellow]"
                )
            memory = mgr if mgr.available else None

        if memory is not None:
            memory.record_turn("user", prompt)
            messages = memory.get_context_messages(prompt, system_prompt=cfg.system_prompt)
        else:
            messages = [
                {"role": "system", "content": cfg.system_prompt},
                {"role": "user", "content": prompt},
            ]

        # Stream-and-cache (#6): cache reads and writes are both valid on
        # the streaming path now. The writer still gates on a clean
        # stream completion so a truncated reply never lands in Redis.
        cache = ResponseCache.eligible(
            cfg, no_stream=True, no_cache=no_cache, prebuilt=session.cache
        )
        cached: Optional[str] = cache.get(messages, cfg) if cache else None

        if cached is not None:
            reply, _finished = render_response(
                iter([("content", cached)]), mode=mode, stream=False, field=field, model=cfg.model
            )
            if memory is not None and memory.available:
                memory.record_turn("assistant", reply)
            return

        if cache_only and no_stream:
            console.print("[red]gemma ask: no cache hit and --cache-only was set.[/red]")
            raise typer.Exit(code=1)

        # --- Agent loop path ---
        if agent:
            import ollama as _ollama

            # Register built-in tools and build the dispatcher.
            import gemma.tools.builtins  # noqa: F401 — triggers @tool decorators
            from gemma.tools.capabilities import GatingContext
            from gemma.tools.dispatcher import Dispatcher

            ctx = GatingContext(
                allow_writes=False,
                allow_network=True,
                is_tty=sys.stdout.isatty(),
                auto_approve_writes=False,
            )
            dispatcher = Dispatcher(
                ctx=ctx,
                session_id=getattr(session, "_session_id", "ask"),
                budget=cfg.agent_max_turns,
            )
            tool_schemas = dispatcher.advertised_schemas()
            ollama_client = _ollama.Client(host=cfg.ollama_host)

            session_cache = None
            if cfg.agent_tool_cache:
                from gemma.agent.cache import AgentSessionCache
                session_cache = AgentSessionCache()

            reply, _exhausted = _agent_loop(
                ollama_client, cfg, messages, tool_schemas,
                cfg.agent_max_turns,
                dispatch=dispatcher.dispatch,
                session_cache=session_cache,
                session_id=getattr(session, "_session_id", "ask"),
            )

            # Render the final reply via the standard output path.
            reply, finished = render_response(
                iter([("content", reply)]), mode=mode, stream=False, field=field, model=cfg.model
            )
        else:
            gen = client_chat(messages, cfg, stream=not no_stream)
            reply, finished = render_response(
                gen, mode=mode, stream=not no_stream, field=field, model=cfg.model
            )
            finished = True  # ensure cache write below triggers

        if memory is not None and memory.available:
            memory.record_turn("assistant", reply)

        if cache and reply and finished:
            cache.put(messages, cfg, reply)


# -----------------------------------------------------------------------------
# chat
# -----------------------------------------------------------------------------

@app.command()
def chat(
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    system: Optional[str] = typer.Option(None, "--system", "-s"),
    fresh: bool = typer.Option(False, "--fresh", help="Start with an empty sliding window."),
    no_memory: bool = typer.Option(False, "--no-memory", help="Disable all memory features."),
    think: bool = typer.Option(False, "--think", help="Enable Gemma 4 extended thinking mode."),
    keep_alive: Optional[str] = typer.Option(
        None, "--keep-alive",
        help="Ollama model-residency duration (e.g. '30m', '2h', '-1' to pin, '0' to evict).",
    ),
) -> None:
    """Interactive chat REPL with memory-augmented context."""
    cfg = _make_config(
        model=model, system=system, memory_enabled=not no_memory,
        thinking_mode=think, keep_alive=keep_alive,
    )

    with GemmaSession(cfg) as session:
        memory = session.memory
        if memory.degraded and cfg.memory_enabled:
            console.print(
                "[yellow]memory: Redis unreachable, running without long-term memory[/yellow]"
            )

        if fresh:
            memory.clear_session()

        mode = "memory" if memory.available else "degraded"
        console.print(
            f"[dim]gemma chat ({cfg.model}, {mode} mode) "
            f"-- type 'exit' or Ctrl-D to quit[/dim]"
        )

        while True:
            try:
                user_input = console.input("[bold cyan]you> [/bold cyan]").strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                break
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", ":q"}:
                break

            memory.record_turn("user", user_input)
            messages = memory.get_context_messages(user_input, system_prompt=cfg.system_prompt)

            console.print("[bold green]gemma>[/bold green] ", end="")
            gen = client_chat(messages, cfg, stream=True)
            reply, _finished = render_response(gen, mode=OutputMode.RICH, stream=True, model=cfg.model)
            memory.record_turn("assistant", reply)


# -----------------------------------------------------------------------------
# pipe
# -----------------------------------------------------------------------------

@app.command()
def pipe(
    instruction: str = typer.Argument("Analyze this input.", help="What to do with the piped content."),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    no_stream: bool = typer.Option(False, "--no-stream"),
    json_output: bool = typer.Option(False, "--json", help="Emit a JSON object (for piping to jq)."),
    only: Optional[str] = typer.Option(None, "--only", help="Print a single JSON field (content, model, elapsed_ms, cache_hit)."),
    code: bool = typer.Option(False, "--code", help="Extract and emit only fenced code blocks."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the response cache."),
    cache_only: bool = typer.Option(
        False, "--cache-only",
        help="Error if no cache hit (requires --no-stream).",
    ),
) -> None:
    """Read stdin and send it together with an instruction. Stateless (no memory)."""
    mode, field = _resolve_output_mode(json_output, only, code)

    if sys.stdin.isatty():
        console.print("[red]gemma pipe: expected input on stdin[/red]")
        raise typer.Exit(code=2)

    piped = sys.stdin.read()
    cfg = _make_config(model=model, memory_enabled=False)
    combined = f"{instruction}\n\n---\n\n{piped}"
    messages = [
        {"role": "system", "content": cfg.system_prompt},
        {"role": "user", "content": combined},
    ]

    with GemmaSession(cfg) as session:
        # Stream-and-cache (#6): see ``ask`` for the full rationale.
        cache = ResponseCache.eligible(
            cfg, no_stream=True, no_cache=no_cache, prebuilt=session.cache
        )
        cached: Optional[str] = cache.get(messages, cfg) if cache else None

        if cached is not None:
            render_response(
                iter([("content", cached)]), mode=mode, stream=False, field=field, model=cfg.model
            )
            return

        if cache_only and no_stream:
            console.print("[red]gemma pipe: no cache hit and --cache-only was set.[/red]")
            raise typer.Exit(code=1)

        gen = client_chat(messages, cfg, stream=not no_stream)
        result, finished = render_response(gen, mode=mode, stream=not no_stream, field=field, model=cfg.model)

        if cache and result and finished:
            cache.put(messages, cfg, result)


# -----------------------------------------------------------------------------
# history subcommands
# -----------------------------------------------------------------------------

@history_app.command("show")
def history_show() -> None:
    """Print the current session (JSON fallback history, if any)."""
    cfg = Config()
    history = SessionHistory(cfg)
    turns = history.show()
    if not turns:
        console.print("[dim]no history[/dim]")
        return
    for turn in turns:
        role = turn.get("role", "?")
        content = turn.get("content", "")
        color = {"system": "magenta", "user": "cyan", "assistant": "green"}.get(role, "white")
        console.print(f"[bold {color}]{role}>[/bold {color}] {content}")


@history_app.command("clear")
def history_clear() -> None:
    """Wipe both the JSON fallback history and the current-session sliding window."""
    cfg = Config()
    SessionHistory(cfg).clear()
    mgr = MemoryManager(cfg)
    mgr.initialize()
    mgr.clear_session()
    console.print("[yellow]history cleared (condensed memories preserved)[/yellow]")


@history_app.command("memories")
def history_memories(
    limit: int = typer.Option(50, "--limit", "-n", help="Max number of memories to show."),
) -> None:
    """List condensed memories stored in Redis, sorted by importance."""
    cfg = Config()
    mgr = MemoryManager(cfg)
    mgr.initialize()

    if not mgr.available:
        console.print("[yellow]memory system unavailable[/yellow]")
        return

    records = mgr.list_memories(limit=limit)
    if not records:
        console.print("[dim]no condensed memories yet[/dim]")
        return

    table = Table(title=f"Condensed memories ({len(records)})")
    table.add_column("Importance", justify="right", style="bold")
    table.add_column("Category", style="cyan")
    table.add_column("Content")
    for r in records:
        table.add_row(str(r.importance), r.category.value, r.content)
    console.print(table)


@history_app.command("stats")
def history_stats() -> None:
    """Print memory-system statistics."""
    cfg = Config()
    mgr = MemoryManager(cfg)
    mgr.initialize()
    stats = mgr.get_stats()

    if not stats.get("available"):
        console.print("[yellow]memory system unavailable[/yellow]")
        console.print(f"fallback turns: {stats.get('fallback_turns', 0)}")
        return

    console.print(f"[bold]session:[/bold] {stats['session_id']}")
    console.print(f"[bold]active memories:[/bold] {stats['active_memories']}")
    console.print(f"[bold]window turns:[/bold] {stats['window_turns']}")

    if stats["by_category"]:
        console.print("\n[bold]by category:[/bold]")
        for cat, count in sorted(stats["by_category"].items()):
            console.print(f"  {cat}: {count}")

    if stats["by_importance"]:
        console.print("\n[bold]by importance:[/bold]")
        for imp in sorted(stats["by_importance"].keys(), reverse=True):
            console.print(f"  {imp}: {stats['by_importance'][imp]}")


if __name__ == "__main__":
    app()
