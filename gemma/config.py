"""Central configuration for gemma-cli.

Defines the Config dataclass which holds all tunable parameters for the CLI
and (in later phases) the memory system. Kept as a plain dataclass so it is
easy to inspect, override via CLI flags, or extend.

Profile support: named TOML files at ~/.config/gemma/profiles/<name>.toml can
override any Config field. Load them with Config.load_profile(name) or via the
top-level --profile CLI flag.
"""

import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional


# TTL mapping by importance score (1-5). None = no expiry.
# High-importance memories persist; trivial ones expire quickly.
_DEFAULT_TTL_MAP: dict[int, Optional[int]] = {
    5: None,          # critical: no expiry
    4: 7 * 86400,     # high: 7 days
    3: 3 * 86400,     # medium: 72 hours
    2: 86400,         # low: 24 hours
    1: 6 * 3600,      # trivial: 6 hours
}


@dataclass
class Config:
    """Runtime configuration for the gemma CLI.

    Base CLI fields (Phase 1) and memory-system fields (Phase 2+) are both
    populated. Memory features are gated behind `memory_enabled` so the CLI
    still works if Redis or the embedding model is unavailable.
    """

    # --- Base CLI ---
    model: str = "gemma3:4b-it-q4_K_M"
    system_prompt: str = "You are a helpful assistant."
    temperature: float = 0.7
    # Ollama pre-allocates a KV cache sized to this value, independent of the
    # actual prompt length — 16k covers chat / ask / commit / sh / code review
    # comfortably while keeping KV cache in the hundreds of MB rather than
    # multiple GB. Bump via profile for long-document RAG or sessions that
    # approach 100+ turns, but expect a proportional jump in resident memory.
    context_window: int = 16384
    history_file: str = "~/.gemma_history.json"
    ollama_host: str = "http://localhost:11434"
    thinking_mode: bool = False  # Gemma 4 extended thinking — off by default (opt-in via --think or profile; roughly doubles tokens per query)
    # How long Ollama should keep the model resident in RAM between calls.
    # Any duration string accepted by Ollama ("30m", "2h", "-1" = forever, "0" = evict).
    # Keeps TTFT low across repeated CLI invocations in the same shell session.
    ollama_keep_alive: str = "2m"
    show_context_metrics: bool = True  # Print token-count footer after each response

    # --- Response cache ---
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600          # 0 = disable caching
    cache_temperature_max: float = 0.3     # skip caching above this temperature

    # --- Agent loop ---
    agent_max_turns: int = 8          # upper bound on tool-call turns per ask
    agent_tool_cache: bool = True     # enable per-session READ-tool memoization
    # Parallel tool dispatch (item #20). When the model emits multiple
    # tool_calls in a single turn, fan them out across a thread pool of
    # this size. Value is clamped to [1, 16] at use-time so a misconfigured
    # profile cannot create a thread storm; setting it to ``1`` restores
    # the legacy serial path and skips ``ThreadPoolExecutor`` entirely.
    agent_tool_concurrency: int = 4

    # --- Planner/executor split (item #19) ---
    # When True, the ``plan`` meta-tool is advertised and, on use,
    # switches the loop into executor mode (each step runs in its own
    # sub-conversation so large tool bodies stay out of the parent
    # log). Off by default while we gather field data; flip to True
    # per-profile once results are confirmed.
    plan_tool_enabled: bool = False
    # Maximum nesting depth for plan(). 1 means "one level of planning
    # only" — a step's sub-conversation may not itself call plan.
    # Higher values are supported but rarely useful; keep at 1.
    agent_max_plan_depth: int = 1
    # Interactive confirmation threshold: plans with MORE than this many
    # steps prompt the user for y/N before executing, in TTY sessions.
    # A value of 0 disables confirmation entirely.
    plan_confirm_threshold: int = 3
    # Lower bound on the per-step agent budget. Even when the parent
    # budget divided evenly across steps would give less, each step
    # still gets at least this many turns so it has a chance to
    # call-then-reply. A value of 2 is the practical minimum.
    plan_min_step_budget: int = 2

    # --- Web search tool (item #16) ---
    # Pluggable backend selector — see gemma/tools/backends/ for the
    # available implementations. ``duckduckgo`` requires no API key and
    # ships with the ``[agent]`` optional extra (``ddgs`` package).
    web_search_backend: str = "duckduckgo"
    web_search_max_per_turn: int = 3   # soft cap on search calls per agent turn
    web_search_timeout_s: int = 10     # wall-clock budget per backend call

    # --- Warm-start (item #4) ---
    # When True, ``main_callback`` spawns two short-lived daemon threads at
    # CLI startup: one issues a 1-token chat probe against ``cfg.model`` and
    # the other calls ``embed`` against ``cfg.embedding_model``. Both carry
    # ``cfg.ollama_keep_alive`` so Ollama retains the weights for the real
    # request that follows. The cost when the models are already resident is
    # a few ms of server load; the payoff when they have been evicted is the
    # 1–8 s cold-load tax disappearing from the user's first ``ask``.
    warm_start: bool = True
    # Set to True inside ``tests/conftest.py`` so CliRunner invocations never
    # accidentally spawn warm-up threads against a real Ollama. Production
    # code leaves this at False; users should not need to touch it.
    in_test_mode: bool = False

    # --- RAG indexer (items #9, #10) ---
    # Bounded parallelism for the embedding step of ``RAGIndexer``.
    # Each worker is given its *own* ``ollama.Client`` via
    # ``RAGIndexer._embedder_factory`` so requests don't serialise on a
    # shared HTTP session. Shipped at 1 (serial) so rollout is
    # observational; flip to 2 per-profile once field measurements land.
    # A value of 1 bypasses ``ThreadPoolExecutor`` entirely, preserving
    # the micro-benchmarked serial baseline.
    embed_concurrency: int = 1
    # Content-addressable cache of embed vectors keyed by
    # sha256(embed_input). Skipping an embed call is O(100 µs) vs.
    # O(30 ms) for a real Ollama round-trip, so this is the single
    # largest win on branch switches and reset+reindex.
    embed_cache_enabled: bool = True
    # TTL for cached embed vectors. 30 days keeps stale vectors from
    # accumulating forever while comfortably covering a sprint's worth
    # of reindex cycles on the same content.
    embed_cache_ttl_days: int = 30

    # --- Memory system ---
    memory_enabled: bool = True
    redis_url: str = "redis://localhost:6379/0"
    embedding_model: str = "nomic-embed-text"
    sliding_window_size: int = 8           # raw turns retained in context
    memory_top_k: int = 5                  # condensed memories retrieved per turn
    memory_min_similarity: float = 0.3     # cosine threshold for retrieval
    memory_conflict_threshold: float = 0.7 # cosine threshold for supersession
    memory_max_count: int = 200            # triggers reconsolidation above this
    condensation_async: bool = True        # background-thread condensation
    ttl_map: dict[int, Optional[int]] = field(
        default_factory=lambda: dict(_DEFAULT_TTL_MAP)
    )

    def resolved_history_path(self) -> Path:
        """Return the history file path with ~ expanded to the user's home."""
        return Path(self.history_file).expanduser()

    def ttl_for(self, importance: int) -> Optional[int]:
        """Return the TTL seconds for a memory at the given importance (1-5)."""
        # Clamp importance to the valid range before lookup
        bucket = max(1, min(5, int(importance)))
        return self.ttl_map.get(bucket)

    @classmethod
    def from_toml(cls, path: Path) -> "Config":
        """Load a Config from a TOML file, falling back to defaults for any missing field.

        Unknown keys in the TOML file emit a UserWarning (forward-compatibility).

        Args:
            path: Absolute or relative path to a readable .toml file.

        Returns:
            A Config with TOML values overlaid on dataclass defaults.
        """
        try:
            import tomllib  # stdlib, Python 3.11+
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Profile support requires Python 3.11+ (stdlib tomllib). "
                "Upgrade Python or install the 'tomli' back-port."
            ) from exc

        with open(path, "rb") as fh:
            data = tomllib.load(fh)

        valid = {f.name for f in fields(cls)}
        for key in data:
            if key not in valid:
                warnings.warn(
                    f"gemma profile {path}: unknown field {key!r} — ignored",
                    stacklevel=2,
                )

        known = {k: v for k, v in data.items() if k in valid}
        return cls(**known)

    @classmethod
    def load_profile(cls, name: str) -> "Config":
        """Load a named profile from ~/.config/gemma/profiles/<name>.toml.

        Args:
            name: Profile name (no extension). Maps to
                  ~/.config/gemma/profiles/<name>.toml.

        Returns:
            A Config populated from the profile file.

        Raises:
            FileNotFoundError: If the profile file does not exist.
        """
        profile_path = Path.home() / ".config" / "gemma" / "profiles" / f"{name}.toml"
        if not profile_path.exists():
            raise FileNotFoundError(
                f"Profile {name!r} not found. "
                f"Expected file: {profile_path}"
            )
        return cls.from_toml(profile_path)
