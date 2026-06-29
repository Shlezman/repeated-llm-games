"""Typed run configuration (pydantic) — the single source of truth for a study run.

A run is fully described by a :class:`RunSpec`: which models play (each behind a
provider adapter), which games, base vs SCoT, robustness toggles, opponents, cache
backend, and output location. No model identifiers are hardcoded anywhere in the
codebase — they enter only through :class:`ModelSpec`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ParamsSpec(BaseModel):
    """Generation parameters passed through to the chat model.

    Attributes:
        temperature: Sampling temperature; 0.0 for deterministic study runs.
        max_tokens: Max output tokens (kept small but >1 so chat models can answer).
        request_timeout: Per-call timeout in seconds. A hung/slow gateway call fails
            fast and becomes an unparseable round instead of freezing the run.
    """

    temperature: float = 0.0
    max_tokens: int = 8
    request_timeout: float = 30.0


class ModelSpec(BaseModel):
    """A model under test, selected entirely by configuration.

    Attributes:
        id: Stable label used in results (e.g. "model_a").
        provider: LangChain backend selector ("openai", "anthropic", "auto", "mock",
            ...). "auto" lets LangChain infer the provider from the model id. Use
            "openai" for an OpenAI-compatible gateway (e.g. the internal llm-gw).
        model: The model identifier passed to the backend. Required — never
            defaulted — so no model name is ever implied by code.
        base_url: Optional API base URL for OpenAI-compatible gateways/proxies. The
            API key is read from the environment by the integration (never here).
        params: Generation parameters.
    """

    id: str
    provider: str = "auto"
    model: str
    base_url: str | None = None
    params: ParamsSpec = Field(default_factory=ParamsSpec)


class RobustnessSpec(BaseModel):
    """Game-preserving robustness transforms (all off-by-default-safe).

    Attributes:
        randomize_order: Shuffle which option is offered first each round (seeded).
        utility_label: Unit word in the payoff text ("points", "dollars", "coins").
        cover_story: Narrative framing ("none", "cooking", "project").
        labels: Display labels for the two actions. ``None`` uses the canonical
            ("J", "F") (appropriate for PD/BoS); "random" draws a seeded uppercase
            pair per run (the paper's all-144 protocol to avoid label priming); or
            give an explicit two-letter list.
    """

    randomize_order: bool = True
    utility_label: str = "points"
    cover_story: Literal["none", "cooking", "project"] = "none"
    labels: list[str] | Literal["random"] | None = None


class CacheSpec(BaseModel):
    """Response cache configuration.

    Attributes:
        backend: "postgres" (default; reads DATABASE_URL via SQLAlchemy) or "memory".
        dsn: Optional explicit DSN; falls back to the DATABASE_URL env var.
    """

    backend: Literal["postgres", "memory"] = "postgres"
    dsn: str | None = None


class GameSelector(BaseModel):
    """Which games to run.

    Attributes:
        canonical: Canonical games by key ("pd", "bos").
        families: Family names from the 144-game taxonomy (e.g. "Win-win").
        all_144: Include every game in the 144-game taxonomy.
        games_csv: Path to the taxonomy CSV (used by families/all_144).
    """

    canonical: list[Literal["pd", "bos"]] = Field(default_factory=lambda: ["pd", "bos"])
    families: list[str] = Field(default_factory=list)
    all_144: bool = False
    games_csv: str = "config/games.csv"


class RunSpec(BaseModel):
    """A complete, reproducible study run.

    Attributes:
        name: Run name, used for the output directory.
        rounds: Rounds per match (paper default: 10).
        seed: Master seed for all randomized arms.
        mode: "base" or "scot" (Social Chain-of-Thought).
        reasoning: If true, prompts ask the model to explain briefly before choosing;
            the rationale is captured to thoughts.csv and shown in the replay.
        models: Models under test (>= 1).
        opponents: Hand-coded strategy names to include as opponents.
        self_play: Whether models also play against each other / themselves.
        model_vs_model: If false, skip LLM-vs-LLM matches (models play only the
            hand-coded strategies) — useful to keep SCoT/reasoning runs affordable.
        games: Which games to run.
        robustness: Robustness transforms.
        cache: Cache backend configuration.
        output_dir: Where results (CSVs, figures, results.md) are written.
    """

    name: str = "paper_default"
    rounds: int = 10
    seed: int = 42
    mode: Literal["base", "scot"] = "base"
    reasoning: bool = False
    models: list[ModelSpec] = Field(min_length=1)
    opponents: list[str] = Field(default_factory=list)
    self_play: bool = True
    model_vs_model: bool = True
    games: GameSelector = Field(default_factory=GameSelector)
    robustness: RobustnessSpec = Field(default_factory=RobustnessSpec)
    cache: CacheSpec = Field(default_factory=CacheSpec)
    output_dir: str = "results"
