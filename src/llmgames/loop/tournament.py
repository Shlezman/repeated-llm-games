"""Round-robin tournament: turns a :class:`RunSpec` into match results + a tidy CSV.

Builds the player roster (LLM models behind cached providers + hand-coded
strategies), resolves the game set, plays every applicable ordered pairing, and
persists one row per round. Fresh player instances are created per match so
per-match state (SCoT prediction cache, strategy RNG) never leaks between matches.
"""

from __future__ import annotations

import itertools
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from ..engine.families import canonical_bos, canonical_pd, filter_by_family, load_games_csv
from ..engine.game import Game
from ..players.base import Player
from ..players.llm_player import LLMPlayer
from ..players.strategies import make_strategy
from ..prompts.transforms import DEFAULT_LABELS, build_framing, order_sequence
from ..providers.cache import configure_cache
from ..providers.models import build_chat_model
from .match import play_match

PlayerFactory = Callable[[], Player]


@dataclass(frozen=True)
class RunResult:
    """Outputs of a tournament run.

    Attributes:
        results: Every match result, in play order.
        rounds_csv: Path to the tidy per-round CSV.
        output_dir: Directory holding all run artifacts.
    """

    results: list
    rounds_csv: Path
    output_dir: Path


def resolve_games(selector) -> list[Game]:
    """Resolves the configured game set into concrete :class:`Game` objects.

    Args:
        selector: A :class:`~llmgames.config.schema.GameSelector`.

    Returns:
        The ordered list of games to play (canonical first, then taxonomy).
    """
    games: list[Game] = []
    if "pd" in selector.canonical:
        games.append(canonical_pd())
    if "bos" in selector.canonical:
        games.append(canonical_bos())
    if selector.all_144 or selector.families:
        taxonomy = load_games_csv(selector.games_csv)
        games.extend(taxonomy if selector.all_144 else filter_by_family(taxonomy, selector.families))
    return games


def _resolve_labels(spec_labels, seed: int) -> tuple[str, str]:
    """Resolves robustness label config into a concrete two-letter pair.

    Args:
        spec_labels: ``None`` (canonical J/F), "random" (seeded uppercase pair), or
            an explicit two-element list.
        seed: Run seed used for the "random" option.

    Returns:
        The (label_a, label_b) display labels.
    """
    if spec_labels is None:
        return DEFAULT_LABELS
    if spec_labels == "random":
        pair = random.Random(seed).sample(string.ascii_uppercase, 2)
        return (pair[0], pair[1])
    return (spec_labels[0], spec_labels[1])


def _build_llm_factories(run) -> list[tuple[str, PlayerFactory]]:
    """Builds (name, factory) pairs for each configured model.

    Args:
        run: The run specification.

    Returns:
        A list of ``(player_name, factory)`` tuples. The chat model is shared
        (stateless); a fresh :class:`LLMPlayer` (graph + per-round state) is built
        per match.
    """
    framing = build_framing(
        cover_story=run.robustness.cover_story,
        utility_label=run.robustness.utility_label,
        labels=_resolve_labels(run.robustness.labels, run.seed),
    )
    scot = run.mode == "scot"
    factories: list[tuple[str, PlayerFactory]] = []
    for spec in run.models:
        model = build_chat_model(
            spec.provider,
            spec.model,
            temperature=spec.params.temperature,
            max_tokens=spec.params.max_tokens,
        )
        name = f"{spec.id}" + ("+scot" if scot else "")

        def factory(model=model, name=name) -> Player:
            return LLMPlayer(name, model, framing, scot=scot)

        factories.append((name, factory))
    return factories


def _build_strategy_factories(run) -> list[tuple[str, PlayerFactory]]:
    """Builds (name, factory) pairs for each configured opponent strategy."""
    factories: list[tuple[str, PlayerFactory]] = []
    for name in run.opponents:
        def factory(name=name, seed=run.seed) -> Player:
            return make_strategy(name, seed=seed)
        factories.append((name, factory))
    return factories


def _match_rows(result, game: Game, mode: str) -> list[dict]:
    """Serializes one match into per-round CSV rows with valid-round running totals.

    Args:
        result: The completed match result.
        game: The game played.
        mode: Run mode ("base" or "scot").

    Returns:
        One dict per round.
    """
    totals1 = result.running_totals(for_player1=True)
    totals2 = result.running_totals(for_player1=False)
    return [
        {
            "game_name": game.name,
            "family": game.family,
            "mode": mode,
            "player1": result.player1_name,
            "player2": result.player2_name,
            "round": rnd + 1,
            "action1": result.actions_p1[rnd],
            "action2": result.actions_p2[rnd],
            "points1": result.points_p1[rnd],
            "points2": result.points_p2[rnd],
            "total1": totals1[rnd],
            "total2": totals2[rnd],
            "pred1": result.predictions_p1[rnd],
            "pred2": result.predictions_p2[rnd],
        }
        for rnd in range(len(result.actions_p1))
    ]


def run_tournament(run) -> RunResult:
    """Runs the full tournament described by ``run`` and writes a per-round CSV.

    Args:
        run: A :class:`~llmgames.config.schema.RunSpec`.

    Returns:
        A :class:`RunResult` with results and artifact paths.
    """
    configure_cache(run.cache.backend, run.cache.dsn)
    llm_factories = _build_llm_factories(run)
    strat_factories = _build_strategy_factories(run)
    roster = llm_factories + strat_factories
    llm_names = {name for name, _ in llm_factories}

    games = resolve_games(run.games)
    output_dir = Path(run.output_dir) / run.name
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    results = []
    match_counter = 0

    for game in games:
        for (name1, make1), (name2, make2) in itertools.product(roster, roster):
            if not run.self_play and not (name1 in llm_names or name2 in llm_names):
                continue

            orders = order_sequence(
                seed=run.seed + match_counter,
                num_rounds=run.rounds,
                randomize=run.robustness.randomize_order,
            )
            match_counter += 1

            result = play_match(game, make1(), make2(), num_rounds=run.rounds, orders=orders)
            results.append(result)
            rows.extend(_match_rows(result, game, run.mode))

    rounds_csv = output_dir / "rounds.csv"
    pd.DataFrame(rows).to_csv(rounds_csv, index=False)
    return RunResult(results=results, rounds_csv=rounds_csv, output_dir=output_dir)
