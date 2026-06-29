"""Round-robin tournament: turns a :class:`RunSpec` into match results + a tidy CSV.

Builds the player roster (LLM models behind cached providers + hand-coded
strategies), resolves the game set, plays every applicable ordered pairing, and
persists one row per round. Fresh player instances are created per match so
per-match state (SCoT prediction cache, strategy RNG) never leaks between matches.
"""

from __future__ import annotations

import itertools
import logging
import random
import string
import time
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

_LOGGER = logging.getLogger(__name__)
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
    thoughts_csv: Path | None = None


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
            base_url=spec.base_url,
            timeout=spec.params.request_timeout,
        )
        name = f"{spec.id}" + ("+scot" if scot else "")

        def factory(model=model, name=name) -> Player:
            return LLMPlayer(name, model, framing, scot=scot, reasoning=run.reasoning)

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


def _thought_rows(result, game: Game) -> list[dict]:
    """Serializes one match's per-round reasoning capture (one row per player-seat).

    Args:
        result: The completed match result.
        game: The game played.

    Returns:
        Rows with the SCoT prediction and raw reply text for each seat, per round.
    """
    rows: list[dict] = []
    seats = (
        (result.player1_name, result.player2_name, result.thoughts_p1, result.actions_p1),
        (result.player2_name, result.player1_name, result.thoughts_p2, result.actions_p2),
    )
    for player, opponent, thoughts, actions in seats:
        for rnd, thought in enumerate(thoughts):
            if not thought:
                continue
            rows.append(
                {
                    "game_name": game.name,
                    "player": player,
                    "opponent": opponent,
                    "round": rnd + 1,
                    "action": actions[rnd],
                    "predicted": thought.get("predicted") or "",
                    "predict_text": thought.get("predict_text", ""),
                    "decide_text": thought.get("decide_text", ""),
                }
            )
    return rows


def run_tournament(run) -> RunResult:
    """Runs the full tournament described by ``run`` and writes per-round CSVs.

    Args:
        run: A :class:`~llmgames.config.schema.RunSpec`.

    Returns:
        A :class:`RunResult` with results and artifact paths.
    """
    start = time.monotonic()
    configure_cache(run.cache.backend, run.cache.dsn)
    llm_factories = _build_llm_factories(run)
    strat_factories = _build_strategy_factories(run)
    roster = llm_factories + strat_factories
    llm_names = {name for name, _ in llm_factories}

    games = resolve_games(run.games)
    output_dir = Path(run.output_dir) / run.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Total matches up front (for progress), then a throttled per-match log.
    n_llm, n_strat = len(llm_factories), len(strat_factories)
    pairs = len(roster) ** 2
    if not run.self_play:
        pairs -= n_strat ** 2  # drop strategy-vs-strategy
    if not run.model_vs_model:
        pairs -= n_llm ** 2  # drop model-vs-model
    total = len(games) * pairs
    log_every = 1 if total <= 300 else max(1, total // 100)
    _LOGGER.info(
        "Run '%s' started: %d models + %d strategies, %d game(s), mode=%s%s -> %d matches",
        run.name, len(llm_factories), len(strat_factories), len(games), run.mode,
        " +reasoning" if run.reasoning else "", total,
    )

    rows: list[dict] = []
    thought_rows: list[dict] = []
    results = []
    match_counter = 0

    for game in games:
        for (name1, make1), (name2, make2) in itertools.product(roster, roster):
            if not run.self_play and not (name1 in llm_names or name2 in llm_names):
                continue
            if not run.model_vs_model and name1 in llm_names and name2 in llm_names:
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
            thought_rows.extend(_thought_rows(result, game))

            if match_counter % log_every == 0 or match_counter == total:
                _LOGGER.info(
                    "[%d/%d] %s: %s vs %s (%.0fs elapsed)",
                    match_counter, total, game.name, name1, name2, time.monotonic() - start,
                )

    rounds_csv = output_dir / "rounds.csv"
    pd.DataFrame(rows).to_csv(rounds_csv, index=False)

    thoughts_csv: Path | None = None
    if thought_rows:
        thoughts_csv = output_dir / "thoughts.csv"
        pd.DataFrame(thought_rows).to_csv(thoughts_csv, index=False)

    _LOGGER.info(
        "Run '%s' FINISHED: %d matches, %d rounds in %.0fs -> %s%s",
        run.name, match_counter, len(rows), time.monotonic() - start, rounds_csv,
        f" (+ {thoughts_csv})" if thoughts_csv else "",
    )

    return RunResult(
        results=results, rounds_csv=rounds_csv, output_dir=output_dir, thoughts_csv=thoughts_csv
    )
