"""Theory-of-mind metric (Fig 6): SCoT opponent-prediction accuracy.

Compares each player's predicted opponent move (recorded only in SCoT runs)
against the opponent's actual move that round, over rounds where both the
prediction and the opponent's action are valid.
"""

from __future__ import annotations

import pandas as pd

from ..engine.payoff import is_valid_action
from ..players.base import MatchResult


def _prediction_hits(predictions, opponent_actions):
    """Yields (is_correct: int) for each round with a valid prediction and action."""
    for predicted, actual in zip(predictions, opponent_actions):
        if predicted is None or not is_valid_action(predicted):
            continue
        if not is_valid_action(actual):
            continue
        yield int(predicted == actual)


def prediction_accuracy_table(results: list[MatchResult]) -> pd.DataFrame:
    """Aggregates prediction accuracy per player.

    Args:
        results: Match results (SCoT runs carry predictions).

    Returns:
        A DataFrame with columns ``[player, accuracy, n_predictions]``.
    """
    rows: list[dict] = []
    for result in results:
        for hit in _prediction_hits(result.predictions_p1, result.actions_p2):
            rows.append({"player": result.player1_name, "hit": hit})
        for hit in _prediction_hits(result.predictions_p2, result.actions_p1):
            rows.append({"player": result.player2_name, "hit": hit})

    if not rows:
        return pd.DataFrame(columns=["player", "accuracy", "n_predictions"])
    frame = pd.DataFrame(rows)
    return (
        frame.groupby("player")["hit"].agg(["mean", "count"]).reset_index()
        .rename(columns={"mean": "accuracy", "count": "n_predictions"})
    )


def prediction_accuracy_trajectory(results: list[MatchResult]) -> pd.DataFrame:
    """Computes mean prediction accuracy per round (player-1 predictions).

    Args:
        results: Match results.

    Returns:
        A DataFrame with columns ``[player, round, accuracy]``.
    """
    rows: list[dict] = []
    for result in results:
        for rnd, (predicted, actual) in enumerate(
            zip(result.predictions_p1, result.actions_p2), start=1
        ):
            if predicted is None or not is_valid_action(predicted) or not is_valid_action(actual):
                continue
            rows.append({"player": result.player1_name, "round": rnd, "hit": int(predicted == actual)})
    if not rows:
        return pd.DataFrame(columns=["player", "round", "accuracy"])
    frame = pd.DataFrame(rows)
    return (
        frame.groupby(["player", "round"])["hit"].mean().reset_index()
        .rename(columns={"hit": "accuracy"})
    )
