"""Prisoner's Dilemma metrics: defection rates, accrued scores, trajectories (Fig 3).

Operates on the subset of match results for the PD game. Matrices are indexed by
(player1, player2) so they can be rendered as heat maps; rounds with invalid
actions are ignored when computing rates.
"""

from __future__ import annotations

import pandas as pd

from ..engine.game import ACTION_A, ACTION_B
from ..engine.payoff import is_valid_action
from ..players.base import MatchResult


def _valid_pairs(actions_self, actions_opp):
    """Yields (self_action, opp_action) pairs where both actions are valid."""
    for mine, theirs in zip(actions_self, actions_opp):
        if is_valid_action(mine) and is_valid_action(theirs):
            yield mine, theirs


def action_rate_matrix(results: list[MatchResult], target_action: str = ACTION_B) -> pd.DataFrame:
    """Builds a (player1 x player2) matrix of player 1's rate of ``target_action``.

    Args:
        results: PD match results.
        target_action: The action to measure (default B = defect).

    Returns:
        A pivot DataFrame; cell = fraction of player 1's valid actions equal to target.
    """
    rows: list[dict] = []
    for result in results:
        pairs = list(_valid_pairs(result.actions_p1, result.actions_p2))
        if not pairs:
            continue
        rate = sum(1 for mine, _ in pairs if mine == target_action) / len(pairs)
        rows.append({"player1": result.player1_name, "player2": result.player2_name, "rate": rate})
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return frame.pivot_table(index="player1", columns="player2", values="rate", aggfunc="mean")


def defection_rate_matrix(results: list[MatchResult]) -> pd.DataFrame:
    """Returns the (player1 x player2) defection-rate matrix."""
    return action_rate_matrix(results, target_action=ACTION_B)


def cooperation_rate_matrix(results: list[MatchResult]) -> pd.DataFrame:
    """Returns the (player1 x player2) cooperation-rate matrix."""
    return action_rate_matrix(results, target_action=ACTION_A)


def score_matrix(results: list[MatchResult]) -> pd.DataFrame:
    """Builds a (player1 x player2) matrix of player 1's mean total score."""
    rows = [
        {"player1": r.player1_name, "player2": r.player2_name, "total": r.total_p1}
        for r in results
    ]
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return frame.pivot_table(index="player1", columns="player2", values="total", aggfunc="mean")


def cooperation_trajectory(results: list[MatchResult]) -> pd.DataFrame:
    """Computes each player's mean cooperation rate per round, across both seats.

    Args:
        results: PD match results.

    Returns:
        A DataFrame with columns ``[player, round, cooperation_rate]``.
    """
    rows: list[dict] = []
    for result in results:
        seats = (
            (result.player1_name, result.actions_p1),
            (result.player2_name, result.actions_p2),
        )
        for player_name, actions in seats:
            for rnd, action in enumerate(actions, start=1):
                if is_valid_action(action):
                    rows.append(
                        {"player": player_name, "round": rnd, "coop": int(action == ACTION_A)}
                    )
    if not rows:
        return pd.DataFrame(columns=["player", "round", "cooperation_rate"])
    frame = pd.DataFrame(rows)
    return (
        frame.groupby(["player", "round"])["coop"].mean().reset_index()
        .rename(columns={"coop": "cooperation_rate"})
    )
