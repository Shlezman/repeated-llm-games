"""Battle of the Sexes metrics: collaboration rate, P1-preferred frequency (Fig 5).

"Successful collaboration" = both players choose the same option in a round.
"P1-preferred" = both choose player 1's preferred option (ACTION_A). Matrices are
indexed by (player1, player2) for heat-map rendering.
"""

from __future__ import annotations

import pandas as pd

from ..engine.game import ACTION_A
from ..engine.outcomes import is_coordinated, is_preferred_equilibrium
from ..engine.payoff import is_valid_action
from ..players.base import MatchResult


def _valid_rounds(result: MatchResult):
    """Yields (action1, action2) pairs for rounds with two valid actions."""
    for a1, a2 in zip(result.actions_p1, result.actions_p2):
        if is_valid_action(a1) and is_valid_action(a2):
            yield a1, a2


def _rate_matrix(results: list[MatchResult], predicate) -> pd.DataFrame:
    """Builds a (player1 x player2) matrix of the rate at which ``predicate`` holds."""
    rows: list[dict] = []
    for result in results:
        pairs = list(_valid_rounds(result))
        if not pairs:
            continue
        rate = sum(1 for a1, a2 in pairs if predicate(a1, a2)) / len(pairs)
        rows.append({"player1": result.player1_name, "player2": result.player2_name, "rate": rate})
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return frame.pivot_table(index="player1", columns="player2", values="rate", aggfunc="mean")


def collaboration_rate_matrix(results: list[MatchResult]) -> pd.DataFrame:
    """Returns the (player1 x player2) successful-collaboration-rate matrix."""
    return _rate_matrix(results, lambda a1, a2: is_coordinated(a1, a2))


def preferred_frequency_matrix(results: list[MatchResult]) -> pd.DataFrame:
    """Returns the (player1 x player2) rate of landing on P1's preferred equilibrium."""
    return _rate_matrix(
        results, lambda a1, a2: is_preferred_equilibrium(a1, a2, preferred_action=ACTION_A)
    )


def collaboration_trajectory(results: list[MatchResult]) -> pd.DataFrame:
    """Computes the mean collaboration rate per round across pairings.

    Args:
        results: BoS match results.

    Returns:
        A DataFrame with columns ``[pairing, round, collaboration_rate]``.
    """
    rows: list[dict] = []
    for result in results:
        pairing = f"{result.player1_name} vs {result.player2_name}"
        for rnd, (a1, a2) in enumerate(zip(result.actions_p1, result.actions_p2), start=1):
            if is_valid_action(a1) and is_valid_action(a2):
                rows.append({"pairing": pairing, "round": rnd, "collab": int(is_coordinated(a1, a2))})
    if not rows:
        return pd.DataFrame(columns=["pairing", "round", "collaboration_rate"])
    frame = pd.DataFrame(rows)
    return (
        frame.groupby(["pairing", "round"])["collab"].mean().reset_index()
        .rename(columns={"collab": "collaboration_rate"})
    )
