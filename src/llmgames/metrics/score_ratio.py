"""Table 1 metric: score ratio = achieved score / best achievable given the opponent.

For each round the best achievable is the payoff of this player's best response to
the opponent's *actual* action; the ratio is achieved/best over valid rounds. This
"under-ideal-play" benchmark is the default; ``method="max_cell"`` instead divides
by the game's single highest payoff cell. Rounds with an unparseable/invalid action
on either side are skipped.
"""

from __future__ import annotations

import pandas as pd

from ..engine.game import ACTIONS
from ..engine.payoff import is_valid_action
from ..players.base import MatchResult


def _best_round_payoff(result: MatchResult, opp_action: str, *, am_row_player: bool) -> int:
    """Returns this player's best-response payoff to a fixed opponent action."""
    return max(
        result.game.payoff_for(mine, opp_action, am_row_player=am_row_player)[0]
        for mine in ACTIONS
    )


def _max_cell_payoff(result: MatchResult, *, am_row_player: bool) -> int:
    """Returns the single highest payoff this player can score in any cell."""
    return max(
        result.game.payoff_for(mine, theirs, am_row_player=am_row_player)[0]
        for mine in ACTIONS
        for theirs in ACTIONS
    )


def match_score_ratio(
    result: MatchResult, *, for_player1: bool, method: str = "best_response"
) -> float | None:
    """Computes one player's score ratio for a single match.

    Args:
        result: The match result.
        for_player1: True to score the row player, False for the column player.
        method: "best_response" (default) or "max_cell".

    Returns:
        The score ratio in [0, 1], or None if no valid rounds exist.
    """
    my_actions = result.actions_p1 if for_player1 else result.actions_p2
    opp_actions = result.actions_p2 if for_player1 else result.actions_p1

    achieved = 0
    best = 0
    for mine, theirs in zip(my_actions, opp_actions):
        if not (is_valid_action(mine) and is_valid_action(theirs)):
            continue
        achieved += result.game.payoff_for(mine, theirs, am_row_player=for_player1)[0]
        if method == "max_cell":
            best += _max_cell_payoff(result, am_row_player=for_player1)
        else:
            best += _best_round_payoff(result, theirs, am_row_player=for_player1)
    return (achieved / best) if best > 0 else None


def score_ratio_table(results: list[MatchResult], *, method: str = "best_response") -> pd.DataFrame:
    """Aggregates score ratios per player and game family (Table 1).

    Args:
        results: All match results.
        method: Score-ratio benchmark method.

    Returns:
        A DataFrame with columns ``[player, family, score_ratio, n_matches]``.
    """
    rows: list[dict] = []
    for result in results:
        for for_p1, name in ((True, result.player1_name), (False, result.player2_name)):
            ratio = match_score_ratio(result, for_player1=for_p1, method=method)
            if ratio is not None:
                rows.append({"player": name, "family": result.game.family, "score_ratio": ratio})

    if not rows:
        return pd.DataFrame(columns=["player", "family", "score_ratio", "n_matches"])

    frame = pd.DataFrame(rows)
    table = (
        frame.groupby(["player", "family"])["score_ratio"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "score_ratio", "count": "n_matches"})
    )
    return table
