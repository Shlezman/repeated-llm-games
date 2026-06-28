"""Pure scoring functions over a :class:`~llmgames.engine.game.Game`.

A sentinel score is returned for invalid/unparseable actions, mirroring the
original study's ``-9999`` marker so downstream code can detect and exclude them.
"""

from __future__ import annotations

from .game import ACTIONS, Action, Game

INVALID_SCORE: int = -9999


def is_valid_action(action: Action) -> bool:
    """Returns True if ``action`` is one of the game's two legal actions."""
    return action in ACTIONS


def score_round(game: Game, action_p1: Action, action_p2: Action) -> tuple[int, int]:
    """Scores a single round for both players.

    Args:
        game: The game being played.
        action_p1: Player 1's (row) action.
        action_p2: Player 2's (column) action.

    Returns:
        A ``(player1_points, player2_points)`` tuple, or
        ``(INVALID_SCORE, INVALID_SCORE)`` if either action is illegal.
    """
    if not (is_valid_action(action_p1) and is_valid_action(action_p2)):
        return INVALID_SCORE, INVALID_SCORE
    return game.payoff(action_p1, action_p2)
