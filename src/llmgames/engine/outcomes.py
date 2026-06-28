"""Semantic classification of round outcomes used by the figure metrics.

These helpers are label-agnostic: the "cooperate"/"preferred" action defaults to
:data:`~llmgames.engine.game.ACTION_A` (matching the canonical PD/BoS encoding
where A=cooperate / A=player-1-preferred) but is configurable per game.
"""

from __future__ import annotations

from .game import ACTION_A, Action


def is_cooperation(action: Action, *, cooperate_action: Action = ACTION_A) -> bool:
    """Returns True if ``action`` is the cooperative action (Prisoner's Dilemma)."""
    return action == cooperate_action


def is_mutual_cooperation(a1: Action, a2: Action, *, cooperate_action: Action = ACTION_A) -> bool:
    """Returns True if both players chose the cooperative action."""
    return a1 == cooperate_action and a2 == cooperate_action


def is_coordinated(a1: Action, a2: Action) -> bool:
    """Returns True if both players chose the same action (Battle of the Sexes success)."""
    return a1 == a2


def is_preferred_equilibrium(a1: Action, a2: Action, *, preferred_action: Action = ACTION_A) -> bool:
    """Returns True if both players landed on player 1's preferred coordination point."""
    return a1 == preferred_action and a2 == preferred_action
