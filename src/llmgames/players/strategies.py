"""Hand-coded baseline strategies, ported from the released study code.

Internal encoding follows the canonical games: :data:`ACTION_A` = cooperate /
player-1-preferred ("J"), :data:`ACTION_B` = defect / other ("F"). Strategies
needing randomness receive a seeded RNG so runs stay reproducible.

Note: the original ``htft`` source returns the defect action on move 1 despite a
comment describing first-move cooperation; we reproduce the released behaviour.
"""

from __future__ import annotations

import random
from typing import Callable

from ..engine.game import ACTION_A, ACTION_B, Action
from .base import PlayerView

StrategyFn = Callable[[PlayerView, random.Random], Action]


def always_cooperate(view: PlayerView, rng: random.Random) -> Action:
    """Always plays the cooperative action."""
    return ACTION_A


def always_defect(view: PlayerView, rng: random.Random) -> Action:
    """Always plays the defect action."""
    return ACTION_B


def defect_once(view: PlayerView, rng: random.Random) -> Action:
    """Defects in round 1, then cooperates for the rest of the match."""
    return ACTION_B if view.round_index == 1 else ACTION_A


def alternate_ab(view: PlayerView, rng: random.Random) -> Action:
    """Alternates starting from cooperate: A, B, A, B, ..."""
    return ACTION_A if view.round_index % 2 == 1 else ACTION_B


def alternate_ba(view: PlayerView, rng: random.Random) -> Action:
    """Alternates starting from defect: B, A, B, A, ..."""
    return ACTION_B if view.round_index % 2 == 1 else ACTION_A


def tit_for_tat(view: PlayerView, rng: random.Random) -> Action:
    """Cooperates first, then copies the opponent's previous action."""
    if view.round_index == 1:
        return ACTION_A
    return view.opponent_last(1) or ACTION_A


def tit_for_two_tats(view: PlayerView, rng: random.Random) -> Action:
    """Cooperates first; defects only after two consecutive opponent defects."""
    if view.round_index == 1:
        return ACTION_A
    if view.opponent_last(1) == ACTION_B and view.opponent_last(2) == ACTION_B:
        return ACTION_B
    return ACTION_A


def suspicious_tft(view: PlayerView, rng: random.Random) -> Action:
    """Defects first, then copies the opponent's previous action."""
    if view.round_index == 1:
        return ACTION_B
    return view.opponent_last(1) or ACTION_A


def reverse_tft(view: PlayerView, rng: random.Random) -> Action:
    """Defects first, then plays the opposite of the opponent's previous action."""
    if view.round_index == 1:
        return ACTION_B
    return ACTION_B if view.opponent_last(1) == ACTION_A else ACTION_A


def hard_tft(view: PlayerView, rng: random.Random) -> Action:
    """Defects if the opponent defected in any of the last three rounds.

    Reproduces the released code's first-move defect behaviour.
    """
    if view.round_index == 1:
        return ACTION_B
    recent = [view.opponent_last(1), view.opponent_last(2), view.opponent_last(3)]
    return ACTION_B if ACTION_B in recent else ACTION_A


def _naive_prober(view: PlayerView, rng: random.Random, *, prob: float) -> Action:
    """Tit-for-tat that defects with probability ``prob`` instead of copying."""
    if view.round_index == 1:
        return ACTION_A
    if rng.random() < prob:
        return ACTION_B
    return view.opponent_last(1) or ACTION_A


def naive_prober_10(view: PlayerView, rng: random.Random) -> Action:
    """Tit-for-tat with a 10% chance of defecting each round."""
    return _naive_prober(view, rng, prob=0.1)


def naive_prober_20(view: PlayerView, rng: random.Random) -> Action:
    """Tit-for-tat with a 20% chance of defecting each round."""
    return _naive_prober(view, rng, prob=0.2)


STRATEGIES: dict[str, StrategyFn] = {
    "always_cooperate": always_cooperate,
    "always_defect": always_defect,
    "defect_once": defect_once,
    "alternate": alternate_ab,
    "alternate_ab": alternate_ab,
    "alternate_ba": alternate_ba,
    "tit_for_tat": tit_for_tat,
    "tit_for_two_tats": tit_for_two_tats,
    "suspicious_tft": suspicious_tft,
    "reverse_tft": reverse_tft,
    "hard_tft": hard_tft,
    "naive_prober_10": naive_prober_10,
    "naive_prober_20": naive_prober_20,
}


class StrategyPlayer:
    """A :class:`~llmgames.players.base.Player` backed by a hand-coded strategy.

    Attributes:
        name: The strategy name.
    """

    def __init__(self, name: str, fn: StrategyFn, seed: int = 0) -> None:
        """Creates a strategy player.

        Args:
            name: Identifier used in result records.
            fn: The strategy function.
            seed: Seed for any randomized strategy, for reproducibility.
        """
        self.name = name
        self._fn = fn
        self._rng = random.Random(seed)

    def choose(self, view: PlayerView) -> Action:
        """Returns the strategy's action for the current round."""
        return self._fn(view, self._rng)

    def predict_opponent(self, view: PlayerView) -> Action | None:
        """Hand-coded strategies do not predict; always returns None."""
        return None

    def last_thoughts(self) -> dict:
        """Hand-coded strategies have no reasoning to report."""
        return {}


def make_strategy(name: str, seed: int = 0) -> StrategyPlayer:
    """Builds a :class:`StrategyPlayer` by registered name.

    Args:
        name: A key in :data:`STRATEGIES`.
        seed: Seed for randomized strategies.

    Returns:
        The constructed strategy player.

    Raises:
        KeyError: If ``name`` is not a registered strategy.
    """
    if name not in STRATEGIES:
        raise KeyError(f"Unknown strategy {name!r}. Available: {sorted(STRATEGIES)}")
    return StrategyPlayer(name=name, fn=STRATEGIES[name], seed=seed)
