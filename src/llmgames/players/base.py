"""Player abstraction and the immutable per-turn view handed to players.

A :class:`Player` exposes :meth:`Player.choose` and, for theory-of-mind / SCoT
agents, :meth:`Player.predict_opponent`. Players reason purely over abstract
actions; all presentation (display labels, prompt text) is handled elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..engine.game import ACTION_A, ACTION_B, Action, Game
from ..engine.payoff import is_valid_action

# Sentinel action returned when a model reply cannot be parsed into a legal action.
# It is not a valid game action, so scoring marks the round invalid.
UNPARSEABLE: Action = "?"


@dataclass(frozen=True)
class PlayerView:
    """Everything a player may use to decide its next action.

    Histories are ordered oldest-to-newest and exclude the current round.

    Attributes:
        game: The game being played.
        am_row_player: True if this player is player 1 (row), False if player 2.
        round_index: The 1-based index of the round about to be played.
        num_rounds: Total number of rounds in the match.
        my_actions: This player's past actions.
        opponent_actions: The opponent's past actions.
        my_points: This player's per-round points so far.
        opponent_points: The opponent's per-round points so far.
        option_order: The order the two options are offered this round (for the
            order-randomization robustness transform).
    """

    game: Game
    am_row_player: bool
    round_index: int
    num_rounds: int
    my_actions: tuple[Action, ...] = ()
    opponent_actions: tuple[Action, ...] = ()
    my_points: tuple[int, ...] = ()
    opponent_points: tuple[int, ...] = ()
    option_order: tuple[Action, Action] = (ACTION_A, ACTION_B)

    def opponent_last(self, back: int = 1) -> Action | None:
        """Returns the opponent's action ``back`` rounds ago, or None if unavailable.

        Args:
            back: How many rounds to look back (1 = most recent).

        Returns:
            The opponent's action, or None if the history is too short.
        """
        if back <= 0 or back > len(self.opponent_actions):
            return None
        return self.opponent_actions[-back]


@runtime_checkable
class Player(Protocol):
    """An agent that selects an action each round.

    Attributes:
        name: Stable identifier used in result records.
    """

    name: str

    def choose(self, view: PlayerView) -> Action:
        """Returns this player's action for the round described by ``view``."""
        ...

    def predict_opponent(self, view: PlayerView) -> Action | None:
        """Returns a predicted opponent action (SCoT), or None if unsupported."""
        ...


@dataclass(frozen=True)
class MatchResult:
    """The full record of one 10-round match between two players.

    Attributes:
        game: The game played.
        player1_name: Identifier of the row player.
        player2_name: Identifier of the column player.
        actions_p1: Player 1's action each round.
        actions_p2: Player 2's action each round.
        points_p1: Player 1's points each round.
        points_p2: Player 2's points each round.
        predictions_p1: Player 1's opponent predictions (None when not SCoT).
        predictions_p2: Player 2's opponent predictions (None when not SCoT).
        thoughts_p1: Per-round reasoning capture for player 1 — dicts with keys
            ``predicted`` (display label or None), ``predict_text``, ``decide_text``.
            Empty for non-LLM players.
        thoughts_p2: Per-round reasoning capture for player 2.
    """

    game: Game
    player1_name: str
    player2_name: str
    actions_p1: tuple[Action, ...]
    actions_p2: tuple[Action, ...]
    points_p1: tuple[int, ...]
    points_p2: tuple[int, ...]
    predictions_p1: tuple[Action | None, ...] = field(default_factory=tuple)
    predictions_p2: tuple[Action | None, ...] = field(default_factory=tuple)
    thoughts_p1: tuple[dict, ...] = field(default_factory=tuple)
    thoughts_p2: tuple[dict, ...] = field(default_factory=tuple)

    def _valid_total(self, points: tuple[int, ...]) -> int:
        """Sums points over rounds where both players' actions are valid.

        Args:
            points: Per-round points for one player.

        Returns:
            The total, excluding rounds with an unparseable/invalid action on
            either side (so the ``INVALID_SCORE`` sentinel never leaks into totals).
        """
        return sum(
            pts
            for pts, a1, a2 in zip(points, self.actions_p1, self.actions_p2)
            if is_valid_action(a1) and is_valid_action(a2)
        )

    @property
    def total_p1(self) -> int:
        """Player 1's total score across valid rounds."""
        return self._valid_total(self.points_p1)

    @property
    def total_p2(self) -> int:
        """Player 2's total score across valid rounds."""
        return self._valid_total(self.points_p2)

    def running_totals(self, *, for_player1: bool) -> list[int]:
        """Returns cumulative valid-round totals after each round (for CSV/plots).

        Args:
            for_player1: True for the row player's totals, False for the column player.

        Returns:
            A list of cumulative totals, one per round, excluding invalid rounds.
        """
        points = self.points_p1 if for_player1 else self.points_p2
        running = 0
        out: list[int] = []
        for pts, a1, a2 in zip(points, self.actions_p1, self.actions_p2):
            if is_valid_action(a1) and is_valid_action(a2):
                running += pts
            out.append(running)
        return out
