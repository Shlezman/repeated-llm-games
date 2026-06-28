"""2x2 game definition: payoff matrices over two abstract actions.

Games are stored with neutral internal action identifiers (``"A"`` / ``"B"``) and
are fully decoupled from presentation (display labels such as ``J``/``F`` or
``Recipe X``/``Recipe Y`` live in the prompt layer). Payoffs are stored from the
row player's perspective; :meth:`Game.payoff_for` resolves either player's view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

Action = str

ACTION_A: Action = "A"
ACTION_B: Action = "B"
ACTIONS: tuple[Action, Action] = (ACTION_A, ACTION_B)


@dataclass(frozen=True)
class Game:
    """An immutable 2x2 normal-form game.

    Attributes:
        name: Human-readable game name (e.g. "Prisoner's Dilemma").
        family: Game family label (e.g. "PD Family", "Win-win").
        payoffs: Maps an ``(action_row, action_col)`` pair to the
            ``(row_player_points, col_player_points)`` outcome. Must contain all
            four combinations of :data:`ACTIONS`.
    """

    name: str
    family: str
    payoffs: Mapping[tuple[Action, Action], tuple[int, int]]

    def __post_init__(self) -> None:
        """Validates that the payoff matrix is a complete 2x2 grid.

        Raises:
            ValueError: If any of the four action combinations is missing.
        """
        expected = {(a, b) for a in ACTIONS for b in ACTIONS}
        missing = expected - set(self.payoffs)
        if missing:
            raise ValueError(f"{self.name}: payoff matrix missing entries {sorted(missing)}")

    def payoff(self, row_action: Action, col_action: Action) -> tuple[int, int]:
        """Returns ``(row_points, col_points)`` for a canonical action pair.

        Args:
            row_action: Action chosen by the row player (player 1).
            col_action: Action chosen by the column player (player 2).

        Returns:
            The ``(row_player_points, col_player_points)`` tuple.
        """
        return self.payoffs[(row_action, col_action)]

    def payoff_for(
        self, my_action: Action, opponent_action: Action, *, am_row_player: bool
    ) -> tuple[int, int]:
        """Returns ``(my_points, opponent_points)`` from one player's perspective.

        Args:
            my_action: The action chosen by this player.
            opponent_action: The action chosen by the opponent.
            am_row_player: True if this player occupies the row position (player 1).

        Returns:
            A ``(my_points, opponent_points)`` tuple oriented to this player.
        """
        if am_row_player:
            return self.payoff(my_action, opponent_action)
        # This player is the column player: the opponent occupies the row.
        row_pts, col_pts = self.payoff(opponent_action, my_action)
        return col_pts, row_pts

    def outcomes_for(self, *, am_row_player: bool) -> dict[tuple[Action, Action], tuple[int, int]]:
        """Returns all four outcomes oriented to one player's perspective.

        Args:
            am_row_player: True for the row player's view, False for the column player.

        Returns:
            A mapping of ``(my_action, opponent_action)`` to ``(my_points, opp_points)``.
        """
        return {
            (mine, theirs): self.payoff_for(mine, theirs, am_row_player=am_row_player)
            for mine in ACTIONS
            for theirs in ACTIONS
        }
