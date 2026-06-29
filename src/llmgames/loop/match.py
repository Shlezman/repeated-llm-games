"""The repeated-game execution loop for a single pairing.

Both players decide independently each round using only the history of prior
rounds (no within-round leakage); the system scores the round and appends the
outcome to both players' histories. SCoT predictions are captured for the
theory-of-mind metric.
"""

from __future__ import annotations

from ..engine.game import ACTIONS, Action, Game
from ..engine.payoff import score_round
from ..players.base import MatchResult, Player, PlayerView


def _view(
    game: Game,
    *,
    am_row_player: bool,
    round_index: int,
    num_rounds: int,
    my_actions: list[Action],
    opp_actions: list[Action],
    my_points: list[int],
    opp_points: list[int],
    option_order: tuple[Action, Action],
) -> PlayerView:
    """Builds an immutable per-round view for one player."""
    return PlayerView(
        game=game,
        am_row_player=am_row_player,
        round_index=round_index,
        num_rounds=num_rounds,
        my_actions=tuple(my_actions),
        opponent_actions=tuple(opp_actions),
        my_points=tuple(my_points),
        opponent_points=tuple(opp_points),
        option_order=option_order,
    )


def play_match(
    game: Game,
    player1: Player,
    player2: Player,
    *,
    num_rounds: int,
    orders: list[tuple[Action, Action]] | None = None,
) -> MatchResult:
    """Plays one repeated game between two players.

    Args:
        game: The game to play.
        player1: The row player.
        player2: The column player.
        num_rounds: Number of rounds to play.
        orders: Per-round option presentation order; defaults to canonical order.

    Returns:
        The completed :class:`MatchResult`.
    """
    actions_p1: list[Action] = []
    actions_p2: list[Action] = []
    points_p1: list[int] = []
    points_p2: list[int] = []
    preds_p1: list[Action | None] = []
    preds_p2: list[Action | None] = []
    thoughts_p1: list[dict] = []
    thoughts_p2: list[dict] = []

    for rnd in range(1, num_rounds + 1):
        order = orders[rnd - 1] if orders else ACTIONS

        view1 = _view(
            game, am_row_player=True, round_index=rnd, num_rounds=num_rounds,
            my_actions=actions_p1, opp_actions=actions_p2,
            my_points=points_p1, opp_points=points_p2, option_order=order,
        )
        view2 = _view(
            game, am_row_player=False, round_index=rnd, num_rounds=num_rounds,
            my_actions=actions_p2, opp_actions=actions_p1,
            my_points=points_p2, opp_points=points_p1, option_order=order,
        )

        action1 = player1.choose(view1)
        action2 = player2.choose(view2)
        preds_p1.append(player1.predict_opponent(view1))
        preds_p2.append(player2.predict_opponent(view2))
        thoughts_p1.append(_thoughts(player1))
        thoughts_p2.append(_thoughts(player2))

        round_p1, round_p2 = score_round(game, action1, action2)

        actions_p1.append(action1)
        actions_p2.append(action2)
        points_p1.append(round_p1)
        points_p2.append(round_p2)

    return MatchResult(
        game=game,
        player1_name=player1.name,
        player2_name=player2.name,
        actions_p1=tuple(actions_p1),
        actions_p2=tuple(actions_p2),
        points_p1=tuple(points_p1),
        points_p2=tuple(points_p2),
        predictions_p1=tuple(preds_p1),
        predictions_p2=tuple(preds_p2),
        thoughts_p1=tuple(thoughts_p1),
        thoughts_p2=tuple(thoughts_p2),
    )


def _thoughts(player: Player) -> dict:
    """Returns a player's last-round reasoning capture ({} if it has none)."""
    getter = getattr(player, "last_thoughts", None)
    return getter() if callable(getter) else {}
