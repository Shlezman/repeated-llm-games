"""Barebones prompt construction faithful to Akata et al., decoupled from framing.

The :class:`Framing` carries all surface choices (display labels, the word for an
"option", the unit word, the opponent noun) so that robustness transforms — label
swaps and cover stories — reuse the exact same mechanics and cannot accidentally
change the underlying game. The builder renders the static rules block, the
appended history, and the per-round decision question.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..engine.game import ACTIONS, Action, Game
from ..players.base import PlayerView


@dataclass(frozen=True)
class Framing:
    """Surface presentation of a game; never alters payoffs or structure.

    Attributes:
        action_labels: Maps each internal action to its displayed label (e.g. A->"J").
        option_word: Noun for a choice ("Option", "Recipe", "Approach").
        round_word: Noun for a round ("round", "dish", "phase").
        unit_word: Payoff unit ("points", "dollars", "coins").
        opponent_word: How the opponent is referred to ("the other player").
        choose_verb: Verb for selecting ("choose").
        intro: Opening sentence template; ``{opponent}`` is substituted.
    """

    action_labels: dict[Action, str]
    option_word: str = "Option"
    round_word: str = "round"
    unit_word: str = "points"
    opponent_word: str = "the other player"
    choose_verb: str = "choose"
    intro: str = "You are playing a game repeatedly with {opponent}."

    def label(self, action: Action) -> str:
        """Returns the display label for a known internal action."""
        return self.action_labels[action]

    def display(self, action: Action) -> str:
        """Returns the display label, or the raw action for unknown/unparseable ones.

        Used when rendering history, which may contain the unparseable sentinel from
        a prior round; this must never raise.
        """
        return self.action_labels.get(action, action)


def rules_text(game: Game, framing: Framing, *, am_row_player: bool, num_rounds: int) -> str:
    """Renders the static rules block stating all four outcomes explicitly.

    Args:
        game: The game being played.
        framing: Surface presentation.
        am_row_player: Whether the addressed player is the row player.
        num_rounds: Total rounds in the match.

    Returns:
        The multi-line rules string (no trailing round question).
    """
    opt = framing.option_word
    unit = framing.unit_word
    opp = framing.opponent_word
    la, lb = framing.label(ACTIONS[0]), framing.label(ACTIONS[1])

    lines = [
        f"{framing.intro.format(opponent=opp)} "
        f"In this game, you can {framing.choose_verb} between {opt} {la} and {opt} {lb}. "
        f"You will play {num_rounds} {framing.round_word}s in total with the same player. "
        "The rules of the game are as follows:",
    ]
    for mine in ACTIONS:
        for theirs in ACTIONS:
            my_pts, opp_pts = game.payoff_for(mine, theirs, am_row_player=am_row_player)
            lines.append(
                f"If you {framing.choose_verb} {opt} {framing.label(mine)} and {opp} "
                f"{framing.choose_verb}s {opt} {framing.label(theirs)}, then you win "
                f"{my_pts} {unit} and {opp} wins {opp_pts} {unit}."
            )
    return "\n".join(lines) + "\n\n"


def history_text(view: PlayerView, framing: Framing) -> str:
    """Renders the appended per-round history from this player's perspective.

    Args:
        view: The current player view (histories oldest-to-newest).
        framing: Surface presentation.

    Returns:
        The history string (empty if no rounds have been played).
    """
    opt = framing.option_word
    unit = framing.unit_word
    opp = framing.opponent_word
    rounds = []
    for idx, (mine, theirs, my_pts, opp_pts) in enumerate(
        zip(view.my_actions, view.opponent_actions, view.my_points, view.opponent_points), start=1
    ):
        rounds.append(
            f"In {framing.round_word} {idx}, you chose {opt} {framing.display(mine)} and {opp} "
            f"chose {opt} {framing.display(theirs)}. Thus, you won {my_pts} {unit} and {opp} "
            f"won {opp_pts} {unit}."
        )
    return ("\n".join(rounds) + "\n") if rounds else ""


def _answer_cue(framing: Framing, order: tuple[Action, Action]) -> str:
    """Builds the explicit single-label answer instruction for chat models.

    Args:
        framing: Surface presentation.
        order: The order in which the two options are offered this round.

    Returns:
        An instruction telling the model to answer with exactly one label.
    """
    l0, l1 = framing.label(order[0]), framing.label(order[1])
    opt = framing.option_word
    return (
        f"Which {opt} do you {framing.choose_verb}, {opt} {l0} or {opt} {l1}? "
        f"Respond with exactly one letter ({l0} or {l1}) and nothing else."
    )


def decision_prompt(
    view: PlayerView, framing: Framing, order: tuple[Action, Action]
) -> str:
    """Builds the full base-mode decision prompt for the current round.

    Args:
        view: The current player view.
        framing: Surface presentation.
        order: Order in which the two options are offered (for order randomization).

    Returns:
        The complete prompt string.
    """
    rules = rules_text(
        view.game, framing, am_row_player=view.am_row_player, num_rounds=view.num_rounds
    )
    history = history_text(view, framing)
    head = f"You are currently playing {framing.round_word} {view.round_index}.\n"
    return rules + history + head + _answer_cue(framing, order)


def prediction_prompt(
    view: PlayerView, framing: Framing, order: tuple[Action, Action]
) -> str:
    """Builds the SCoT step-1 prompt asking the model to predict the opponent's move.

    Args:
        view: The current player view.
        framing: Surface presentation.
        order: Order in which the two options are offered.

    Returns:
        The opponent-prediction prompt string.
    """
    rules = rules_text(
        view.game, framing, am_row_player=view.am_row_player, num_rounds=view.num_rounds
    )
    history = history_text(view, framing)
    l0, l1 = framing.label(order[0]), framing.label(order[1])
    opt = framing.option_word
    head = f"You are currently playing {framing.round_word} {view.round_index}.\n"
    ask = (
        f"Which {opt} do you predict {framing.opponent_word} will {framing.choose_verb}, "
        f"{opt} {l0} or {opt} {l1}? Respond with exactly one letter ({l0} or {l1})."
    )
    return rules + history + head + ask


def conditioned_decision_prompt(
    view: PlayerView,
    framing: Framing,
    order: tuple[Action, Action],
    predicted_label: str,
) -> str:
    """Builds the SCoT step-2 prompt: choose, given the predicted opponent move.

    Args:
        view: The current player view.
        framing: Surface presentation.
        order: Order in which the two options are offered.
        predicted_label: The display label predicted for the opponent in step 1.

    Returns:
        The conditioned decision prompt string.
    """
    rules = rules_text(
        view.game, framing, am_row_player=view.am_row_player, num_rounds=view.num_rounds
    )
    history = history_text(view, framing)
    l0, l1 = framing.label(order[0]), framing.label(order[1])
    opt = framing.option_word
    head = f"You are currently playing {framing.round_word} {view.round_index}.\n"
    ask = (
        f"Given that you think {framing.opponent_word} will {framing.choose_verb} {opt} "
        f"{predicted_label} in this {framing.round_word}, which {opt} do you think is best "
        f"for you to {framing.choose_verb}, {opt} {l0} or {opt} {l1}? "
        f"Respond with exactly one letter ({l0} or {l1}) and nothing else."
    )
    return rules + history + head + ask
