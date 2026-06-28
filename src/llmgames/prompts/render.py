"""Render the dynamic parts of prompts (rules block, history) and build the
variable dicts consumed by the markdown question templates.

Payoff numbers always come from the :class:`~llmgames.engine.game.Game`, so framing
changes (labels, cover story, unit) can never alter the underlying game.
"""

from __future__ import annotations

from ..engine.game import ACTIONS, Action
from ..players.base import PlayerView
from .loader import load_template
from .transforms import Framing

CANONICAL_ORDER: tuple[Action, Action] = (ACTIONS[0], ACTIONS[1])


def render_rules(game, framing: Framing, *, am_row_player: bool, num_rounds: int) -> str:
    """Renders the static rules block stating all four outcomes for one perspective."""
    intro_t = load_template("rules_intro")
    outcome_t = load_template("rules_outcome")
    lines = [
        intro_t.format(
            intro=framing.intro_text(),
            verb=framing.choose_verb,
            option_word=framing.option_word,
            label_a=framing.label(ACTIONS[0]),
            label_b=framing.label(ACTIONS[1]),
            num_rounds=num_rounds,
            round_word=framing.round_word,
        )
    ]
    for mine in ACTIONS:
        for theirs in ACTIONS:
            my_pts, opp_pts = game.payoff_for(mine, theirs, am_row_player=am_row_player)
            lines.append(
                outcome_t.format(
                    verb=framing.choose_verb,
                    option_word=framing.option_word,
                    my_label=framing.label(mine),
                    opponent=framing.opponent_word,
                    opp_label=framing.label(theirs),
                    my_pts=my_pts,
                    opp_pts=opp_pts,
                    unit=framing.unit_word,
                )
            )
    return "\n".join(lines)


def render_history(view: PlayerView, framing: Framing) -> str:
    """Renders the appended per-round history from this player's perspective."""
    line_t = load_template("history_line")
    rows = []
    for idx, (mine, theirs, my_pts, opp_pts) in enumerate(
        zip(view.my_actions, view.opponent_actions, view.my_points, view.opponent_points), start=1
    ):
        rows.append(
            line_t.format(
                round_word=framing.round_word,
                idx=idx,
                option_word=framing.option_word,
                my_label=framing.display(mine),
                opponent=framing.opponent_word,
                opp_label=framing.display(theirs),
                my_pts=my_pts,
                opp_pts=opp_pts,
                unit=framing.unit_word,
            )
        )
    return ("\n".join(rows) + "\n") if rows else ""


def _common_vars(view: PlayerView, framing: Framing, order: tuple[Action, Action]) -> dict:
    """Builds the variables shared by every question template."""
    return {
        "rules": render_rules(
            view.game, framing, am_row_player=view.am_row_player, num_rounds=view.num_rounds
        ),
        "history": render_history(view, framing),
        "round_word": framing.round_word,
        "round_index": view.round_index,
        "option_word": framing.option_word,
        "verb": framing.choose_verb,
        "l0": framing.label(order[0]),
        "l1": framing.label(order[1]),
    }


def decision_vars(view: PlayerView, framing: Framing, order: tuple[Action, Action]) -> dict:
    """Variables for ``base_decision.md``."""
    return _common_vars(view, framing, order)


def predict_vars(view: PlayerView, framing: Framing) -> dict:
    """Variables for ``scot_predict.md`` (canonical option order)."""
    vars_ = _common_vars(view, framing, CANONICAL_ORDER)
    vars_["opponent"] = framing.opponent_word
    return vars_


def scot_decide_vars(view: PlayerView, framing: Framing, predicted_label: str) -> dict:
    """Variables for ``scot_decide.md`` (canonical order, conditioned on a prediction)."""
    vars_ = _common_vars(view, framing, CANONICAL_ORDER)
    vars_["opponent"] = framing.opponent_word
    vars_["predicted"] = predicted_label
    return vars_
