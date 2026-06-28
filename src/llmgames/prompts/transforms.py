"""Framing and robustness transforms: surface presentation, never the game itself.

A :class:`Framing` carries the vocabulary (display labels, the word for an option,
the unit word, the opponent noun, cover-story intro) that fills the markdown prompt
templates. Every transform here is *game-preserving* — payoffs always come from the
:class:`~llmgames.engine.game.Game`; framing only changes wording, labels, the unit,
the cover story, and the order options are offered.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..engine.game import ACTIONS, Action

DEFAULT_LABELS: tuple[str, str] = ("J", "F")


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
        intro: Opening sentence template; ``{opponent}`` is substituted at render time.
    """

    action_labels: dict[Action, str]
    option_word: str = "Option"
    round_word: str = "round"
    unit_word: str = "points"
    opponent_word: str = "the other player"
    choose_verb: str = "choose"
    # Matches the original opening sentence ("...with another player."); cover stories
    # override with their own intro using the {opponent} placeholder.
    intro: str = "You are playing a game repeatedly with another player."

    def label(self, action: Action) -> str:
        """Returns the display label for a known internal action."""
        return self.action_labels[action]

    def display(self, action: Action) -> str:
        """Returns the display label, or the raw action for unknown/unparseable ones.

        Used when rendering history, which may contain the unparseable sentinel from a
        prior round; this must never raise.
        """
        return self.action_labels.get(action, action)

    def intro_text(self) -> str:
        """Returns the intro sentence with the opponent noun substituted in."""
        return self.intro.format(opponent=self.opponent_word)


def build_framing(
    *,
    cover_story: str = "none",
    utility_label: str = "points",
    labels: tuple[str, str] = DEFAULT_LABELS,
) -> Framing:
    """Builds a :class:`Framing` from robustness settings.

    Args:
        cover_story: One of "none", "cooking", "project".
        utility_label: Unit word for payoffs ("points", "dollars", "coins").
        labels: Display labels for (ACTION_A, ACTION_B).

    Returns:
        The constructed framing.

    Raises:
        ValueError: If ``cover_story`` is unknown.
    """
    action_labels = {ACTIONS[0]: labels[0], ACTIONS[1]: labels[1]}

    if cover_story == "none":
        return Framing(action_labels=action_labels, unit_word=utility_label)
    if cover_story == "cooking":
        return Framing(
            action_labels=action_labels,
            option_word="Recipe",
            round_word="dish",
            unit_word=utility_label,
            opponent_word="the other contestant",
            intro="You are taking part in a cooking competition with {opponent}.",
        )
    if cover_story == "project":
        return Framing(
            action_labels=action_labels,
            option_word="Approach",
            round_word="phase",
            unit_word=utility_label,
            opponent_word="your collaborator",
            intro="You are working on a collaborative project with {opponent}.",
        )
    raise ValueError(f"Unknown cover story: {cover_story!r}")


def order_sequence(*, seed: int, num_rounds: int, randomize: bool) -> list[tuple[Action, Action]]:
    """Produces a deterministic per-round option presentation order.

    Args:
        seed: Seed controlling the shuffle (reproducible across re-runs).
        num_rounds: Number of rounds to generate orders for.
        randomize: If False, every round uses the canonical (A, B) order.

    Returns:
        A list of ``(first_action, second_action)`` tuples, one per round.
    """
    canonical = (ACTIONS[0], ACTIONS[1])
    if not randomize:
        return [canonical for _ in range(num_rounds)]

    rng = random.Random(seed)
    orders: list[tuple[Action, Action]] = []
    for _ in range(num_rounds):
        pair = list(ACTIONS)
        rng.shuffle(pair)
        orders.append((pair[0], pair[1]))
    return orders
