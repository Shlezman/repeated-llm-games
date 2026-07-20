"""Tests for markdown-template rendering and game-preserving robustness transforms."""

from __future__ import annotations

from llmgames.engine.families import canonical_bos, canonical_pd
from llmgames.engine.game import ACTION_A, ACTION_B
from llmgames.players.base import PlayerView
from llmgames.prompts import render
from llmgames.prompts.loader import load_template
from llmgames.prompts.transforms import build_framing, order_sequence


def test_order_sequence_no_randomize_is_canonical():
    """With randomization off, every round uses the canonical (A, B) order."""
    assert order_sequence(seed=1, num_rounds=10, randomize=False) == [(ACTION_A, ACTION_B)] * 10


def test_order_sequence_is_seed_deterministic():
    """Randomized order is reproducible for a fixed seed and contains both orders."""
    a = order_sequence(seed=42, num_rounds=20, randomize=True)
    b = order_sequence(seed=42, num_rounds=20, randomize=True)
    assert a == b
    assert {(ACTION_A, ACTION_B), (ACTION_B, ACTION_A)} <= set(a)


def test_cover_story_preserves_payoffs():
    """A cooking cover story changes framing words but not the payoff numbers."""
    framing = build_framing(cover_story="cooking", utility_label="dollars")
    text = render.render_rules(canonical_pd(), framing, am_row_player=True, num_rounds=10)
    assert "Recipe" in text and "dollars" in text and "contestant" in text
    for value in ("8 dollars", "0 dollars", "10 dollars", "5 dollars"):
        assert value in text


def test_rules_player_two_perspective_transposed():
    """Player 2's BoS rules reflect the transposed payoffs (prefers option F)."""
    text = render.render_rules(canonical_bos(), build_framing(), am_row_player=False, num_rounds=10)
    assert "you win 7 points and the other player wins 10 points" in text
    assert "you win 10 points and the other player wins 7 points" in text


def test_base_decision_template_includes_history_and_question():
    """The rendered base decision prompt appends history and asks for one letter."""
    view = PlayerView(
        game=canonical_pd(), am_row_player=True, round_index=2, num_rounds=10,
        my_actions=(ACTION_A,), opponent_actions=(ACTION_B,),
        my_points=(0,), opponent_points=(10,),
    )
    prompt = load_template("base_decision").format(**render.decision_vars(view, build_framing(), (ACTION_A, ACTION_B)))
    assert "In round 1, you chose Option J" in prompt
    assert "currently playing round 2" in prompt
    assert "exactly one letter" in prompt
