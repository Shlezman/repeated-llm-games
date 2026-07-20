"""Regression tests for issues found in the adversarial review (LangChain build)."""

from __future__ import annotations

import math

from llmgames.engine.families import canonical_pd
from llmgames.engine.game import ACTION_A, ACTION_B
from llmgames.loop.tournament import _resolve_labels
from llmgames.players.base import UNPARSEABLE, MatchResult, PlayerView
from llmgames.players.llm_player import LLMPlayer
from llmgames.prompts import render
from llmgames.prompts.loader import load_template
from llmgames.prompts.transforms import DEFAULT_LABELS, build_framing
from llmgames.providers.models import build_chat_model

PD = canonical_pd()


def test_invalid_rounds_excluded_from_totals():
    """The -9999 sentinel from an unparseable round never leaks into match totals."""
    result = MatchResult(
        game=PD,
        player1_name="m",
        player2_name="opp",
        actions_p1=(ACTION_A, UNPARSEABLE, ACTION_A),
        actions_p2=(ACTION_B, ACTION_B, ACTION_A),
        points_p1=(0, -9999, 8),
        points_p2=(10, -9999, 8),
    )
    assert result.total_p1 == 8
    assert result.total_p2 == 18
    assert result.running_totals(for_player1=True) == [0, 0, 8]


def test_scot_falls_back_when_prediction_unparseable():
    """SCoT step-1 unparseable -> step-2 uses the unconditioned decision (no fabrication)."""
    # Mock cycles: prediction -> "no" (unparseable), decision -> "J" (cooperate).
    model = build_chat_model("mock", "x", mock_responses=["no", "J"])
    player = LLMPlayer("m", model, build_framing(), scot=True)
    view = PlayerView(game=PD, am_row_player=True, round_index=1, num_rounds=10)
    assert player.choose(view) == ACTION_A
    assert player.predict_opponent(view) is None  # no fabricated belief stored


def test_scot_records_conditioning_prediction():
    """The recorded SCoT prediction equals the one that conditioned the decision."""
    # Mock cycles: predict -> "F" (=B), decide -> "J" (=A).
    model = build_chat_model("mock", "x", mock_responses=["F", "J"])
    player = LLMPlayer("m", model, build_framing(), scot=True)
    view = PlayerView(game=PD, am_row_player=True, round_index=1, num_rounds=10)
    action = player.choose(view)
    assert player.predict_opponent(view) == ACTION_B  # parsed prediction "F"
    assert action == ACTION_A  # conditioned decision "J"


def test_resolve_labels():
    """Label resolution: default J/F, seeded-random reproducible, explicit passthrough."""
    assert _resolve_labels(None, 42) == DEFAULT_LABELS
    rand1 = _resolve_labels("random", 7)
    rand2 = _resolve_labels("random", 7)
    assert rand1 == rand2 and len(set(rand1)) == 2 and all(c.isupper() for c in rand1)
    assert _resolve_labels(["Q", "X"], 0) == ("Q", "X")


def test_history_with_unparseable_action_does_not_crash():
    """Rendering history that contains a prior unparseable action must not raise."""
    view = PlayerView(
        game=PD, am_row_player=True, round_index=2, num_rounds=10,
        my_actions=(UNPARSEABLE,), opponent_actions=(ACTION_B,),
        my_points=(-9999,), opponent_points=(-9999,),
    )
    prompt = load_template("base_decision").format(**render.decision_vars(view, build_framing(), (ACTION_A, ACTION_B)))
    assert "Option ?" in prompt


def test_human_success_metric_is_game_appropriate():
    """BoS gets a score-derived success rate; PD gets mutual-cooperation, not coordination."""
    from llmgames.metrics import human

    summary = human.summarize_human(human.load_human_data())
    bos = summary[summary["game"] == "BoS"]
    pd_rows = summary[summary["game"] == "PD"]
    assert bos["success_rate"].notna().all()
    assert math.isnan(pd_rows["success_rate"].iloc[0])
    assert pd_rows["mutual_cooperation_rate"].notna().all()
