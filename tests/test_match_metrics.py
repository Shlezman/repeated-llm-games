"""Tests for the match loop and the figure/Table-1 metrics."""

from __future__ import annotations

import math

from llmgames.engine.families import canonical_bos, canonical_pd
from llmgames.engine.game import ACTION_A, ACTION_B
from llmgames.loop.match import play_match
from llmgames.metrics import bos_metrics, pd_metrics, prediction, score_ratio
from llmgames.players.base import MatchResult
from llmgames.players.strategies import make_strategy


def test_play_match_tft_vs_always_defect_scores():
    """TFT vs Always-Defect in PD: TFT scores 20, defector scores 30 over 5 rounds."""
    pd = canonical_pd()
    result = play_match(
        pd, make_strategy("tit_for_tat"), make_strategy("always_defect"), num_rounds=5
    )
    assert result.actions_p1 == (ACTION_A, ACTION_B, ACTION_B, ACTION_B, ACTION_B)
    assert result.actions_p2 == (ACTION_B,) * 5
    assert result.total_p1 == 20
    assert result.total_p2 == 30


def test_score_ratio_best_response():
    """Score ratio for TFT vs Always-Defect is 20/25 = 0.8."""
    pd = canonical_pd()
    result = play_match(
        pd, make_strategy("tit_for_tat"), make_strategy("always_defect"), num_rounds=5
    )
    ratio = score_ratio.match_score_ratio(result, for_player1=True)
    assert math.isclose(ratio, 0.8, rel_tol=1e-9)


def test_pd_defection_matrix():
    """TFT defects in 4 of 5 rounds against a constant defector."""
    pd = canonical_pd()
    result = play_match(
        pd, make_strategy("tit_for_tat"), make_strategy("always_defect"), num_rounds=5
    )
    matrix = pd_metrics.defection_rate_matrix([result])
    assert math.isclose(matrix.loc["tit_for_tat", "always_defect"], 0.8, rel_tol=1e-9)


def test_bos_collaboration_and_preferred():
    """Two alternating players in BoS coordinate every round, on A in half of them."""
    bos = canonical_bos()
    result = play_match(bos, make_strategy("alternate"), make_strategy("alternate"), num_rounds=4)
    collab = bos_metrics.collaboration_rate_matrix([result])
    pref = bos_metrics.preferred_frequency_matrix([result])
    assert math.isclose(collab.loc["alternate", "alternate"], 1.0, rel_tol=1e-9)
    assert math.isclose(pref.loc["alternate", "alternate"], 0.5, rel_tol=1e-9)


def test_prediction_accuracy():
    """Prediction accuracy compares predictions against the opponent's actual moves."""
    pd = canonical_pd()
    result = MatchResult(
        game=pd,
        player1_name="m",
        player2_name="opp",
        actions_p1=(ACTION_A, ACTION_A),
        actions_p2=(ACTION_B, ACTION_A),
        points_p1=(0, 8),
        points_p2=(10, 8),
        predictions_p1=(ACTION_B, ACTION_B),  # 1 correct (B), 1 wrong (predicted B, actual A)
        predictions_p2=(None, None),
    )
    table = prediction.prediction_accuracy_table([result])
    row = table[table["player"] == "m"].iloc[0]
    assert math.isclose(row["accuracy"], 0.5, rel_tol=1e-9)
    assert row["n_predictions"] == 2
