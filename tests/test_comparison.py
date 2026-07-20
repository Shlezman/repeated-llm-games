"""Tests for the paper-vs-implementation comparison builder."""

from __future__ import annotations

import pandas as pd

from llmgames.viz.comparison import build_comparison

_PAPER_PD = "data/paper/pd.csv"
_PAPER_BOS = "data/paper/bos.csv"


def _impl_rounds(tmp_path):
    """Writes a tiny implementation rounds.csv (PD + BoS, one model vs a strategy)."""
    rows = []
    # PD: model_a cooperates, strategy defects -> a couple of rounds.
    for rnd, (a1, a2, p1, p2) in enumerate([("A", "B", 0, 10), ("B", "B", 5, 5)], start=1):
        rows.append({"game_name": "Prisoner's Dilemma", "player1": "model_a", "player2": "tit_for_tat",
                     "round": rnd, "action1": a1, "action2": a2, "points1": p1, "points2": p2})
    # BoS: coordinate on J both rounds.
    for rnd in (1, 2):
        rows.append({"game_name": "Battle of the Sexes", "player1": "model_a", "player2": "tit_for_tat",
                     "round": rnd, "action1": "A", "action2": "A", "points1": 10, "points2": 7})
    path = tmp_path / "rounds.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_build_comparison_has_paper_and_impl(tmp_path):
    """Comparison includes paper models (incl GPT-4+SCoT) and the implementation models."""
    c = build_comparison(_PAPER_PD, _PAPER_BOS, _impl_rounds(tmp_path))
    pd_paper = {row["label"] for row in c["pd"]["paper"]}
    bos_paper = {row["label"] for row in c["bos"]["paper"]}
    assert "GPT-4" in pd_paper and "GPT-3" in pd_paper          # paper PD models
    assert "GPT-4+SCoT" in bos_paper                            # paper's released SCoT run
    assert any(row["label"] == "model_a" for row in c["pd"]["impl"])  # our model present


def test_scot_effect_shows_paper_gpt4_lift(tmp_path):
    """The SCoT-effect series leads with the paper's GPT-4 base->SCoT lift in BoS."""
    c = build_comparison(_PAPER_PD, _PAPER_BOS, _impl_rounds(tmp_path))
    gpt4 = c["scot_bos"][0]
    assert gpt4["label"] == "GPT-4 (paper)"
    assert gpt4["scot"] > gpt4["base"]  # SCoT improves coordination
