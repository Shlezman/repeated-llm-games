"""Tests for the animated HTML game-replay generator."""

from __future__ import annotations

import json
import re

import pandas as pd
import pytest

from llmgames.viz.replay import generate_replay_html

_ROWS = [
    {"game_name": "Prisoner's Dilemma", "family": "PD Family", "mode": "base",
     "player1": "m1", "player2": "m2", "round": 1, "action1": "A", "action2": "B",
     "points1": 0, "points2": 10, "total1": 0, "total2": 10, "pred1": "", "pred2": ""},
    {"game_name": "Prisoner's Dilemma", "family": "PD Family", "mode": "base",
     "player1": "m1", "player2": "m2", "round": 2, "action1": "B", "action2": "B",
     "points1": 5, "points2": 5, "total1": 5, "total2": 15, "pred1": "", "pred2": ""},
]


def _make_csv(tmp_path):
    """Writes a tiny rounds CSV and returns its path."""
    path = tmp_path / "rounds.csv"
    pd.DataFrame(_ROWS).to_csv(path, index=False)
    return path


def test_replay_html_embeds_match_data(tmp_path):
    """The generated HTML is self-contained and embeds the match/leaderboard data."""
    out = generate_replay_html(_make_csv(tmp_path), tmp_path / "replay.html", run_name="t")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "const DATA" in text and "<svg" in text  # self-contained: data + inline chart

    data = json.loads(re.search(r"const DATA = (\{.*?\});", text, re.S).group(1))
    assert len(data["matches"]) == 1
    match = data["matches"][0]
    assert (match["p1"], match["p2"]) == ("m1", "m2")
    assert match["rounds"][0]["a1"] == "J" and match["rounds"][0]["a2"] == "F"  # A->J, B->F
    assert match["total1"] == 5 and match["total2"] == 15
    assert {row["player"] for row in data["leaderboard"]} == {"m1", "m2"}


def test_replay_missing_csv_raises(tmp_path):
    """A missing rounds CSV raises rather than producing an empty report."""
    with pytest.raises(FileNotFoundError):
        generate_replay_html(tmp_path / "absent.csv", tmp_path / "out.html")
