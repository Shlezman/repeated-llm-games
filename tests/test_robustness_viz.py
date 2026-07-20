"""Tests for the robustness driver helpers and the replay HTML data builders."""

from __future__ import annotations

import pandas as pd
import pytest

from llmgames.engine.game import ACTION_A, ACTION_B
from llmgames.robustness import _bos_sweep, _rates
from llmgames.viz.replay import build_heatmaps, build_robustness


class _Result:
    """Minimal stand-in for MatchResult carrying only the action lists."""

    def __init__(self, a1, a2):
        self.actions_p1 = a1
        self.actions_p2 = a2


class TestRates:
    def test_defection_and_coordination(self):
        res = _Result([ACTION_A, ACTION_B, ACTION_B], [ACTION_A, ACTION_B, ACTION_A])
        defect, coord = _rates(res)
        assert defect == pytest.approx(2 / 3)
        assert coord == pytest.approx(2 / 3)

    def test_invalid_rounds_excluded(self):
        res = _Result([ACTION_A, "?", ACTION_B], [ACTION_A, ACTION_A, "?"])
        defect, coord = _rates(res)
        assert (defect, coord) == (0.0, 1.0)

    def test_all_invalid_returns_zero(self):
        assert _rates(_Result(["?"], ["?"])) == (0.0, 0.0)


class TestBosSweep:
    def test_paper_matrices(self):
        games = _bos_sweep()
        assert [g.payoffs[(ACTION_A, ACTION_A)] for g in games] == [
            (10, 7), (9, 8), (8, 8), (8, 9), (7, 10)
        ]
        for g in games:
            jj = g.payoffs[(ACTION_A, ACTION_A)]
            assert g.payoffs[(ACTION_B, ACTION_B)] == (jj[1], jj[0])
            assert g.payoffs[(ACTION_A, ACTION_B)] == (0, 0)
            assert g.payoffs[(ACTION_B, ACTION_A)] == (0, 0)


class TestBuildRobustness:
    def test_missing_csv_returns_none(self, tmp_path):
        assert build_robustness(None) is None
        assert build_robustness(tmp_path / "missing.csv") is None

    def test_matrix_layout(self, tmp_path):
        csv = tmp_path / "robustness.csv"
        pd.DataFrame(
            [
                {"arm": "framing", "variant": "base", "model": "m1", "game": "PD",
                 "metric": "defection_rate", "value": 0.5},
                {"arm": "framing", "variant": "labels=Q/X", "model": "m1", "game": "PD",
                 "metric": "defection_rate", "value": 0.9},
                {"arm": "ending_prob", "variant": "0% end/round", "model": "m1", "game": "PD",
                 "metric": "defection_rate", "value": 0.1},
            ]
        ).to_csv(csv, index=False)
        rb = build_robustness(csv)
        assert [a["arm"] for a in rb["arms"]] == ["Framing invariance", "PD ending probability"]
        m = rb["arms"][0]["matrices"][0]
        assert m["cols"] == ["base", "labels=Q/X"]
        assert m["cells"]["m1"]["labels=Q/X"] == 0.9


class TestBuildHeatmaps:
    def _rounds(self):
        rows = []
        for r, (a1, a2) in enumerate([("A", "A"), ("B", "A")], start=1):
            rows.append({"game_name": "Prisoner's Dilemma", "mode": "base",
                         "player1": "model_x", "player2": "tit_for_tat", "round": r,
                         "action1": a1, "action2": a2, "total1": 10 * r, "total2": 8})
        rows.append({"game_name": "Battle of the Sexes", "mode": "base",
                     "player1": "model_x", "player2": "tit_for_tat", "round": 1,
                     "action1": "A", "action2": "A", "total1": 10, "total2": 7})
        return pd.DataFrame(rows)

    def test_maps_and_ordering(self):
        hm = build_heatmaps(self._rounds())
        titles = [m["title"] for m in hm["maps"]]
        assert len(titles) == 3
        defect = hm["maps"][0]
        assert defect["players"] == ["model_x", "tit_for_tat"]  # models before strategies
        assert defect["cells"]["model_x"]["tit_for_tat"] == 0.5
        score = hm["maps"][1]
        assert score["cells"]["model_x"]["tit_for_tat"] == 20  # total1 at last round
        coord = hm["maps"][2]
        assert coord["cells"]["model_x"]["tit_for_tat"] == 1.0

    def test_non_base_modes_excluded(self):
        df = self._rounds()
        df["mode"] = "scot"
        assert build_heatmaps(df) is None
