"""Extra metric coverage flagged by review: max_cell ratio, family grouping, trajectories."""

from __future__ import annotations

import math

from llmgames.engine.families import canonical_bos, canonical_pd
from llmgames.loop.match import play_match
from llmgames.metrics import bos_metrics, pd_metrics, score_ratio
from llmgames.players.strategies import make_strategy


def test_score_ratio_max_cell_method():
    """max_cell benchmark divides by the highest single-cell payoff (10 in PD)."""
    result = play_match(
        canonical_pd(), make_strategy("tit_for_tat"), make_strategy("always_defect"), num_rounds=5
    )
    ratio = score_ratio.match_score_ratio(result, for_player1=True, method="max_cell")
    assert math.isclose(ratio, 20 / 50, rel_tol=1e-9)  # achieved 20, 5 rounds x max cell 10


def test_score_ratio_table_groups_by_family():
    """Table 1 aggregates across game families."""
    pd_res = play_match(
        canonical_pd(), make_strategy("tit_for_tat"), make_strategy("always_defect"), num_rounds=5
    )
    bos_res = play_match(
        canonical_bos(), make_strategy("alternate"), make_strategy("alternate"), num_rounds=4
    )
    table = score_ratio.score_ratio_table([pd_res, bos_res])
    assert set(table["family"]) == {"PD Family", "BoS"}


def test_collaboration_trajectory_all_coordinated():
    """Two alternating BoS players coordinate every round."""
    res = play_match(
        canonical_bos(), make_strategy("alternate"), make_strategy("alternate"), num_rounds=4
    )
    traj = bos_metrics.collaboration_trajectory([res])
    assert (traj["collaboration_rate"] == 1.0).all()


def test_cooperation_trajectory_per_seat():
    """A constant cooperator shows cooperation_rate 1.0 each round."""
    res = play_match(
        canonical_pd(), make_strategy("always_cooperate"), make_strategy("always_defect"), num_rounds=3
    )
    coop = pd_metrics.cooperation_trajectory([res])
    cooperator = coop[coop["player"] == "always_cooperate"]
    assert (cooperator["cooperation_rate"] == 1.0).all()
