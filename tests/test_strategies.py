"""Tests for the hand-coded baseline strategies."""

from __future__ import annotations

from llmgames.engine.families import canonical_pd
from llmgames.engine.game import ACTION_A, ACTION_B
from llmgames.players.base import PlayerView
from llmgames.players.strategies import make_strategy

GAME = canonical_pd()


def view(round_index: int, opponent_actions: tuple[str, ...] = ()) -> PlayerView:
    """Builds a minimal player view for strategy testing."""
    return PlayerView(
        game=GAME,
        am_row_player=True,
        round_index=round_index,
        num_rounds=10,
        opponent_actions=opponent_actions,
    )


def test_always_cooperate_and_defect():
    """Constant strategies ignore history."""
    assert make_strategy("always_cooperate").choose(view(3, (ACTION_B,))) == ACTION_A
    assert make_strategy("always_defect").choose(view(3, (ACTION_A,))) == ACTION_B


def test_defect_once():
    """Defects in round 1, cooperates afterwards."""
    s = make_strategy("defect_once")
    assert s.choose(view(1)) == ACTION_B
    assert s.choose(view(2, (ACTION_A,))) == ACTION_A


def test_alternate_starts_with_cooperate():
    """Alternate plays A, B, A across rounds."""
    s = make_strategy("alternate")
    assert [s.choose(view(r)) for r in (1, 2, 3)] == [ACTION_A, ACTION_B, ACTION_A]


def test_tit_for_tat_copies_opponent():
    """TFT cooperates first, then mirrors the opponent's last move."""
    s = make_strategy("tit_for_tat")
    assert s.choose(view(1)) == ACTION_A
    assert s.choose(view(2, (ACTION_B,))) == ACTION_B
    assert s.choose(view(3, (ACTION_B, ACTION_A))) == ACTION_A


def test_tit_for_two_tats():
    """TFTT defects only after two consecutive opponent defects."""
    s = make_strategy("tit_for_two_tats")
    assert s.choose(view(3, (ACTION_B, ACTION_A))) == ACTION_A
    assert s.choose(view(3, (ACTION_B, ACTION_B))) == ACTION_B


def test_suspicious_and_reverse_tft_start_with_defect():
    """STFT and RTFT both defect on the first move."""
    assert make_strategy("suspicious_tft").choose(view(1)) == ACTION_B
    rtft = make_strategy("reverse_tft")
    assert rtft.choose(view(1)) == ACTION_B
    assert rtft.choose(view(2, (ACTION_A,))) == ACTION_B  # opposite of opponent A
    assert rtft.choose(view(2, (ACTION_B,))) == ACTION_A


def test_hard_tft_three_round_memory():
    """HTFT defects if the opponent defected in any of the last three rounds."""
    s = make_strategy("hard_tft")
    assert s.choose(view(5, (ACTION_A, ACTION_A, ACTION_A, ACTION_B))) == ACTION_B
    assert s.choose(view(5, (ACTION_A, ACTION_A, ACTION_A, ACTION_A))) == ACTION_A


def test_naive_prober_is_seed_deterministic():
    """Same seed yields identical probabilistic strategy sequences across a match."""
    s1 = make_strategy("naive_prober_20", seed=7)
    s2 = make_strategy("naive_prober_20", seed=7)
    seq1 = [s1.choose(view(r, (ACTION_A,) * (r - 1))) for r in range(1, 11)]
    seq2 = [s2.choose(view(r, (ACTION_A,) * (r - 1))) for r in range(1, 11)]
    assert seq1 == seq2
