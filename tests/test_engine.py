"""Tests for the game engine: payoffs, perspectives, scoring, taxonomy loading."""

from __future__ import annotations

import pytest

from llmgames.engine.families import canonical_bos, canonical_pd, load_games_csv
from llmgames.engine.game import ACTION_A, ACTION_B, Game
from llmgames.engine.outcomes import is_coordinated, is_mutual_cooperation, is_preferred_equilibrium
from llmgames.engine.payoff import INVALID_SCORE, score_round

GAMES_CSV = "config/games.csv"


def test_pd_payoffs_match_paper():
    """Canonical PD payoffs equal the paper's (8,8)/(0,10)/(10,0)/(5,5)."""
    pd = canonical_pd()
    assert pd.payoff(ACTION_A, ACTION_A) == (8, 8)
    assert pd.payoff(ACTION_A, ACTION_B) == (0, 10)
    assert pd.payoff(ACTION_B, ACTION_A) == (10, 0)
    assert pd.payoff(ACTION_B, ACTION_B) == (5, 5)


def test_bos_payoffs_match_paper():
    """Canonical BoS payoffs equal the paper's (10,7)/(0,0)/(0,0)/(7,10)."""
    bos = canonical_bos()
    assert bos.payoff(ACTION_A, ACTION_A) == (10, 7)
    assert bos.payoff(ACTION_A, ACTION_B) == (0, 0)
    assert bos.payoff(ACTION_B, ACTION_A) == (0, 0)
    assert bos.payoff(ACTION_B, ACTION_B) == (7, 10)


def test_payoff_for_perspective_is_symmetric():
    """Player 2's view of (my=A, opp=B) equals player 1's (A,B) with points swapped."""
    pd = canonical_pd()
    p1_mine, p1_opp = pd.payoff_for(ACTION_A, ACTION_B, am_row_player=True)
    p2_mine, p2_opp = pd.payoff_for(ACTION_A, ACTION_B, am_row_player=False)
    # Row player choosing A vs B -> (0, 10). Column player choosing A vs B uses cell (B,A)=(10,0)
    # oriented to the column player -> (0, 10).
    assert (p1_mine, p1_opp) == (0, 10)
    assert (p2_mine, p2_opp) == (0, 10)


def test_score_round_invalid_action_returns_sentinel():
    """An unparseable action yields the invalid-score sentinel for both players."""
    pd = canonical_pd()
    assert score_round(pd, ACTION_A, "?") == (INVALID_SCORE, INVALID_SCORE)


def test_game_requires_complete_matrix():
    """Constructing a game with a missing cell raises ValueError."""
    with pytest.raises(ValueError):
        Game(name="bad", family="x", payoffs={(ACTION_A, ACTION_A): (1, 1)})


def test_load_144_games_and_family_counts():
    """The taxonomy loads 144 games with the expected family distribution."""
    games = load_games_csv(GAMES_CSV)
    assert len(games) == 144
    counts: dict[str, int] = {}
    for game in games:
        counts[game.family] = counts.get(game.family, 0) + 1
    assert counts == {
        "Win-win": 36,
        "Biased": 44,
        "Unfair": 19,
        "Cyclic": 18,
        "Second Best": 12,
        "Tragic": 8,
        "PD Family": 7,
    }


def test_outcome_classifiers():
    """Outcome helpers classify coordination and preferred equilibria correctly."""
    assert is_mutual_cooperation(ACTION_A, ACTION_A)
    assert not is_mutual_cooperation(ACTION_A, ACTION_B)
    assert is_coordinated(ACTION_B, ACTION_B)
    assert not is_coordinated(ACTION_A, ACTION_B)
    assert is_preferred_equilibrium(ACTION_A, ACTION_A)
    assert not is_preferred_equilibrium(ACTION_B, ACTION_B)
