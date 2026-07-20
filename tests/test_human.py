"""Tests for the Fig 7 human-study analysis over the real dataset."""

from __future__ import annotations

import pytest

from llmgames.metrics import human


def test_load_human_data_has_195_participants():
    """The released dataset contains exactly 195 participants."""
    df = human.load_human_data()
    assert df["id"].nunique() == 195
    assert set(df["opponent"].unique()) == {"Base", "Prompted"}
    assert set(df["game"].unique()) == {"PD", "BoS"}


def test_summarize_human_produces_conditions():
    """The summary reports one row per (game, opponent) with the expected columns."""
    df = human.load_human_data()
    summary = human.summarize_human(df)
    assert {"avg_score", "cooperation_rate", "success_rate", "p_human", "n_participants"} <= set(
        summary.columns
    )
    assert len(summary) == 4  # 2 games x 2 conditions
    assert (summary["p_human"].between(0, 1)).all()


def test_missing_human_data_raises(tmp_path):
    """A missing dataset raises rather than fabricating participants."""
    with pytest.raises(FileNotFoundError):
        human.load_human_data(tmp_path / "absent.csv")
