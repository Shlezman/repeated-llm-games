"""Fig 7 human-vs-LLM analysis over the released 195-participant dataset.

The dataset (``data/human/repgames.csv``) is the authors' real participant data.
This module only *analyses* real data — it never synthesizes participants. If the
file is absent, callers get a clear error rather than fabricated results.

Column semantics (decoded from the released file):
    game: "PD" or "BoS"
    action: 1 = cooperate (J), 0 = defect (F)
    score: points won that round
    opponent: "Base" or "Prompted" (Prompted = SCoT-prompted GPT-4)
    guess: "Human" or "LLM" — what the participant thought they played
    coordination: 1 if both chose the same option that round, else 0
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_HUMAN_DATA = "data/human/repgames.csv"
_COOPERATE_ACTION = 1


def load_human_data(path: str | Path = DEFAULT_HUMAN_DATA) -> pd.DataFrame:
    """Loads the human-study dataset.

    Args:
        path: Path to the participant CSV.

    Returns:
        The raw dataset as a DataFrame.

    Raises:
        FileNotFoundError: If the dataset is not present (no data is fabricated).
    """
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Human-study data not found at {data_path}. Fig 7 requires the real "
            "participant dataset; this harness will not fabricate human data."
        )
    return pd.read_csv(data_path)


def summarize_human(df: pd.DataFrame) -> pd.DataFrame:
    """Summarizes outcomes per game and opponent condition.

    Args:
        df: The human-study dataset.

    Returns:
        A DataFrame indexed by ``[game, opponent]`` with average score, cooperation
        rate, and game-appropriate metrics derived from score: ``success_rate`` for
        BoS (coordinated == score != 0) and ``mutual_cooperation_rate`` for PD
        (both cooperate == score == 8). Each is NaN for the game where it is
        undefined, matching the paper (which defines coordination only for BoS).
        Also includes P(thought opponent was human) and counts.
    """
    df = df.copy()
    df["is_cooperate"] = (df["action"] == _COOPERATE_ACTION).astype(int)
    # Derive outcomes from score rather than the raw 'coordination' column, which is
    # unreliable for PD and diverges from the paper's score-based BoS definition.
    df["bos_success"] = np.where(df["game"] == "BoS", (df["score"] != 0).astype(float), np.nan)
    df["pd_mutual_coop"] = np.where(df["game"] == "PD", (df["score"] == 8).astype(float), np.nan)

    per_round = (
        df.groupby(["game", "opponent"])
        .agg(
            avg_score=("score", "mean"),
            cooperation_rate=("is_cooperate", "mean"),
            success_rate=("bos_success", "mean"),
            mutual_cooperation_rate=("pd_mutual_coop", "mean"),
            n_rounds=("score", "size"),
        )
        .reset_index()
    )

    # P(human) is a per-participant attribute; collapse to one value per participant first.
    per_participant = (
        df.groupby(["game", "opponent", "id"])["guess"].first().reset_index()
    )
    per_participant["thought_human"] = (per_participant["guess"] == "Human").astype(int)
    human_rate = (
        per_participant.groupby(["game", "opponent"])
        .agg(p_human=("thought_human", "mean"), n_participants=("id", "nunique"))
        .reset_index()
    )

    return per_round.merge(human_rate, on=["game", "opponent"], how="left")
