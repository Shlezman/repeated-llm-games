"""Game catalogue: canonical PD/BoS plus the 144-game ordinal taxonomy.

The 144-game CSV is the dataset released by Akata et al. (2025), reproduced under
its MIT license. Column convention (``;``-delimited)::

    game_number;game_row;game_col;game_name;game_family;
    r1_c1_p1;r1_c1_p2;r1_c2_p1;r1_c2_p2;r2_c1_p1;r2_c1_p2;r2_c2_p1;r2_c2_p2

Row option 1 maps to :data:`~llmgames.engine.game.ACTION_A`, option 2 to
``ACTION_B``; ``_p1``/``_p2`` are the row/column player payoffs.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .game import ACTION_A, ACTION_B, Game

# Canonical payoffs as released in the paper (A = "cooperate"/preferred, B = other).
_PD_PAYOFFS = {
    (ACTION_A, ACTION_A): (8, 8),
    (ACTION_A, ACTION_B): (0, 10),
    (ACTION_B, ACTION_A): (10, 0),
    (ACTION_B, ACTION_B): (5, 5),
}
_BOS_PAYOFFS = {
    (ACTION_A, ACTION_A): (10, 7),
    (ACTION_A, ACTION_B): (0, 0),
    (ACTION_B, ACTION_A): (0, 0),
    (ACTION_B, ACTION_B): (7, 10),
}


def canonical_pd() -> Game:
    """Returns the canonical iterated Prisoner's Dilemma (A=cooperate, B=defect)."""
    return Game(name="Prisoner's Dilemma", family="PD Family", payoffs=dict(_PD_PAYOFFS))


def canonical_bos() -> Game:
    """Returns the canonical Battle of the Sexes (A = player-1-preferred option)."""
    return Game(name="Battle of the Sexes", family="BoS", payoffs=dict(_BOS_PAYOFFS))


def _row_to_game(row: dict[str, str]) -> Game:
    """Builds a :class:`Game` from one CSV row of the 144-game taxonomy.

    Args:
        row: A parsed CSV row keyed by the column names above.

    Returns:
        The corresponding :class:`Game` with ordinal payoffs.
    """
    payoffs = {
        (ACTION_A, ACTION_A): (int(row["r1_c1_p1"]), int(row["r1_c1_p2"])),
        (ACTION_A, ACTION_B): (int(row["r1_c2_p1"]), int(row["r1_c2_p2"])),
        (ACTION_B, ACTION_A): (int(row["r2_c1_p1"]), int(row["r2_c1_p2"])),
        (ACTION_B, ACTION_B): (int(row["r2_c2_p1"]), int(row["r2_c2_p2"])),
    }
    return Game(name=row["game_name"], family=row["game_family"], payoffs=payoffs)


def load_games_csv(path: str | Path) -> list[Game]:
    """Loads all games from the 144-game taxonomy CSV.

    Args:
        path: Filesystem path to the ``;``-delimited games CSV.

    Returns:
        A list of :class:`Game` objects in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    games_path = Path(path)
    with games_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        return [_row_to_game(row) for row in reader]


def filter_by_family(games: list[Game], families: list[str]) -> list[Game]:
    """Returns the subset of ``games`` whose family is in ``families``."""
    wanted = set(families)
    return [game for game in games if game.family in wanted]
