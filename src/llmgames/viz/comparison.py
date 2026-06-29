"""Compute a paper-vs-implementation comparison from released + local results.

The paper's released PD/BoS CSVs (``data/paper/``) use the same schema and J/F
actions as our runs, so we score both with the *same* metric functions — a true
apples-to-apples comparison. Produces a JSON-able dict for the replay's
"Paper vs Implementation" tab (score-ratio bars + the SCoT-effect on coordination).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..engine.families import canonical_bos, canonical_pd
from ..metrics import score_ratio
from ..players.base import MatchResult

# Display labels for the paper's player ids (act_gpt4_predictfirst = GPT-4 + SCoT).
_PAPER_LABELS = {
    "act_gpt4": "GPT-4",
    "act_gpt35": "GPT-3.5",
    "act_gpt3": "GPT-3",
    "act_claude": "Claude-2",
    "act_llama2": "Llama-2",
    "act_gpt4_predictfirst": "GPT-4+SCoT",
}
_PAPER_MODELS = set(_PAPER_LABELS)
_JF_TO_ACTION = {"J": "A", "F": "B"}
_STRATEGIES = {"tit_for_tat", "always_defect"}
_PD, _BOS = "PD Family", "BoS"


def _map_action(value: object) -> str:
    """Maps a CSV action cell (J/F or A/B) to an internal action."""
    text = str(value)
    return _JF_TO_ACTION.get(text, text if text in ("A", "B") else "?")


def _game_for(name: str):
    """Returns the canonical game for a game name."""
    return canonical_pd() if "Prisoner" in name else canonical_bos()


def results_from_csv(path: str | Path, default_game=None) -> list[MatchResult]:
    """Builds MatchResults from a rounds/experiment CSV (paper or implementation).

    Args:
        path: CSV with player1/player2, round, points1/2, and either action1/2 (A/B)
            or answer1/2 (J/F). A ``game_name`` column selects the game per match.
        default_game: Game to use when the CSV has no ``game_name`` column.

    Returns:
        One MatchResult per (game, player1, player2) match.
    """
    df = pd.read_csv(path)
    a1 = "action1" if "action1" in df.columns else "answer1"
    a2 = "action2" if "action2" in df.columns else "answer2"
    keys = ["game_name", "player1", "player2"] if "game_name" in df.columns else ["player1", "player2"]

    results: list[MatchResult] = []
    for key, group in df.groupby(keys, sort=False):
        group = group.sort_values("round")
        game = _game_for(key[0]) if "game_name" in df.columns else default_game
        results.append(
            MatchResult(
                game=game,
                player1_name=str(group["player1"].iloc[0]),
                player2_name=str(group["player2"].iloc[0]),
                actions_p1=tuple(_map_action(v) for v in group[a1]),
                actions_p2=tuple(_map_action(v) for v in group[a2]),
                points_p1=tuple(int(v) for v in group["points1"]),
                points_p2=tuple(int(v) for v in group["points2"]),
            )
        )
    return results


def _ratios(results: list[MatchResult], family: str, keep) -> dict[str, float]:
    """Returns ``{player: score_ratio}`` for a family, keeping players where keep(name)."""
    table = score_ratio.score_ratio_table(results)
    table = table[table["family"] == family]
    return {row.player: round(float(row.score_ratio), 3) for row in table.itertuples() if keep(row.player)}


def _is_paper_model(name: str) -> bool:
    """Returns True for the paper's model players (excludes hand-coded strategies)."""
    return name in _PAPER_MODELS


def build_comparison(
    paper_pd_csv: str | Path,
    paper_bos_csv: str | Path,
    impl_base_rounds: str | Path,
    impl_scot_rounds: str | Path | None = None,
) -> dict:
    """Builds the paper-vs-implementation comparison payload.

    Args:
        paper_pd_csv: Paper's released PD experiment CSV.
        paper_bos_csv: Paper's released BoS experiment CSV (includes GPT-4+SCoT).
        impl_base_rounds: Our base-mode rounds.csv (PD + BoS).
        impl_scot_rounds: Our SCoT rounds.csv (optional; for the SCoT-effect chart).

    Returns:
        A dict with ``pd``, ``bos`` (each {paper, impl} score-ratio lists) and
        ``scot_bos`` (per-model base->scot, paper GPT-4 first).
    """
    paper_pd = results_from_csv(paper_pd_csv, canonical_pd())
    paper_bos = results_from_csv(paper_bos_csv, canonical_bos())
    impl_base = results_from_csv(impl_base_rounds)

    keep_impl = lambda n: n not in _STRATEGIES and not n.endswith("+scot")
    pd_paper = _ratios(paper_pd, _PD, _is_paper_model)
    pd_impl = _ratios(impl_base, _PD, keep_impl)
    bos_paper = _ratios(paper_bos, _BOS, _is_paper_model)
    bos_impl = _ratios(impl_base, _BOS, keep_impl)

    def as_list(ratios: dict, label=lambda n: n) -> list[dict]:
        return [{"label": label(n), "ratio": r} for n, r in sorted(ratios.items(), key=lambda kv: -kv[1])]

    scot_bos: list[dict] = []
    # Paper GPT-4 base -> SCoT (predictfirst), from the paper's own BoS data.
    if "act_gpt4" in bos_paper and "act_gpt4_predictfirst" in bos_paper:
        scot_bos.append(
            {"label": "GPT-4 (paper)", "base": bos_paper["act_gpt4"], "scot": bos_paper["act_gpt4_predictfirst"]}
        )
    # Our models base -> SCoT.
    if impl_scot_rounds and Path(impl_scot_rounds).exists():
        scot_impl = _ratios(results_from_csv(impl_scot_rounds), _BOS, lambda n: n.endswith("+scot"))
        for name, base_ratio in sorted(bos_impl.items(), key=lambda kv: -kv[1]):
            scot_ratio = scot_impl.get(f"{name}+scot")
            if scot_ratio is not None:
                scot_bos.append({"label": name, "base": base_ratio, "scot": scot_ratio})

    return {
        "pd": {"paper": as_list(pd_paper, _PAPER_LABELS.get), "impl": as_list(pd_impl)},
        "bos": {"paper": as_list(bos_paper, _PAPER_LABELS.get), "impl": as_list(bos_impl)},
        "scot_bos": scot_bos,
    }
