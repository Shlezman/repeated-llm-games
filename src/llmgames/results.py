"""Generate ``results.md`` — a single human-readable summary of a run.

Includes the run configuration, payoff matrices for the games played, the Table 1
score-ratio summary, and embedded figures (PD/BoS heat maps, trajectories,
prediction accuracy, and the Fig 7 human summary when the dataset is present).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .engine.game import ACTIONS, Game
from .loop.tournament import RunResult
from .metrics import bos_metrics, human, pd_metrics, prediction, score_ratio
from .prompts.transforms import DEFAULT_LABELS
from .viz import heatmaps, prediction_plots, trajectories
from .viz.replay import generate_replay_html

_PD_NAME = "Prisoner's Dilemma"
_BOS_NAME = "Battle of the Sexes"


def _payoff_markdown(game: Game, labels: tuple[str, str] = DEFAULT_LABELS) -> str:
    """Renders a game's payoff matrix as a markdown table of ``(p1, p2)`` cells."""
    la, lb = labels
    header = f"| P1 \\ P2 | {la} | {lb} |\n|---|---|---|\n"
    rows = []
    for ri, mine in enumerate(ACTIONS):
        cells = []
        for theirs in ACTIONS:
            p1, p2 = game.payoff(mine, theirs)
            cells.append(f"({p1}, {p2})")
        rows.append(f"| **{labels[ri]}** | {cells[0]} | {cells[1]} |")
    return header + "\n".join(rows)


def _df_markdown(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Renders a DataFrame as a GitHub-flavoured markdown table."""
    if df.empty:
        return "_no data_"
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda v: floatfmt.format(v) if pd.notna(v) else "")
    header = "| " + " | ".join(map(str, formatted.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in formatted.columns) + " |"
    body = "\n".join("| " + " | ".join(map(str, row)) + " |" for row in formatted.values)
    return "\n".join([header, sep, body])


def _filter(results: list, game_name: str) -> list:
    """Returns the subset of results for a given game name."""
    return [r for r in results if r.game.name == game_name]


def _config_section(run) -> str:
    """Renders the run-configuration section."""
    model_lines = "\n".join(
        f"- `{m.id}` — provider `{m.provider}`, model `{m.model}`, temp {m.params.temperature}"
        for m in run.models
    )
    return (
        "## Run configuration\n"
        f"- **Mode**: {run.mode}\n- **Rounds**: {run.rounds}\n- **Seed**: {run.seed}\n"
        f"- **Opponents**: {', '.join(run.opponents) or 'none'}\n"
        f"- **Robustness**: order_randomized={run.robustness.randomize_order}, "
        f"unit={run.robustness.utility_label}, cover_story={run.robustness.cover_story}, "
        f"labels={run.robustness.labels or 'J/F'}\n"
        f"- **Models**:\n{model_lines}\n"
    )


def _payoff_section(results: list) -> str:
    """Renders payoff matrices for every distinct game played."""
    games_seen: dict[str, Game] = {}
    for result in results:
        games_seen.setdefault(result.game.name, result.game)
    blocks = []
    for name, game in games_seen.items():
        labels = DEFAULT_LABELS if name in (_PD_NAME, _BOS_NAME) else ("A", "B")
        blocks.append(f"### {name} ({game.family})\n\n{_payoff_markdown(game, labels)}\n")
    return "## Payoff matrices\n\n" + "\n".join(blocks)


def _table1_section(results: list) -> str:
    """Renders the Table 1 score-ratio summary."""
    table1 = score_ratio.score_ratio_table(results)
    return "## Table 1 — score ratio (achieved / best achievable)\n\n" + _df_markdown(table1)


def _pd_section(pd_results: list, fig_dir: Path) -> str:
    """Renders the PD heat maps and cooperation trajectory, or empty if no PD data."""
    if not pd_results:
        return ""
    defection = heatmaps.save_heatmap(
        pd_metrics.defection_rate_matrix(pd_results),
        title="PD defection rate (player 1)", path=fig_dir / "pd_defection.png",
        cbar_label="defection rate",
    )
    score = heatmaps.save_heatmap(
        pd_metrics.score_matrix(pd_results),
        title="PD mean accrued score (player 1)", path=fig_dir / "pd_score.png",
        cbar_label="mean total", fmt="{:.0f}",
    )
    trajectory = trajectories.save_trajectory(
        pd_metrics.cooperation_trajectory(pd_results),
        value="cooperation_rate", group="player",
        title="PD cooperation over rounds", path=fig_dir / "pd_trajectory.png",
        ylabel="cooperation rate",
    )
    return (
        "## Prisoner's Dilemma\n\n"
        f"![PD defection](figures/{defection.name})\n\n"
        f"![PD score](figures/{score.name})\n\n"
        f"![PD trajectory](figures/{trajectory.name})\n"
    )


def _bos_section(bos_results: list, fig_dir: Path) -> str:
    """Renders the BoS collaboration and preferred-equilibrium heat maps."""
    if not bos_results:
        return ""
    collab = heatmaps.save_heatmap(
        bos_metrics.collaboration_rate_matrix(bos_results),
        title="BoS successful-collaboration rate", path=fig_dir / "bos_collab.png",
        cbar_label="collaboration rate",
    )
    preferred = heatmaps.save_heatmap(
        bos_metrics.preferred_frequency_matrix(bos_results),
        title="BoS P1-preferred frequency", path=fig_dir / "bos_preferred.png",
        cbar_label="P1-preferred rate",
    )
    return (
        "## Battle of the Sexes\n\n"
        f"![BoS collaboration](figures/{collab.name})\n\n"
        f"![BoS preferred](figures/{preferred.name})\n"
    )


def _prediction_section(results: list, fig_dir: Path) -> str:
    """Renders the SCoT prediction-accuracy table and bar chart (Fig 6)."""
    table = prediction.prediction_accuracy_table(results)
    if table.empty:
        return ""
    path = prediction_plots.save_bar(
        table, category="player", value="accuracy",
        title="SCoT opponent-prediction accuracy", path=fig_dir / "prediction_accuracy.png",
        ylabel="accuracy",
    )
    return (
        "## Prediction accuracy (theory of mind, Fig 6)\n\n"
        + _df_markdown(table)
        + f"\n\n![Prediction accuracy](figures/{path.name})\n"
    )


def _human_section(fig_dir: Path) -> str:
    """Renders the Fig 7 human-study summary, or a skip note if data is absent."""
    try:
        summary = human.summarize_human(human.load_human_data())
    except FileNotFoundError:
        return (
            "## Human study (Fig 7)\n\n"
            "_Requires the external participant dataset; not present — skipped. "
            "No human data is fabricated._\n"
        )
    path = prediction_plots.save_grouped_bar(
        summary, category="game", group="opponent", value="avg_score",
        title="Human vs LLM: average score by condition (Fig 7)",
        path=fig_dir / "human_avg_score.png", ylabel="avg score",
    )
    return (
        "## Human study (Fig 7)\n\n"
        "Real 195-participant dataset (Base vs SCoT-Prompted GPT-4).\n\n"
        + _df_markdown(summary)
        + f"\n\n![Human avg score](figures/{path.name})\n"
    )


def generate_results_md(run, run_result: RunResult) -> Path:
    """Builds figures and writes ``results.md`` into the run's output directory.

    Args:
        run: The :class:`~llmgames.config.schema.RunSpec` that produced the results.
        run_result: The tournament outputs.

    Returns:
        The path to the written ``results.md``.
    """
    out_dir = run_result.output_dir
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    results = run_result.results

    replay_html = generate_replay_html(
        run_result.rounds_csv,
        out_dir / "game_replay.html",
        run_name=run.name,
        thoughts_csv=run_result.thoughts_csv,
    )

    sections = [
        f"# Results — `{run.name}`\n",
        "Reproduction of Akata et al. (2025), *Playing repeated games with large "
        "language models* (Nature Human Behaviour). Generated by the `llmgames` harness.\n",
        f"▶ **Animated game replay:** open [`{replay_html.name}`]({replay_html.name}) "
        "in a browser to watch any matchup play out round by round.\n",
        _config_section(run),
        _payoff_section(results),
        _table1_section(results),
        _pd_section(_filter(results, _PD_NAME), fig_dir),
        _bos_section(_filter(results, _BOS_NAME), fig_dir),
        _prediction_section(results, fig_dir),
        _human_section(fig_dir),
    ]

    results_md = out_dir / "results.md"
    results_md.write_text("\n\n".join(s for s in sections if s), encoding="utf-8")
    return results_md
