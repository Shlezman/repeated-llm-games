"""Round-by-round trajectory line plots over the repeated game."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def save_trajectory(
    df: pd.DataFrame,
    *,
    value: str,
    group: str,
    title: str,
    path: str | Path,
    round_col: str = "round",
    ylabel: str | None = None,
) -> Path:
    """Plots one line per group of ``value`` versus round and writes a PNG.

    Args:
        df: Long-format DataFrame with round, value, and group columns.
        value: Name of the value column to plot on the y-axis.
        group: Name of the column whose distinct values become separate lines.
        title: Figure title.
        path: Output PNG path.
        round_col: Name of the round column.
        ylabel: Optional y-axis label (defaults to ``value``).

    Returns:
        The output path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    if df.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.axis("off")
    else:
        for key, sub in df.groupby(group):
            sub = sub.sort_values(round_col)
            ax.plot(sub[round_col], sub[value], marker="o", label=str(key))
        ax.set_xlabel("round")
        ax.set_ylabel(ylabel or value)
        ax.set_title(title)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    fig.savefig(out, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return out
