"""Bar charts for prediction accuracy (Fig 6) and human-vs-LLM summaries (Fig 7)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def save_bar(
    df: pd.DataFrame,
    *,
    category: str,
    value: str,
    title: str,
    path: str | Path,
    ylabel: str | None = None,
) -> Path:
    """Renders a simple bar chart and writes a PNG.

    Args:
        df: DataFrame with a category column and a numeric value column.
        category: Name of the category column (x-axis labels).
        value: Name of the value column (bar heights).
        title: Figure title.
        path: Output PNG path.
        ylabel: Optional y-axis label.

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
        labels = df[category].astype(str).tolist()
        ax.bar(labels, df[value].tolist(), color="#4C72B0")
        ax.set_ylabel(ylabel or value)
        ax.set_title(title)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.savefig(out, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return out


def save_grouped_bar(
    df: pd.DataFrame,
    *,
    category: str,
    group: str,
    value: str,
    title: str,
    path: str | Path,
    ylabel: str | None = None,
) -> Path:
    """Renders a grouped bar chart (e.g. metric by opponent condition) and writes a PNG.

    Args:
        df: Long-format DataFrame.
        category: Column for the primary x grouping.
        group: Column for the secondary grouping (side-by-side bars).
        value: Numeric value column.
        title: Figure title.
        path: Output PNG path.
        ylabel: Optional y-axis label.

    Returns:
        The output path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    if df.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out, bbox_inches="tight", dpi=120)
        plt.close(fig)
        return out

    pivot = df.pivot_table(index=category, columns=group, values=value, aggfunc="mean")
    categories = list(pivot.index)
    groups = list(pivot.columns)
    width = 0.8 / max(len(groups), 1)
    for gi, g in enumerate(groups):
        offsets = [x + gi * width for x in range(len(categories))]
        ax.bar(offsets, pivot[g].tolist(), width=width, label=str(g))
    ax.set_xticks([x + width * (len(groups) - 1) / 2 for x in range(len(categories))])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel(ylabel or value)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    fig.savefig(out, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return out
