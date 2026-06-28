"""Heat-map rendering for (player1 x player2) metric matrices (Fig 3 & 5)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render to files, never to a display
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def save_heatmap(
    matrix: pd.DataFrame,
    *,
    title: str,
    path: str | Path,
    cbar_label: str = "rate",
    fmt: str = "{:.2f}",
) -> Path:
    """Renders a labelled heat map of ``matrix`` and writes it to ``path``.

    Args:
        matrix: A pivot DataFrame (rows = player1, cols = player2).
        title: Figure title.
        path: Output PNG path.
        cbar_label: Colour-bar label.
        fmt: Cell annotation format string.

    Returns:
        The output path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if matrix.empty:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out, bbox_inches="tight", dpi=120)
        plt.close(fig)
        return out

    fig, ax = plt.subplots(figsize=(1.4 + 0.7 * len(matrix.columns), 1.4 + 0.6 * len(matrix.index)))
    image = ax.imshow(matrix.values, cmap="viridis", aspect="auto", vmin=0)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_xlabel("player 2")
    ax.set_ylabel("player 1")
    ax.set_title(title)

    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            value = matrix.values[i, j]
            if pd.notna(value):
                ax.text(j, i, fmt.format(value), ha="center", va="center", color="white", fontsize=7)

    fig.colorbar(image, ax=ax, label=cbar_label)
    fig.savefig(out, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return out
