"""Low-level matplotlib rendering helpers shared by the artifact modules.

Single-axis panel renderers (no figure management) plus a figure-save helper that
infers the format from the path suffix. Kept deliberately plain — clean grids,
muted palettes, no styling state the caller has to reset — so the higher-level
``plot_*`` functions just lay out a grid of these.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


def save_figure(fig, out_path: str, *, dpi: int = 150) -> None:
    """Tight-layout and write ``fig`` to ``out_path`` (PNG/PDF by suffix), then close it."""
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _blank_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])


# =============================================================================
# Maze
# =============================================================================


def render_maze_panel(ax, maze_img, pred, wall, start, goal, path_token) -> None:
    """Grayscale maze with the predicted path overlaid in blue.

    ``maze_img``: ``[H, W]`` ground-truth/label tokens (walls vs free space).
    ``pred``: ``[H, W]`` predicted tokens; cells equal to ``path_token`` are the
    predicted path. Start / goal cells get green / red markers.
    """
    maze_img = np.asarray(maze_img)
    pred = np.asarray(pred)
    base = np.where(maze_img == wall, 0.15, 1.0)  # walls dark, free space light
    ax.imshow(base, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")

    path = pred == path_token
    overlay = np.zeros((*path.shape, 4), dtype=np.float32)
    overlay[path] = (0.10, 0.35, 0.85, 0.85)  # blue, semi-opaque
    ax.imshow(overlay, interpolation="nearest")

    for token, color in ((start, "#16a34a"), (goal, "#dc2626")):
        ys, xs = np.where(maze_img == token)
        ax.scatter(xs, ys, s=42, c=color, marker="s", edgecolors="white", linewidths=0.6, zorder=3)
    _blank_axis(ax)


# =============================================================================
# Sudoku
# =============================================================================


def sudoku_violations(pred: np.ndarray) -> np.ndarray:
    """Boolean ``[9, 9]`` mask of cells whose digit repeats in its row, column, or box."""
    pred = np.asarray(pred)
    viol = np.zeros((9, 9), dtype=bool)

    def mark(cells):
        # cells: list of (r, c); flag any digit (1-9) appearing more than once.
        vals = [pred[r, c] for r, c in cells]
        for d in range(1, 10):
            idx = [i for i, v in enumerate(vals) if v == d]
            if len(idx) > 1:
                for i in idx:
                    r, c = cells[i]
                    viol[r, c] = True

    for r in range(9):
        mark([(r, c) for c in range(9)])
    for c in range(9):
        mark([(r, c) for r in range(9)])
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            mark([(br + i, bc + j) for i in range(3) for j in range(3)])
    return viol


def render_sudoku_panel(ax, givens, pred) -> None:
    """9x9 Sudoku board: given clues black, filled digits blue, violations red-shaded.

    ``givens``: ``[9, 9]`` input grid (0 = empty). ``pred``: ``[9, 9]`` predicted
    digits (1-9; 0 left blank).
    """
    givens = np.asarray(givens)
    pred = np.asarray(pred)
    viol = sudoku_violations(pred)

    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.invert_yaxis()
    ax.set_aspect("equal")

    # Violation shading behind the digits.
    for r in range(9):
        for c in range(9):
            if viol[r, c]:
                ax.add_patch(plt.Rectangle((c, r), 1, 1, color="#fca5a5", zorder=0))

    # Grid lines: thin for cells, thick for 3x3 boxes.
    for i in range(10):
        lw = 2.0 if i % 3 == 0 else 0.5
        ax.plot([i, i], [0, 9], color="black", lw=lw, zorder=1)
        ax.plot([0, 9], [i, i], color="black", lw=lw, zorder=1)

    for r in range(9):
        for c in range(9):
            d = int(pred[r, c])
            if d == 0:
                continue
            given = int(givens[r, c]) != 0
            color = "black" if given else "#1d4ed8"
            weight = "bold" if given else "normal"
            ax.text(
                c + 0.5,
                r + 0.5,
                str(d),
                ha="center",
                va="center",
                fontsize=11,
                color=color,
                fontweight=weight,
                zorder=2,
            )
    _blank_axis(ax)


# =============================================================================
# MNIST (per-pixel class consensus)
# =============================================================================

_MNIST_CMAP = ListedColormap(plt.get_cmap("tab10").colors)


def render_mnist_panel(ax, pred, num_classes: int = 10) -> None:
    """Per-pixel argmax-class consensus heatmap (``[H, W]`` class ids in ``tab10``)."""
    pred = np.asarray(pred)
    norm = BoundaryNorm(np.arange(-0.5, num_classes + 0.5, 1.0), num_classes)
    ax.imshow(pred, cmap=_MNIST_CMAP, norm=norm, interpolation="nearest")
    _blank_axis(ax)
