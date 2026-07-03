"""x-vs-z trajectory artifact (paper fig 7).

For each agent, plot the local proposal ``x_i^k`` (blue) and the consensus
iterate ``z_i^k`` (red) across ADMM iterations ``k``, as a small per-agent
2-D phase portrait. The panels are laid out on the agent grid (grid tasks use
the actual ``(y, x)`` centers; Sudoku falls back to a square index grid).

When the stalk dimension ``d_v > 2`` the iterates are projected to 2-D with a
single PCA basis fit jointly over *all* agents' ``x`` and ``z`` across all
iterations, so the panels share one coordinate frame and are comparable. The
shared frame is the point: as ADMM converges, each agent's blue ``x`` and red
``z`` paths should collapse onto a common point.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ._render import save_figure
from .trajectory import Trajectory


def _pca_basis(points: np.ndarray):
    """Fit a 2-component PCA on ``[M, d]`` points; return ``(mean [d], components [d, 2])``."""
    mean = points.mean(axis=0)
    centered = points - mean
    # SVD of the centered data; right-singular vectors are the principal axes.
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    comps = vt[:2].T  # [d, 2]
    return mean, comps


def _project(traj: Trajectory):
    """Project ``x``/``z`` (``[K, N, d_v]``) to 2-D. Returns ``(x2 [K,N,2], z2 [K,N,2])``."""
    K, N, d = traj.x.shape
    if d == 2:
        return traj.x.copy(), traj.z.copy()
    if d == 1:
        zeros_x = np.zeros((K, N, 1), dtype=traj.x.dtype)
        zeros_z = np.zeros((K, N, 1), dtype=traj.z.dtype)
        return np.concatenate([traj.x, zeros_x], -1), np.concatenate([traj.z, zeros_z], -1)
    stacked = np.concatenate([traj.x.reshape(-1, d), traj.z.reshape(-1, d)], axis=0)
    mean, comps = _pca_basis(stacked)
    x2 = ((traj.x.reshape(-1, d) - mean) @ comps).reshape(K, N, 2)
    z2 = ((traj.z.reshape(-1, d) - mean) @ comps).reshape(K, N, 2)
    return x2, z2


def _grid_layout(traj: Trajectory):
    """Map each agent to a ``(row, col)`` subplot slot and return ``(nrows, ncols, slots)``."""
    N = traj.num_agents
    if traj.centers is not None:
        ys = np.unique(traj.centers[:, 0])
        xs = np.unique(traj.centers[:, 1])
        row_of = {v: i for i, v in enumerate(ys)}
        col_of = {v: i for i, v in enumerate(xs)}
        slots = [(row_of[int(c[0])], col_of[int(c[1])]) for c in traj.centers]
        return len(ys), len(xs), slots
    ncols = int(np.ceil(np.sqrt(N)))
    nrows = int(np.ceil(N / ncols))
    slots = [(i // ncols, i % ncols) for i in range(N)]
    return nrows, ncols, slots


def plot_xz_trajectories(traj: Trajectory, out_path: str, *, title: str | None = None):
    """Render the per-agent x (blue) vs z (red) 2-D trajectory grid.

    ``traj`` is a :class:`~sheaf_admm.viz.trajectory.Trajectory` (one example).
    Writes the figure to ``out_path``.
    """
    x2, z2 = _project(traj)  # [K, N, 2]
    nrows, ncols, slots = _grid_layout(traj)

    fig, axes = plt.subplots(nrows, ncols, figsize=(1.7 * ncols, 1.7 * nrows), squeeze=False)
    for ax_row in axes:
        for ax in ax_row:
            ax.set_visible(False)

    # Shared axis limits across panels so trajectories are comparable.
    allpts = np.concatenate([x2.reshape(-1, 2), z2.reshape(-1, 2)], axis=0)
    lo, hi = allpts.min(axis=0), allpts.max(axis=0)
    pad = 0.05 * np.maximum(hi - lo, 1e-6)
    lo, hi = lo - pad, hi + pad

    for a, (r, col) in enumerate(slots):
        ax = axes[r][col]
        ax.set_visible(True)
        ax.plot(x2[:, a, 0], x2[:, a, 1], color="#1d4ed8", lw=1.0, marker="o", ms=2.0, label="x")
        ax.plot(z2[:, a, 0], z2[:, a, 1], color="#dc2626", lw=1.0, marker="o", ms=2.0, label="z")
        # Mark the start (open) and end (filled) of the consensus path.
        ax.scatter(
            z2[0, a, 0], z2[0, a, 1], facecolors="none", edgecolors="#dc2626", s=22, zorder=3
        )
        ax.scatter(z2[-1, a, 0], z2[-1, a, 1], color="#111827", s=18, zorder=3)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"agent {a}", fontsize=6)

    handles = [
        plt.Line2D([], [], color="#1d4ed8", marker="o", ms=3, label="x (local)"),
        plt.Line2D([], [], color="#dc2626", marker="o", ms=3, label="z (consensus)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9)
    if title:
        fig.suptitle(title, fontsize=12)
    save_figure(fig, out_path)
    return out_path
