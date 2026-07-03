"""Coordination-dynamics artifact (paper fig 4).

Visualizes how the ADMM residuals decay as the agents coordinate, from a single
:class:`~sheaf_admm.viz.trajectory.Trajectory`:

* two ``[N, K]`` heatmaps (agents x iteration) of the per-agent **primal**
  residual ``||x_i - z_i||`` and **dual** residual ``rho ||z_i - z_prev_i||``;
* a curves panel with the mean and max over agents of each residual, plus the
  sheaf-consistency RMS, on a log y-axis.

This is the convergence story: residuals start large and decay toward the
consensus fixed point.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from ._render import save_figure
from .trajectory import Trajectory


def _heatmap(ax, data_nk, title, cmap):
    """``data_nk``: ``[N, K]`` per-agent-over-iteration residual heatmap.

    Residuals routinely span several orders of magnitude (a large iteration-0
    transient collapsing to a small steady state), so the color scale is
    logarithmic, floored at the smallest positive entry to keep zeros finite.
    """
    pos = data_nk[data_nk > 0]
    vmin = float(pos.min()) if pos.size else 1e-12
    vmax = float(data_nk.max()) if data_nk.size else 1.0
    norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10))
    im = ax.imshow(
        np.maximum(data_nk, vmin),
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        origin="lower",
        norm=norm,
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("ADMM iteration k")
    ax.set_ylabel("agent i")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_coordination_dynamics(traj: Trajectory, out_path: str, *, title: str | None = None):
    """Render the primal/dual residual heatmaps and the mean/max curves.

    ``traj`` is a :class:`~sheaf_admm.viz.trajectory.Trajectory` (one example).
    Writes the figure to ``out_path``.
    """
    primal = traj.primal_res  # [K, N]
    dual = traj.dual_res  # [K, N]
    crms = traj.consistency_rms  # [K]
    ks = np.arange(1, traj.num_iters + 1)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))

    _heatmap(axes[0], primal.T, "primal residual  ||x_i - z_i||", "viridis")
    _heatmap(axes[1], dual.T, r"dual residual  $\rho\,||z_i - z^{-}_i||$", "magma")

    ax = axes[2]
    ax.plot(ks, primal.mean(axis=1), color="#1d4ed8", label="primal (mean)")
    ax.plot(ks, primal.max(axis=1), color="#1d4ed8", ls="--", alpha=0.6, label="primal (max)")
    ax.plot(ks, dual.mean(axis=1), color="#dc2626", label="dual (mean)")
    ax.plot(ks, dual.max(axis=1), color="#dc2626", ls="--", alpha=0.6, label="dual (max)")
    ax.plot(ks, crms, color="#16a34a", label="consistency RMS")
    ax.set_yscale("log")
    ax.set_xlabel("ADMM iteration k")
    ax.set_ylabel("residual (log)")
    ax.set_title("convergence", fontsize=11)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, which="both", alpha=0.2)

    if title:
        fig.suptitle(title, fontsize=12)
    save_figure(fig, out_path)
    return out_path
