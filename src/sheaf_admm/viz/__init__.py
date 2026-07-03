"""Visualization: turn a trained Sheaf-ADMM model into the paper's analysis artifacts.

Three static-image artifacts, all matplotlib (Agg backend), all driven off the
:mod:`sheaf_admm.training.tasks` ``prepare`` outputs:

* :func:`plot_prediction_evolution` (fig 3/4/5) — the aggregated global
  prediction at increasing ADMM iteration counts ``k``.
* :func:`plot_coordination_dynamics` (fig 4) — per-agent primal/dual residual
  heatmaps and the mean/max convergence curves.
* :func:`plot_xz_trajectories` (fig 7) — per-agent local proposal ``x`` vs
  consensus ``z`` 2-D trajectories on the agent grid.

The latter two consume a :class:`~sheaf_admm.viz.trajectory.Trajectory` built by
:func:`run_trajectory`, which records the full per-iteration ADMM path on a
single example.
"""

from __future__ import annotations

from .coordination import plot_coordination_dynamics
from .prediction import plot_prediction_evolution
from .trajectory import Trajectory, run_trajectory
from .xz import plot_xz_trajectories

__all__ = [
    "Trajectory",
    "run_trajectory",
    "plot_prediction_evolution",
    "plot_coordination_dynamics",
    "plot_xz_trajectories",
]
