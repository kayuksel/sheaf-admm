"""Run the ADMM coordinator and capture its per-iteration trajectory.

A thin, host-side wrapper around
:meth:`sheaf_admm.models.SheafADMMModel.coordinate_history`. It runs the forward
pass on a single (or small) batch and packs the result into a NumPy
:class:`Trajectory` the plotting modules consume — agent states ``x^k``/``z^k``,
the primal/dual residuals, the sheaf-consistency RMS, the per-iterate decoded
logits, and the (static) agent graph. Everything is pulled to host as NumPy so
the plotting code never touches JAX.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy as np

from sheaf_admm.models import SheafADMMModel


@dataclass
class Trajectory:
    """Host-side ADMM trajectory for one example (batch index already selected).

    Shapes (``K`` iterations, ``N`` agents, ``d_v`` stalk):

    * ``x``, ``z``, ``y``: ``[K, N, d_v]`` per-agent iterates.
    * ``primal_res``, ``dual_res``: ``[K, N]`` per-agent residuals.
    * ``consistency_rms``: ``[K]`` sheaf-disagreement RMS.
    * ``logits``: ``[K, N, *out]`` per-iterate decoded per-agent logits.
    * ``edge_indices``: ``[E, 2]`` agent graph (``u, v``).
    * ``centers``: ``[N, 2]`` agent ``(y, x)`` grid centers, or ``None`` (Sudoku).
    * ``rho``: the learned ADMM penalty (scalar).
    """

    x: np.ndarray
    z: np.ndarray
    y: np.ndarray
    primal_res: np.ndarray
    dual_res: np.ndarray
    consistency_rms: np.ndarray
    logits: np.ndarray
    edge_indices: np.ndarray
    centers: np.ndarray | None
    rho: float

    @property
    def num_iters(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_agents(self) -> int:
        return int(self.x.shape[1])


def run_trajectory(
    model: SheafADMMModel,
    params,
    fwd: dict,
    *,
    num_iters: int,
    batch_index: int = 0,
    centers: np.ndarray | None = None,
) -> Trajectory:
    """Run ``num_iters`` ADMM steps and return the host-side :class:`Trajectory`.

    ``fwd`` is a task ``prepare`` output (``patches``, ``edge_indices``,
    ``model_kwargs``). The trajectory is sliced to ``batch_index`` on the batch
    axis. ``centers`` (the grid task's agent centers, from ``aux``) is carried
    through for spatial layout; pass ``None`` for Sudoku.
    """
    history, logits_per_iter, _final, _geom, rho = model.apply(
        params,
        fwd["patches"],
        fwd["edge_indices"],
        num_iters=num_iters,
        **fwd["model_kwargs"],
        training=False,
        method=SheafADMMModel.coordinate_history,
    )
    b = batch_index
    return Trajectory(
        x=np.asarray(history.x[:, :, b]),
        z=np.asarray(history.z[:, :, b]),
        y=np.asarray(history.y[:, :, b]),
        primal_res=np.asarray(history.primal_res[:, :, b]),
        dual_res=np.asarray(history.dual_res[:, :, b]),
        consistency_rms=np.asarray(history.consistency_rms[:, b]),
        logits=np.asarray(logits_per_iter[:, :, b]),
        edge_indices=np.asarray(jax.device_get(fwd["edge_indices"])),
        centers=None if centers is None else np.asarray(centers),
        rho=float(rho),
    )
