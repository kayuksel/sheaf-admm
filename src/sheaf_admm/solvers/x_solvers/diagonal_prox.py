"""Unified diagonal proximal x-solver (closed form).

A single closed form covers every diagonal local objective used in the paper.
For a diagonal curvature ``D``, linear term ``q``, per-coordinate L1 weight
``lambda``, and box ``[lo, hi]``:

    v = z - y
    a = D + l2 + rho
    t = (rho * v - q) / a
    x = clip( soft_threshold(t, lambda / a),  lo,  hi )

Special cases (selected purely by what the encoder emits):

* **quadratic** (``lambda=0``, no box): ``x = (rho v - q) / (D + rho)``.
* **lasso** (scalar ``lambda``, no box): MNIST.
* **non-negative** (``lambda=0``, ``lo=0``, ``hi=+inf``): Sudoku.
* **L1 + box** (per-dim ``lambda``, ``lo=0``, per-dim ``hi``): Maze.

This is exact — no inner iterations. All paper configs use diagonal ``Q``. The
dense quadratic case is implemented separately in ``dense_quadratic.py`` for
tests and code-level extensions.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from flax import struct

from sheaf_admm.solvers.base import XSolverParams, rho_as_vec, soft_threshold


@struct.dataclass
class DiagonalProxParams(XSolverParams):
    """Fallback box bounds used only when the encoder does not emit them.

    Defaults give an unconstrained problem (no box).
    """

    lower: float = -jnp.inf
    upper: float = jnp.inf


class DiagonalProxXSolver:
    """Closed-form diagonal proximal x-update (see module docstring).

    Reads from ``encoder_output``: ``q_diag`` ``[N,B,d]`` (diagonal curvature,
    strictly positive), ``q`` ``[N,B,d]`` (linear term), and optionally
    ``l1_weight`` ``[N,B,d]`` or scalar, ``l2_weight`` scalar, ``lower`` and
    ``upper`` ``[N,B,d]`` or scalar.
    """

    @staticmethod
    def solve(z, y, rho, encoder_output: dict[str, Any], params: DiagonalProxParams):
        rho_v = rho_as_vec(rho, z)
        diag = encoder_output["q_diag"]  # [N, B, d], > 0
        q = encoder_output["q"]
        l1 = encoder_output.get("l1_weight", 0.0)
        l2 = encoder_output.get("l2_weight", 0.0)
        lower = encoder_output.get("lower", params.lower)
        upper = encoder_output.get("upper", params.upper)

        a = diag + l2 + rho_v
        t = (rho_v * (z - y) - q) / a
        x = soft_threshold(t, l1 / a)
        return jnp.clip(x, lower, upper)
