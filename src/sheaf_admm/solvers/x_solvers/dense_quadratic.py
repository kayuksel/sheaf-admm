"""Dense quadratic x-solver: ``f_i(x) = 1/2 x^T Q x + q^T x`` with full ``Q``.

For a non-diagonal PSD ``Q`` the prox step is a single linear solve:

    (Q + rho I) x = rho (z - y) - q

Use :class:`~sheaf_admm.solvers.x_solvers.diagonal_prox.DiagonalProxXSolver`
when ``Q`` is diagonal (the paper configs) — it is exact and avoids the solve.
This solver covers low-rank / dense ``Q`` and is the natural extension point for
richer local objectives.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from flax import struct

from sheaf_admm.solvers.base import XSolverParams, rho_as_vec


@struct.dataclass
class DenseQuadraticParams(XSolverParams):
    pass


class DenseQuadraticXSolver:
    """Dense quadratic x-update. Reads ``Q`` ``[N,B,d,d]`` and ``q`` ``[N,B,d]``."""

    @staticmethod
    def solve(z, y, rho, encoder_output: dict[str, Any], params: DenseQuadraticParams):
        Q = encoder_output["Q"]  # [N, B, d, d]
        q = encoder_output["q"]  # [N, B, d]
        d = Q.shape[-1]
        rho_v = rho_as_vec(rho, z)  # scalar or [N, B, 1]
        A = Q + rho_v[..., None] * jnp.eye(d, dtype=Q.dtype)
        b = rho_v * (z - y) - q  # [N, B, d]
        return jnp.linalg.solve(A, b[..., None]).squeeze(-1)
