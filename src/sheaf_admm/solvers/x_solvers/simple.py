"""Simple tether x-solver: ``f_i(x) = (beta/2) ||x - h||^2``.

The cheapest local objective: each agent is pulled toward an encoder-produced
target ``h`` with learned stiffness ``beta``. Closed form:

    x = (beta * h + rho * (z - y)) / (beta + rho)
"""

from __future__ import annotations

from typing import Any

from flax import struct

from sheaf_admm.solvers.base import XSolverParams, rho_as_vec


@struct.dataclass
class SimpleParams(XSolverParams):
    pass


class SimpleXSolver:
    """Tether x-update. Reads ``h`` ``[N,B,d]`` and scalar ``beta`` from the encoder."""

    @staticmethod
    def solve(z, y, rho, encoder_output: dict[str, Any], params: SimpleParams):
        rho_v = rho_as_vec(rho, z)
        h = encoder_output["h"]
        beta = encoder_output["beta"]
        return (beta * h + rho_v * (z - y)) / (beta + rho_v)
