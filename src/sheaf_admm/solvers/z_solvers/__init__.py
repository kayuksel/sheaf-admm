"""z-update solvers (sheaf consensus) and their registry.

* ``unrolled_cg`` — fixed-``T`` conjugate gradient, the default z-solver.
* ``gd`` — gradient descent with Nesterov momentum (the GD+Nesterov ablation).
"""

from __future__ import annotations

from .gd import GDParams, GDZSolver
from .unrolled_cg import UnrolledCGParams, UnrolledCGZSolver

Z_SOLVERS = {
    "unrolled_cg": (UnrolledCGZSolver, UnrolledCGParams),
    "gd": (GDZSolver, GDParams),
}


def make_z_solver(name: str):
    """Return ``(solver_cls, default_params)`` for a z-solver name."""
    if name not in Z_SOLVERS:
        raise KeyError(f"unknown z-solver {name!r}; available: {sorted(Z_SOLVERS)}")
    cls, params_cls = Z_SOLVERS[name]
    return cls, params_cls()


__all__ = [
    "Z_SOLVERS",
    "make_z_solver",
    "UnrolledCGZSolver",
    "UnrolledCGParams",
    "GDZSolver",
    "GDParams",
]
