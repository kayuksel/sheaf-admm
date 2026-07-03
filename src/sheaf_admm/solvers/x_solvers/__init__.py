"""x-update solvers (per-agent local proximal step) and their registry.

* ``diagonal_prox`` — unified diagonal closed form (quadratic / lasso /
  non-negative / L1+box, selected by what the encoder emits); the solver the
  shipped sheaf configs use.
* ``simple`` — tether ``(beta/2)||x - h||^2``.
* ``dense_quadratic`` — linear solve for non-diagonal ``Q``; useful for tests
  and custom encoders that emit dense quadratic objectives.
"""

from __future__ import annotations

from .dense_quadratic import DenseQuadraticParams, DenseQuadraticXSolver
from .diagonal_prox import DiagonalProxParams, DiagonalProxXSolver
from .simple import SimpleParams, SimpleXSolver

X_SOLVERS = {
    "simple": (SimpleXSolver, SimpleParams),
    "diagonal_prox": (DiagonalProxXSolver, DiagonalProxParams),
    "dense_quadratic": (DenseQuadraticXSolver, DenseQuadraticParams),
}


def make_x_solver(name: str):
    """Return ``(solver_cls, default_params)`` for an x-solver name."""
    if name not in X_SOLVERS:
        raise KeyError(f"unknown x-solver {name!r}; available: {sorted(X_SOLVERS)}")
    cls, params_cls = X_SOLVERS[name]
    return cls, params_cls()


__all__ = [
    "X_SOLVERS",
    "make_x_solver",
    "SimpleXSolver",
    "SimpleParams",
    "DiagonalProxXSolver",
    "DiagonalProxParams",
    "DenseQuadraticXSolver",
    "DenseQuadraticParams",
]
