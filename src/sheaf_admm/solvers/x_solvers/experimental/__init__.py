"""Reserved namespace for non-paper x-solvers.

The shipped configs use ``diagonal_prox`` from the public registry. This package
is intentionally empty so downstream experiments can add custom constrained or
dense-objective solvers without changing the default ADMM loop.
"""

from __future__ import annotations

__all__: list[str] = []
