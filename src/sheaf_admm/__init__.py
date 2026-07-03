"""Sheaf-ADMM: learning multi-agent coordination via sheaf-constrained ADMM.

A differentiable optimization framework. An input is decomposed into overlapping
local views, each processed by an agent that solves a convex subproblem
parameterized by a shared neural encoder. Agents coordinate through ADMM with
inter-agent constraints specified by a cellular sheaf, and a shared decoder maps
the final agent states back to a global prediction. The whole pipeline is
differentiable and trained end-to-end.

Reference implementation for "Learning Multi-Agent Coordination via Sheaf-ADMM"
(ICML 2026).
"""

from __future__ import annotations

from .precision import set_high_precision

# Enforce true float32 matmuls before any compilation happens (see precision.py).
set_high_precision()

__version__ = "0.1.0"

__all__ = ["__version__", "set_high_precision"]
