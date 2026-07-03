"""The unrolled ADMM loop — the coordination core.

Each agent carries three states: the local proposal ``x``, the consensus iterate
``z``, and the scaled dual accumulator ``y`` (``u = lambda / rho``). One ADMM
iteration is::

    z_prev = z
    x       = prox_f(z - y; rho)                       # local x-update (x-solver)
    x_relax = alpha * x + (1 - alpha) * z_prev          # optional over-relaxation
    z       = consensus(x_relax + y; geometry, rho)     # sheaf z-update (z-solver)
    y       = y + (x_relax - z)                          # dual ascent

We unroll ``K`` iterations and backpropagate through the whole trajectory. The
loss is computed from the final ``loss_window`` x-iterates (decoded and
averaged), so the loop returns those. Memory options:

* full unroll (``grad_window=None``): gradients flow through all ``K`` steps.
* truncated BPTT (``grad_window=g``): the first ``K-g`` steps run detached
  (``stop_gradient``); gradients flow only through the last ``g``. Memory is
  ``O(g)`` instead of ``O(K)``.

Either way only the last ``loss_window`` x-iterates are materialized.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from sheaf_admm.geometry.base import SheafGeometry
from sheaf_admm.solvers.base import XSolver, XSolverParams, ZSolver, ZSolverParams


@struct.dataclass
class ADMMState:
    """Per-agent ADMM variables, each ``[N, B, d_v]``."""

    x: jnp.ndarray  # local proposal (primal)
    z: jnp.ndarray  # consensus iterate
    y: jnp.ndarray  # scaled dual accumulator (u = lambda / rho)


def inverse_softplus(x: float) -> float:
    """Numerically safe inverse of softplus, for initializing softplus-parameterized scalars.

    Guards ``log(expm1(x))`` against underflow (clamp to >=1e-7) and overflow
    (identity for large x), so any positive init value maps to a finite raw param.
    """
    x = max(x, 1e-7)
    return x if x > 20.0 else math.log(math.expm1(x))


def run_admm(
    encoder_output: dict[str, Any],
    geometry: SheafGeometry,
    x_solver: XSolver,
    x_params: XSolverParams,
    z_solver: ZSolver,
    z_params: ZSolverParams,
    rho: jnp.ndarray,
    z_init: jnp.ndarray,  # [N, B, d_v] initial consensus state (encoder h, or zeros)
    num_iters: int,
    *,
    relaxation_alpha: float = 1.0,
    loss_window: int = 1,
    grad_window: int | None = None,
) -> tuple[ADMMState, jnp.ndarray]:
    """Run ``num_iters`` ADMM steps.

    Returns the final :class:`ADMMState` and the stacked last ``loss_window``
    local proposals ``x`` of shape ``[W, N, B, d_v]`` (oldest first).
    """
    alpha = relaxation_alpha

    def step(state: ADMMState) -> ADMMState:
        z_prev = state.z
        x = x_solver.solve(state.z, state.y, rho, encoder_output, x_params)
        x_relaxed = x if alpha == 1.0 else alpha * x + (1.0 - alpha) * z_prev
        z_target = x_relaxed + state.y
        z = z_solver.solve(z_target, z_prev, geometry, z_params, rho)
        y = state.y + (x_relaxed - z)
        return ADMMState(x=x, z=z, y=y)

    state = ADMMState(x=z_init, z=z_init, y=jnp.zeros_like(z_init))

    K = num_iters
    n_detached = 0 if grad_window is None else max(0, K - grad_window)
    if n_detached > 0:
        state = jax.lax.fori_loop(0, n_detached, lambda _i, s: step(s), state)
        state = jax.tree_util.tree_map(jax.lax.stop_gradient, state)

    n_grad = K - n_detached
    window = min(loss_window, n_grad)
    n_pre = n_grad - window
    if n_pre > 0:
        state, _ = jax.lax.scan(lambda s, _: (step(s), None), state, None, length=n_pre)

    def collect(s, _):
        ns = step(s)
        return ns, ns.x

    state, x_window = jax.lax.scan(collect, state, None, length=window)
    return state, x_window


@struct.dataclass
class ADMMHistory:
    """Per-iteration ADMM trajectory, for visualization / convergence diagnostics.

    All arrays stack the ``K`` iterations on a leading axis (oldest first). The
    per-agent state arrays are ``[K, N, B, d_v]``; the residual arrays are the
    standard ADMM stopping diagnostics:

    * ``primal_res = ||x_relaxed - z||`` per agent (``[K, N, B]``) — how far the
      local proposal sits from the consensus it agreed to.
    * ``dual_res = rho * ||z - z_prev||`` per agent (``[K, N, B]``) — the change
      in the consensus iterate, scaled by the penalty (the dual feasibility
      residual of scaled-dual ADMM).
    * ``consistency_rms`` (``[K, B]``) — the geometry's sheaf-disagreement RMS,
      ``sqrt(mean over edges and stalk dims of r^2 + eps)``, the quantity the
      model is trained to drive to zero.
    """

    x: jnp.ndarray  # [K, N, B, d_v] local proposals
    z: jnp.ndarray  # [K, N, B, d_v] consensus iterates
    y: jnp.ndarray  # [K, N, B, d_v] scaled duals
    primal_res: jnp.ndarray  # [K, N, B] ||x_relaxed - z|| per agent
    dual_res: jnp.ndarray  # [K, N, B] rho * ||z - z_prev|| per agent
    consistency_rms: jnp.ndarray  # [K, B] sheaf disagreement RMS


def run_admm_history(
    encoder_output: dict[str, Any],
    geometry: SheafGeometry,
    x_solver: XSolver,
    x_params: XSolverParams,
    z_solver: ZSolver,
    z_params: ZSolverParams,
    rho: jnp.ndarray,
    z_init: jnp.ndarray,
    num_iters: int,
    *,
    relaxation_alpha: float = 1.0,
) -> tuple[ADMMState, ADMMHistory]:
    """Run ``num_iters`` ADMM steps, recording the full per-iteration trajectory.

    Same fixed-point iteration as :func:`run_admm`, but every iterate and its
    primal/dual residuals are stacked and returned in an :class:`ADMMHistory`
    (no gradient window — this is a forward-only diagnostic path). Use it to
    visualize coordination dynamics and x-vs-z trajectories on a single example.
    """
    alpha = relaxation_alpha
    rho_s = jnp.asarray(rho)

    def step(state: ADMMState, _):
        z_prev = state.z
        x = x_solver.solve(state.z, state.y, rho, encoder_output, x_params)
        x_relaxed = x if alpha == 1.0 else alpha * x + (1.0 - alpha) * z_prev
        z_target = x_relaxed + state.y
        z = z_solver.solve(z_target, z_prev, geometry, z_params, rho)
        y = state.y + (x_relaxed - z)
        new = ADMMState(x=x, z=z, y=y)
        primal_res = jnp.linalg.norm(x_relaxed - z, axis=-1)  # [N, B]
        dual_res = rho_s * jnp.linalg.norm(z - z_prev, axis=-1)  # [N, B]
        crms = geometry.consistency_rms(z)  # [B]
        return new, (new, primal_res, dual_res, crms)

    state = ADMMState(x=z_init, z=z_init, y=jnp.zeros_like(z_init))
    state, (states, primal_res, dual_res, crms) = jax.lax.scan(step, state, None, length=num_iters)
    history = ADMMHistory(
        x=states.x,
        z=states.z,
        y=states.y,
        primal_res=primal_res,
        dual_res=dual_res,
        consistency_rms=crms,
    )
    return state, history
