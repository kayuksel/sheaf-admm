"""Gradient-descent z-solver (sheaf diffusion with Nesterov momentum).

The alternative to CG for the consensus step, and the paper's "GD + Nesterov"
z-update ablation. Runs a fixed number of accelerated gradient steps on the same
objective CG solves:

* ``project``: ``E(z) = E_sheaf(z)`` (minimized over the consensus directions).
* ``prox``: ``E(z) = gamma * E_sheaf(z) + (rho/2) ||z - z_target||^2``.

The step size ``eta`` is meta-learned (set per ADMM iteration by the loop). Each
diffusion step ``z <- z - eta L_F z`` is a local message-passing operation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from flax import struct

from sheaf_admm.geometry.base import SheafGeometry
from sheaf_admm.solvers.base import ZSolverParams, rho_as_vec


@struct.dataclass
class GDParams(ZSolverParams):
    num_steps: int = struct.field(pytree_node=False, default=10)
    momentum: float = struct.field(pytree_node=False, default=0.9)
    nesterov: bool = struct.field(pytree_node=False, default=True)
    eta: jnp.ndarray = 0.01  # learned step size (dynamic; set per-iteration by the loop)


class GDZSolver:
    @staticmethod
    def solve(z_target, z_prev, geometry: SheafGeometry, params: GDParams, rho):
        gamma = jnp.asarray(params.gamma, dtype=z_target.dtype)
        rho_v = rho_as_vec(rho, z_target)

        if params.mode == "project":

            def energy(z):
                return geometry.energy(z)
        elif params.mode == "prox":

            def energy(z):
                return gamma * geometry.energy(z) + 0.5 * jnp.sum(rho_v * (z - z_target) ** 2)
        else:
            raise ValueError(f"unknown z mode {params.mode!r} (project|prox)")

        opt = optax.sgd(
            learning_rate=params.eta, momentum=params.momentum, nesterov=params.nesterov
        )
        z0 = z_target
        opt_state = opt.init(z0)

        def step(carry, _):
            z, state = carry
            grad = jax.grad(energy)(z)
            updates, state = opt.update(grad, state)
            return (optax.apply_updates(z, updates), state), None

        (z, _), _ = jax.lax.scan(step, (z0, opt_state), None, length=params.num_steps)
        return z
