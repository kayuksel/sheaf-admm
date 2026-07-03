"""Unrolled conjugate-gradient z-solver (the paper's default consensus step).

The z-update solves a PSD linear system in the sheaf Laplacian. We run a *fixed*
number ``T`` of CG iterations and let gradients flow through every step via
autodiff (explicit unroll, not implicit differentiation): implicit diff would
require CG to have converged to give correct gradients, whereas the unrolled
gradients capture the actual (under-solved) optimization trajectory the model is
trained with — essential for meta-learning the restriction maps.

Two modes:

* ``project`` (hard consensus, ``F z = 0``): solve ``(L + eps I) w = L z_target``
  and return ``z_target - w``. The tiny Tikhonov ``eps`` makes the singular ``L``
  invertible (its kernel is the consensus subspace).
* ``prox`` (soft consensus): solve ``(gamma L + rho I) z = rho z_target``.

CG runs batched: each batch element is an independent system, so the inner
products reduce over agents and stalk dims but keep the batch axis.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from sheaf_admm.geometry.base import SheafGeometry
from sheaf_admm.solvers.base import ZSolverParams, rho_as_vec

# CG denominator guard. 1e-8 (not 1e-12) because pTAp / rTr get small near
# convergence and 1e-12 sits below float32 roundoff there, risking 0/0 -> NaN
# in both the forward pass and the unrolled gradient. Distinct from the
# project-mode Tikhonov ``tikhonov_eps`` below.
_CG_DENOM_EPS = 1e-8


@struct.dataclass
class UnrolledCGParams(ZSolverParams):
    num_iters: int = struct.field(pytree_node=False, default=5)  # T
    tikhonov_eps: float = struct.field(pytree_node=False, default=1e-5)  # project-mode L+eps*I
    prox_init: str = struct.field(pytree_node=False, default="legacy")  # "legacy" | "warm"
    scan_unroll: int = struct.field(pytree_node=False, default=5)  # compile-time unroll, not T


def _batched_cg(matvec, b, x0, num_iters, scan_unroll):
    """Batched CG for ``A x = b`` over node states ``[N, B, d]``.

    Inner products reduce over agents (axis 0) and stalk dim (axis 2), keeping
    the batch axis -> per-batch scalars. ``num_iters`` is fixed (no early stop)
    for consistent gradients.
    """

    def bdot(a, c):
        return jnp.sum(a * c, axis=(0, 2))  # [B]

    def bscale(s, v):
        return s[None, :, None] * v  # [B] -> [1, B, 1]

    r0 = b - matvec(x0)
    p0 = r0
    rTr0 = bdot(r0, r0)

    def step(carry, _):
        x, r, p, rTr = carry
        Ap = matvec(p)
        pTAp = bdot(p, Ap)
        alpha = rTr / (pTAp + _CG_DENOM_EPS)
        x = x + bscale(alpha, p)
        r = r - bscale(alpha, Ap)
        rTr_new = bdot(r, r)
        beta = rTr_new / (rTr + _CG_DENOM_EPS)
        p = r + bscale(beta, p)
        return (x, r, p, rTr_new), None

    (x, *_), _ = jax.lax.scan(step, (x0, r0, p0, rTr0), None, length=num_iters, unroll=scan_unroll)
    return x


class UnrolledCGZSolver:
    @staticmethod
    def solve(z_target, z_prev, geometry: SheafGeometry, params: UnrolledCGParams, rho):
        if params.mode == "project":
            eps = jnp.asarray(params.tikhonov_eps, dtype=z_target.dtype)
            matvec = lambda x: geometry.laplacian_apply(x) + eps * x  # noqa: E731
            b = geometry.laplacian_apply(z_target)
            # Warm-start the correction at the ADMM target delta. This is the
            # paper-run inexact hard-consensus path, not a standalone exact
            # Euclidean projector: kernel components in the warm start are not
            # removed by the RHS. Use zero/range-space initialization if exact
            # projection idempotence is required.
            w0 = z_target - z_prev
            w = _batched_cg(matvec, b, w0, params.num_iters, params.scan_unroll)
            return z_target - w

        if params.mode == "prox":
            gamma = jnp.asarray(params.gamma, dtype=z_target.dtype)
            rho_v = rho_as_vec(rho, z_target)
            matvec = lambda x: gamma * geometry.laplacian_apply(x) + rho_v * x  # noqa: E731
            b = rho_v * z_target
            # Detach the warm-start so it does not unroll the whole ADMM chain.
            z0 = z_target if params.prox_init == "legacy" else jax.lax.stop_gradient(z_prev)
            return _batched_cg(matvec, b, z0, params.num_iters, params.scan_unroll)

        raise ValueError(f"unknown z mode {params.mode!r} (project|prox)")
