"""z-solvers: unrolled CG solves the prox normal equations to tolerance, project
mode reduces the consensus residual, and gradients w.r.t. restriction maps are
finite (the whole point of the unrolled — not implicit — solver)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from sheaf_admm.geometry import FixedGeometry
from sheaf_admm.solvers.z_solvers import GDParams, GDZSolver, UnrolledCGParams, UnrolledCGZSolver


def _geometry(key, N=6, E=9, d_e=3, d_v=4):
    k_e, k_rm = jax.random.split(key)
    u = jax.random.randint(k_e, (E,), 0, N)
    v = (u + 1 + jax.random.randint(k_e, (E,), 0, N - 1)) % N
    edges = jnp.stack([u, v], axis=1).astype(jnp.int32)
    rms = jax.random.normal(k_rm, (E, 2, d_e, d_v))
    return FixedGeometry(edge_indices=edges, restriction_maps=rms)


def test_unrolled_cg_prox_solves_normal_equation():
    """For a moderate ``T`` CG drives ``(gamma L + rho I) z = rho z_target`` to tol."""
    N, B, d_v = 6, 2, 4
    geom = _geometry(jax.random.PRNGKey(0), N=N, d_v=d_v)
    z_target = jax.random.normal(jax.random.PRNGKey(1), (N, B, d_v))
    gamma, rho = 5.0, jnp.asarray(0.5)

    # N*d_v = 24 dims per batch element -> CG converges well within 30 iters.
    params = UnrolledCGParams(mode="prox", gamma=gamma, num_iters=30)
    z = UnrolledCGZSolver.solve(z_target, z_target, geom, params, rho)

    resid = gamma * geom.laplacian_apply(z) + rho * z - rho * z_target
    rel = float(jnp.linalg.norm(resid) / jnp.linalg.norm(rho * z_target))
    assert rel < 1e-4, rel


def test_unrolled_cg_project_reduces_consensus_residual():
    """Project mode (``Fz = 0``) shrinks the disagreement vs the input target."""
    N, B, d_v = 6, 2, 4
    geom = _geometry(jax.random.PRNGKey(2), N=N, d_v=d_v)
    z_target = jax.random.normal(jax.random.PRNGKey(3), (N, B, d_v))
    params = UnrolledCGParams(mode="project", num_iters=30, tikhonov_eps=1e-5)
    z = UnrolledCGZSolver.solve(z_target, jnp.zeros_like(z_target), geom, params, jnp.asarray(1.0))

    before = float(jnp.sum(geom.edge_residuals(z_target) ** 2))
    after = float(jnp.sum(geom.edge_residuals(z) ** 2))
    assert after < 0.1 * before, (before, after)


def test_unrolled_cg_gradient_wrt_restriction_maps_finite():
    """Differentiating the unrolled solve through the restriction maps stays finite."""
    N, B, d_e, d_v = 6, 2, 3, 4
    geom = _geometry(jax.random.PRNGKey(4), N=N, d_e=d_e, d_v=d_v)
    z_target = jax.random.normal(jax.random.PRNGKey(5), (N, B, d_v))
    params = UnrolledCGParams(mode="prox", gamma=2.0, num_iters=5)

    def loss(rms):
        g = geom.replace(restriction_maps=rms)
        z = UnrolledCGZSolver.solve(z_target, z_target, g, params, jnp.asarray(0.3))
        return jnp.sum(z**2)

    grad = jax.grad(loss)(geom.restriction_maps)
    assert grad.shape == geom.restriction_maps.shape
    assert np.all(np.isfinite(np.asarray(grad)))
    assert float(jnp.sum(jnp.abs(grad))) > 0.0


def test_gd_prox_reduces_objective():
    """GD + Nesterov decreases the prox objective from the (z_target) start.

    The step size must respect the operator's spectrum: the prox gradient operator
    is ``gamma*L + rho*I``, so GD is stable only for ``eta < 2/lambda_max`` (the
    same bound that motivates the bounded-eta leash in the model config). We
    pick a safe ``eta = 1/lambda_max`` via power iteration rather than hard-coding.
    """
    N, B, d_v = 6, 2, 4
    geom = _geometry(jax.random.PRNGKey(6), N=N, d_v=d_v)
    z_target = jax.random.normal(jax.random.PRNGKey(7), (N, B, d_v))
    gamma, rho = 2.0, jnp.asarray(0.5)

    def obj(z):
        return float(gamma * geom.energy(z) + 0.5 * rho * jnp.sum((z - z_target) ** 2))

    op = lambda z: gamma * geom.laplacian_apply(z) + rho * z  # noqa: E731
    v = jax.random.normal(jax.random.PRNGKey(8), (N, B, d_v))
    for _ in range(40):
        v = op(v) / jnp.linalg.norm(op(v))
    lam_max = float(jnp.sum(v * op(v)) / jnp.sum(v * v))
    eta = 1.0 / lam_max

    params = GDParams(mode="prox", gamma=gamma, num_steps=80, eta=eta)
    z = GDZSolver.solve(z_target, z_target, geom, params, rho)
    assert np.all(np.isfinite(np.asarray(z)))
    assert obj(z) < obj(z_target)
