"""x-solvers: the diagonal prox closed form and its special cases (quadratic,
non-negative, box clip), the simple tether, and the dense-quadratic linear solve."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from sheaf_admm.solvers.x_solvers.dense_quadratic import (
    DenseQuadraticParams,
    DenseQuadraticXSolver,
)
from sheaf_admm.solvers.x_solvers.diagonal_prox import DiagonalProxParams, DiagonalProxXSolver
from sheaf_admm.solvers.x_solvers.simple import SimpleParams, SimpleXSolver


def test_diagonal_prox_quadratic_case():
    """``lambda=0``, no box -> ``x = (rho (z-y) - q) / (D + rho)``."""
    N, B, d = 3, 2, 4
    key = jax.random.PRNGKey(0)
    z, y, q = (jax.random.normal(k, (N, B, d)) for k in jax.random.split(key, 3))
    diag = jax.nn.softplus(jax.random.normal(jax.random.PRNGKey(1), (N, B, d))) + 1e-4
    rho = jnp.asarray(0.3)

    enc = {"q_diag": diag, "q": q}
    x = DiagonalProxXSolver.solve(z, y, rho, enc, DiagonalProxParams())
    want = (rho * (z - y) - q) / (diag + rho)
    np.testing.assert_allclose(np.asarray(x), np.asarray(want), atol=1e-6)


def test_diagonal_prox_quadratic_solves_normal_equation():
    """The quadratic prox minimizes ``1/2 D x^2 + q x + rho/2 ||x - v||^2``:
    gradient ``D x + q + rho (x - v) = 0`` at the solution."""
    N, B, d = 3, 2, 4
    z, y, q = (jax.random.normal(k, (N, B, d)) for k in jax.random.split(jax.random.PRNGKey(2), 3))
    diag = jax.nn.softplus(jax.random.normal(jax.random.PRNGKey(3), (N, B, d))) + 1e-4
    rho = jnp.asarray(0.7)
    x = DiagonalProxXSolver.solve(z, y, rho, {"q_diag": diag, "q": q}, DiagonalProxParams())
    resid = diag * x + q + rho * (x - (z - y))
    np.testing.assert_allclose(np.asarray(resid), 0.0, atol=1e-5)


def test_diagonal_prox_non_negative():
    """``lower=0`` clips negatives to 0 (the Sudoku non-negative prox)."""
    N, B, d = 2, 2, 3
    z = jnp.full((N, B, d), -5.0)  # drive the unconstrained solution negative
    y = jnp.zeros_like(z)
    q = jnp.zeros_like(z)
    diag = jnp.ones_like(z)
    enc = {"q_diag": diag, "q": q, "lower": 0.0}
    x = DiagonalProxXSolver.solve(z, y, jnp.asarray(1.0), enc, DiagonalProxParams())
    assert np.all(np.asarray(x) >= 0.0)
    np.testing.assert_allclose(np.asarray(x), 0.0, atol=1e-7)


def test_diagonal_prox_box_clip():
    """Per-dim ``lower``/``upper`` clip the solution into the box."""
    z = jnp.asarray([[[100.0, -100.0, 0.5]], [[100.0, -100.0, 0.5]]])
    y = jnp.zeros_like(z)
    q = jnp.zeros_like(z)
    diag = jnp.ones_like(z)
    upper = jnp.full_like(z, 1.0)
    enc = {"q_diag": diag, "q": q, "lower": 0.0, "upper": upper}
    x = DiagonalProxXSolver.solve(z, y, jnp.asarray(1.0), enc, DiagonalProxParams())
    assert np.all(np.asarray(x) <= 1.0 + 1e-6)
    assert np.all(np.asarray(x) >= -1e-6)
    # entry 0 -> upper (1), entry 1 -> lower (0), entry 2 -> interior (0.25).
    np.testing.assert_allclose(np.asarray(x)[0, 0], [1.0, 0.0, 0.25], atol=1e-6)


def test_diagonal_prox_l1_soft_threshold():
    """Scalar L1 applies the soft-threshold ``sign(t) max(|t| - lambda/a, 0)``."""
    z = jnp.asarray([[[1.0, -1.0, 0.05, -0.05]]])
    y = jnp.zeros_like(z)
    q = jnp.zeros_like(z)
    diag = jnp.zeros_like(z)  # a = rho here (with l2=0)
    rho = jnp.asarray(1.0)
    enc = {"q_diag": diag, "q": q, "l1_weight": jnp.asarray(0.1)}
    x = DiagonalProxXSolver.solve(z, y, rho, enc, DiagonalProxParams())
    want = np.sign(np.asarray(z)) * np.maximum(np.abs(np.asarray(z)) - 0.1, 0.0)
    np.testing.assert_allclose(np.asarray(x), want, atol=1e-6)


def test_simple_closed_form():
    """``x = (beta h + rho (z - y)) / (beta + rho)``."""
    N, B, d = 3, 2, 4
    z, y, h = (jax.random.normal(k, (N, B, d)) for k in jax.random.split(jax.random.PRNGKey(4), 3))
    beta = jnp.asarray(2.0)
    rho = jnp.asarray(0.5)
    x = SimpleXSolver.solve(z, y, rho, {"h": h, "beta": beta}, SimpleParams())
    want = (beta * h + rho * (z - y)) / (beta + rho)
    np.testing.assert_allclose(np.asarray(x), np.asarray(want), atol=1e-6)


def test_dense_quadratic_solves_system():
    """``(Q + rho I) x = rho (z - y) - q``."""
    N, B, d = 2, 2, 4
    key = jax.random.PRNGKey(5)
    M = jax.random.normal(key, (N, B, d, d))
    Q = jnp.einsum("nbij,nbkj->nbik", M, M) + 0.1 * jnp.eye(d)  # SPD
    z, y, q = (jax.random.normal(k, (N, B, d)) for k in jax.random.split(jax.random.PRNGKey(6), 3))
    rho = jnp.asarray(0.4)
    x = DenseQuadraticXSolver.solve(z, y, rho, {"Q": Q, "q": q}, DenseQuadraticParams())
    lhs = jnp.einsum("nbij,nbj->nbi", Q + rho * jnp.eye(d), x)
    rhs = rho * (z - y) - q
    np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs), atol=1e-4)
