"""ADMM core loop: shapes are correct, truncated BPTT (``grad_window``) is
forward-identical to the full unroll (``stop_gradient`` is identity on the value),
and gradients flow finitely through the windowed trajectory."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from sheaf_admm.admm import ADMMState, run_admm
from sheaf_admm.geometry import FixedGeometry
from sheaf_admm.solvers.x_solvers.diagonal_prox import DiagonalProxParams, DiagonalProxXSolver
from sheaf_admm.solvers.z_solvers import UnrolledCGParams, UnrolledCGZSolver


def _setup(key, N=6, B=2, d_v=4, d_e=3, E=9):
    k_e, k_rm, k_q, k_lin, k_h = jax.random.split(key, 5)
    u = jax.random.randint(k_e, (E,), 0, N)
    v = (u + 1 + jax.random.randint(k_e, (E,), 0, N - 1)) % N
    edges = jnp.stack([u, v], axis=1).astype(jnp.int32)
    rms = jax.random.normal(k_rm, (E, 2, d_e, d_v))
    geom = FixedGeometry(edge_indices=edges, restriction_maps=rms)
    enc = {
        "q_diag": jax.nn.softplus(jax.random.normal(k_q, (N, B, d_v))) + 1e-4,
        "q": jax.random.normal(k_lin, (N, B, d_v)),
    }
    z_init = jax.random.normal(k_h, (N, B, d_v))
    return geom, enc, z_init


def _run(enc, geom, z_init, *, num_iters, loss_window, grad_window):
    return run_admm(
        enc,
        geom,
        DiagonalProxXSolver,
        DiagonalProxParams(),
        UnrolledCGZSolver,
        UnrolledCGParams(mode="prox", gamma=2.0, num_iters=5),
        rho=jnp.asarray(0.3),
        z_init=z_init,
        num_iters=num_iters,
        loss_window=loss_window,
        grad_window=grad_window,
    )


def test_run_admm_shapes():
    N, B, d_v, W = 6, 2, 4, 3
    geom, enc, z_init = _setup(jax.random.PRNGKey(0), N=N, B=B, d_v=d_v)
    state, x_window = _run(enc, geom, z_init, num_iters=10, loss_window=W, grad_window=None)
    assert isinstance(state, ADMMState)
    for arr in (state.x, state.z, state.y):
        assert arr.shape == (N, B, d_v)
    assert x_window.shape == (W, N, B, d_v)
    # The last windowed x is the final state's x.
    np.testing.assert_allclose(np.asarray(x_window[-1]), np.asarray(state.x), atol=1e-6)


def test_truncated_bptt_forward_matches_full_unroll():
    """The forward value is independent of ``grad_window`` (only gradients differ)."""
    geom, enc, z_init = _setup(jax.random.PRNGKey(1))
    K, W = 12, 3
    full_state, full_x = _run(enc, geom, z_init, num_iters=K, loss_window=W, grad_window=None)
    trunc_state, trunc_x = _run(enc, geom, z_init, num_iters=K, loss_window=W, grad_window=4)
    np.testing.assert_allclose(np.asarray(full_x), np.asarray(trunc_x), atol=1e-5)
    np.testing.assert_allclose(np.asarray(full_state.z), np.asarray(trunc_state.z), atol=1e-5)
    np.testing.assert_allclose(np.asarray(full_state.y), np.asarray(trunc_state.y), atol=1e-5)


def test_gradients_finite_full_and_truncated():
    geom, enc, z_init = _setup(jax.random.PRNGKey(2))

    def make_loss(grad_window):
        def loss(rms):
            g = geom.replace(restriction_maps=rms)
            _, x_window = _run(enc, g, z_init, num_iters=12, loss_window=3, grad_window=grad_window)
            return jnp.mean(x_window**2)

        return loss

    g_full = jax.grad(make_loss(None))(geom.restriction_maps)
    g_trunc = jax.grad(make_loss(4))(geom.restriction_maps)
    for g in (g_full, g_trunc):
        assert g.shape == geom.restriction_maps.shape
        assert np.all(np.isfinite(np.asarray(g)))
    # Truncating BPTT changes the gradient (detaches the early steps).
    assert not np.allclose(np.asarray(g_full), np.asarray(g_trunc), atol=1e-6)


def test_loss_window_clamped_to_iters():
    """A loss window larger than the (graded) horizon collapses to the available steps."""
    geom, enc, z_init = _setup(jax.random.PRNGKey(3))
    _, x_window = _run(enc, geom, z_init, num_iters=2, loss_window=5, grad_window=None)
    assert x_window.shape[0] == 2
