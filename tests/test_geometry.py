"""Sheaf geometry: the Laplacian matches a dense build, energy is the quadratic
form, LoRA with ``B=0`` reduces to the fixed geometry, and the consistency RMS is
finite at perfect consensus."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sheaf_admm.geometry import FixedGeometry, create_lora_geometry


def _random_geometry(key, N=5, E=7, d_e=3, d_v=4):
    """A small fixed geometry with random edges and restriction maps."""
    k_e, k_rm = jax.random.split(key)
    # Random (u, v) pairs with u != v.
    u = jax.random.randint(k_e, (E,), 0, N)
    v = (u + 1 + jax.random.randint(k_e, (E,), 0, N - 1)) % N
    edges = jnp.stack([u, v], axis=1).astype(jnp.int32)
    rms = jax.random.normal(k_rm, (E, 2, d_e, d_v))
    return FixedGeometry(edge_indices=edges, restriction_maps=rms), edges, rms


def _dense_coboundary(edges, rms, N, d_e, d_v):
    """Dense coboundary ``F`` of shape ``[E*d_e, N*d_v]`` so ``F z = edge residuals``.

    Block row ``e`` is ``+F_{u->e}`` in column block ``u`` and ``-F_{v->e}`` in
    column block ``v`` (the orientation the geometry's residual uses).
    """
    E = edges.shape[0]
    F = np.zeros((E * d_e, N * d_v), dtype=np.float64)
    rms = np.asarray(rms, dtype=np.float64)
    edges = np.asarray(edges)
    for e in range(E):
        uu, vv = int(edges[e, 0]), int(edges[e, 1])
        F[e * d_e : (e + 1) * d_e, uu * d_v : (uu + 1) * d_v] += rms[e, 0]
        F[e * d_e : (e + 1) * d_e, vv * d_v : (vv + 1) * d_v] -= rms[e, 1]
    return F


def test_laplacian_matches_dense_FtF():
    N, E, d_e, d_v, B = 5, 7, 3, 4, 2
    geom, edges, rms = _random_geometry(jax.random.PRNGKey(0), N, E, d_e, d_v)
    F = _dense_coboundary(edges, rms, N, d_e, d_v)
    L = F.T @ F  # [N*d_v, N*d_v]

    z = jax.random.normal(jax.random.PRNGKey(1), (N, B, d_v))
    got = np.asarray(geom.laplacian_apply(z))  # [N, B, d_v]

    # Apply the dense Laplacian per batch element.
    z_np = np.asarray(z, dtype=np.float64)
    want = np.empty_like(z_np)
    for b in range(B):
        want[:, b, :] = (L @ z_np[:, b, :].reshape(-1)).reshape(N, d_v)
    np.testing.assert_allclose(got, want, atol=1e-4, rtol=1e-4)


def test_energy_is_half_zT_L_z():
    N, E, d_e, d_v = 5, 7, 3, 4
    geom, edges, rms = _random_geometry(jax.random.PRNGKey(2), N, E, d_e, d_v)
    F = _dense_coboundary(edges, rms, N, d_e, d_v)
    L = F.T @ F

    z = jax.random.normal(jax.random.PRNGKey(3), (N, 1, d_v))
    zf = np.asarray(z, dtype=np.float64)[:, 0, :].reshape(-1)
    want = 0.5 * float(zf @ L @ zf)
    got = float(geom.energy(z))
    assert want == pytest.approx(got, abs=1e-4, rel=1e-4)


def test_energy_equals_half_residual_sq():
    """``energy = 1/2 sum ||F_{u}z_u - F_{v}z_v||^2`` directly from the residuals."""
    geom, *_ = _random_geometry(jax.random.PRNGKey(4))
    z = jax.random.normal(jax.random.PRNGKey(5), (5, 3, 4))
    r = geom.edge_residuals(z)
    assert float(geom.energy(z)) == pytest.approx(0.5 * float(jnp.sum(r**2)), rel=1e-5)


def test_lora_with_zero_B_equals_fixed():
    """``F = R + (alpha/r) A B^T``; with ``B = 0`` the LoRA geometry is the fixed one."""
    N, E, d_e, d_v, B, rank = 5, 7, 3, 4, 2, 2
    geom_fixed, edges, rms = _random_geometry(jax.random.PRNGKey(6), N, E, d_e, d_v)
    node_pos = jnp.asarray(np.stack([np.arange(N), np.zeros(N)], axis=1), dtype=jnp.float32)

    A = jax.random.normal(jax.random.PRNGKey(7), (N, B, 4, d_e, rank))  # K=4 directions
    B_factors = jnp.zeros((N, B, 4, d_v, rank))
    geom_lora = create_lora_geometry(
        edges, node_pos, rms, A, B_factors, lora_alpha=1.0, num_directions=4
    )

    z = jax.random.normal(jax.random.PRNGKey(8), (N, B, d_v))
    np.testing.assert_allclose(
        np.asarray(geom_lora.laplacian_apply(z)),
        np.asarray(geom_fixed.laplacian_apply(z)),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(geom_lora.edge_residuals(z)),
        np.asarray(geom_fixed.edge_residuals(z)),
        atol=1e-5,
        rtol=1e-5,
    )


def test_consistency_rms_finite_at_zero_residual():
    """At perfect consensus (``z = 0`` -> ``Fz = 0``) the RMS is finite (``sqrt(eps)``)."""
    geom, *_ = _random_geometry(jax.random.PRNGKey(9))
    z = jnp.zeros((5, 3, 4))
    rms = geom.consistency_rms(z)
    assert rms.shape == (3,)
    assert np.all(np.isfinite(np.asarray(rms)))
    np.testing.assert_allclose(np.asarray(rms), np.sqrt(1e-6), atol=1e-7)

    # And its gradient is finite at the zero-residual target.
    grad = jax.grad(lambda zz: jnp.sum(geom.consistency_rms(zz)))(z)
    assert np.all(np.isfinite(np.asarray(grad)))
