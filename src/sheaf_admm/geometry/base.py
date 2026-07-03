"""The sheaf-geometry interface.

A *geometry* bundles the cellular-sheaf structure used by the z-update: the
agent graph (``edge_indices``), the restriction maps ``F_{i->e}`` (possibly
input-modulated by LoRA), and the operators the consensus solvers need.

The two operators that matter:

* ``laplacian_apply`` — the sheaf Laplacian ``L_F = F^T F`` as a matrix-free,
  block-sparse, per-edge matvec. Conjugate-gradient z-solvers call this.
* ``energy`` — the sheaf energy ``E(z) = 1/2 sum_e ||F_{u->e} z_u - F_{v->e} z_v||^2``.
  Gradient-descent z-solvers differentiate this.

Everything operates on batched node states ``z`` of shape ``[N, B, d_v]`` (N
agents, B batch, d_v vertex-stalk dim). Each batch element is an independent
linear system; the Laplacian never mixes batch elements.
"""

from __future__ import annotations

from typing import Protocol

import jax.numpy as jnp


class SheafGeometry(Protocol):
    """Interface for sheaf geometries (see module docstring).

    Implementations: :class:`~sheaf_admm.geometry.fixed.FixedGeometry` (learned
    but input-independent restriction maps) and
    :class:`~sheaf_admm.geometry.lora.LoRAGeometry` (input-modulated maps
    ``F = R + (alpha/r) A B^T``).
    """

    edge_indices: jnp.ndarray  # [E, 2] (u, v) node indices per edge
    restriction_maps: jnp.ndarray  # [E, 2, d_e, d_v]; [:,0]=F_{u->e}, [:,1]=F_{v->e}

    def edge_residuals(self, z: jnp.ndarray) -> jnp.ndarray:
        """Coboundary ``F z``: per-edge disagreement ``F_{u->e} z_u - F_{v->e} z_v``.

        ``z``: ``[N, B, d_v]`` -> ``r``: ``[E, B, d_e]``.
        """
        ...

    def laplacian_apply(self, z: jnp.ndarray) -> jnp.ndarray:
        """Sheaf Laplacian ``L_F z = F^T F z``. ``[N, B, d_v]`` -> ``[N, B, d_v]``."""
        ...

    def energy(self, z: jnp.ndarray) -> jnp.ndarray:
        """Scalar sheaf energy ``1/2 sum_e ||F_{u->e} z_u - F_{v->e} z_v||^2``."""
        ...

    def consistency_rms(self, z: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
        """Per-batch RMS disagreement ``sqrt(mean_{e,d_e} r^2 + eps)`` -> ``[B]``.

        The ``+eps`` under the sqrt keeps the gradient finite at perfect
        consensus (``r -> 0``), which is exactly the training target.
        """
        ...
