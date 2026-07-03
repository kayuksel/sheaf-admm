"""Fixed geometry: learned but input-independent restriction maps."""

from __future__ import annotations

import jax.numpy as jnp
from flax import struct


@struct.dataclass
class FixedGeometry:
    """Sheaf geometry with restriction maps that do not depend on the input.

    The maps in ``restriction_maps`` are learned parameters (built once per
    forward pass from the shared base maps, see
    :mod:`sheaf_admm.geometry.restriction_maps`) but are the same for every
    batch element.
    """

    edge_indices: jnp.ndarray  # [E, 2]
    restriction_maps: jnp.ndarray  # [E, 2, d_e, d_v]
    edge_mask: jnp.ndarray | None = None  # [E] float mask (0=dropped, 1=kept)

    def edge_residuals(self, z: jnp.ndarray) -> jnp.ndarray:
        u, v = self.edge_indices[:, 0], self.edge_indices[:, 1]
        F_uv, F_vu = self.restriction_maps[:, 0], self.restriction_maps[:, 1]
        Fz_u = jnp.einsum("eij,ebj->ebi", F_uv, z[u])
        Fz_v = jnp.einsum("eij,ebj->ebi", F_vu, z[v])
        r = Fz_u - Fz_v
        if self.edge_mask is not None:
            r = r * self.edge_mask[:, None, None]
        return r

    def energy(self, z: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * jnp.sum(self.edge_residuals(z) ** 2)

    def laplacian_apply(self, z: jnp.ndarray) -> jnp.ndarray:
        r = self.edge_residuals(z)  # [E, B, d_e]
        u, v = self.edge_indices[:, 0], self.edge_indices[:, 1]
        F_uv, F_vu = self.restriction_maps[:, 0], self.restriction_maps[:, 1]
        # F^T r, scattered back to the two endpoints (note the orientation sign on v).
        contrib_u = jnp.einsum("eij,ebi->ebj", F_uv, r)
        contrib_v = jnp.einsum("eij,ebi->ebj", F_vu, r)
        out = jnp.zeros_like(z)
        out = out.at[u].add(contrib_u)
        out = out.at[v].add(-contrib_v)
        return out

    def consistency_rms(self, z: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
        r = self.edge_residuals(z)
        return jnp.sqrt(jnp.mean(r**2, axis=(0, 2)) + eps)
