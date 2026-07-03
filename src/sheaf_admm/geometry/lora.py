"""LoRA geometry: input-modulated restriction maps ``F = R + (alpha/r) A B^T``.

The shared base map ``R`` is modulated per agent and per edge-direction by a
low-rank update produced by the encoder. To keep the Laplacian matvec free of
the expensive per-iteration multi-index gather, we precompute *edge-indexed*
factors once at construction time (``create_lora_geometry`` /
``create_sudoku_lora_geometry``): each edge already knows the A/B factors of its
two endpoints in the relevant slot/direction. The matvec is then pure einsums.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import struct


@struct.dataclass
class LoRAGeometry:
    """Sheaf geometry with per-agent LoRA-modulated restriction maps.

    Effective edge map ``F = R + (alpha/r) A B^T``. The ``*_edge`` tensors are
    the endpoints' A/B factors already gathered for each edge (see the
    ``create_*`` factories below), so ``laplacian_apply`` is gather-free.
    """

    edge_indices: jnp.ndarray  # [E, 2]
    restriction_maps: jnp.ndarray  # [E, 2, d_e, d_v] base maps R
    lora_alpha: float
    A_u_edge: jnp.ndarray  # [E, B, d_e, r]
    A_v_edge: jnp.ndarray  # [E, B, d_e, r]
    B_u_edge: jnp.ndarray  # [E, B, d_v, r]
    B_v_edge: jnp.ndarray  # [E, B, d_v, r]
    gate_u_edge: jnp.ndarray | None = None  # [E, B]
    gate_v_edge: jnp.ndarray | None = None  # [E, B]
    edge_mask: jnp.ndarray | None = None  # [E] float mask

    @property
    def _scale(self) -> float:
        return self.lora_alpha / self.A_u_edge.shape[-1]

    def _apply_endpoint(self, z_e, R, A_edge, B_edge, gate_edge):
        """Effective map applied to one endpoint: ``(R + scale A B^T) z_e``."""
        Rz = jnp.einsum("eij,ebj->ebi", R, z_e)  # [E, B, d_e]
        Btz = jnp.einsum("ebjr,ebj->ebr", B_edge, z_e)  # [E, B, r]
        ABtz = jnp.einsum("ebir,ebr->ebi", A_edge, Btz)  # [E, B, d_e]
        if gate_edge is not None:
            ABtz = ABtz * gate_edge[:, :, None]
        return Rz + self._scale * ABtz

    def _adjoint_endpoint(self, r, R, A_edge, B_edge, gate_edge):
        """Adjoint of ``_apply_endpoint``: ``(R + scale A B^T)^T r``."""
        contrib = jnp.einsum("eij,ebi->ebj", R, r)  # [E, B, d_v]
        Atr = jnp.einsum("ebir,ebi->ebr", A_edge, r)  # [E, B, r]
        if gate_edge is not None:
            Atr = Atr * gate_edge[:, :, None]
        lora = jnp.einsum("ebjr,ebr->ebj", B_edge, Atr)  # [E, B, d_v]
        return contrib + self._scale * lora

    def edge_residuals(self, z: jnp.ndarray) -> jnp.ndarray:
        u, v = self.edge_indices[:, 0], self.edge_indices[:, 1]
        Fz_u = self._apply_endpoint(
            z[u], self.restriction_maps[:, 0], self.A_u_edge, self.B_u_edge, self.gate_u_edge
        )
        Fz_v = self._apply_endpoint(
            z[v], self.restriction_maps[:, 1], self.A_v_edge, self.B_v_edge, self.gate_v_edge
        )
        r = Fz_u - Fz_v
        if self.edge_mask is not None:
            r = r * self.edge_mask[:, None, None]
        return r

    def energy(self, z: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * jnp.sum(self.edge_residuals(z) ** 2)

    def laplacian_apply(self, z: jnp.ndarray) -> jnp.ndarray:
        r = self.edge_residuals(z)  # [E, B, d_e]
        u, v = self.edge_indices[:, 0], self.edge_indices[:, 1]
        contrib_u = self._adjoint_endpoint(
            r, self.restriction_maps[:, 0], self.A_u_edge, self.B_u_edge, self.gate_u_edge
        )
        contrib_v = self._adjoint_endpoint(
            r, self.restriction_maps[:, 1], self.A_v_edge, self.B_v_edge, self.gate_v_edge
        )
        out = jnp.zeros_like(z)
        out = out.at[u].add(contrib_u)
        out = out.at[v].add(-contrib_v)
        return out

    def consistency_rms(self, z: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
        r = self.edge_residuals(z)
        return jnp.sqrt(jnp.mean(r**2, axis=(0, 2)) + eps)


def _gather_edge_factors(edge_indices, A, B, gate, sel_u, sel_v):
    """Gather endpoint LoRA factors into edge-indexed tensors.

    ``A``/``B``: ``[N, B, K, d_*, r]`` per-agent factors (K = directions or
    cell-slots). ``sel_u``/``sel_v``: ``[E]`` slot index per endpoint.
    """
    u, v = edge_indices[:, 0], edge_indices[:, 1]
    E = edge_indices.shape[0]
    e = jnp.arange(E)
    A_u_edge = A[u][e, :, sel_u]  # [E, B, d_e, r]
    A_v_edge = A[v][e, :, sel_v]
    B_u_edge = B[u][e, :, sel_u]
    B_v_edge = B[v][e, :, sel_v]
    gate_u_edge = gate_v_edge = None
    if gate is not None:
        gate_u_edge = gate[u][e, :, sel_u]  # [E, B]
        gate_v_edge = gate[v][e, :, sel_v]
    return A_u_edge, A_v_edge, B_u_edge, B_v_edge, gate_u_edge, gate_v_edge


def create_lora_geometry(
    edge_indices: jnp.ndarray,
    node_positions: jnp.ndarray,
    restriction_maps: jnp.ndarray,
    A: jnp.ndarray,  # [N, B, K, d_e, r]
    B: jnp.ndarray,  # [N, B, K, d_v, r]
    lora_alpha: float,
    num_directions: int,
    gate: jnp.ndarray | None = None,  # [N, B, K]
    edge_mask: jnp.ndarray | None = None,
) -> LoRAGeometry:
    """Directional (grid) LoRA geometry: select factors by edge direction."""
    from .restriction_maps import compute_direction_index

    u, v = edge_indices[:, 0], edge_indices[:, 1]
    dy = node_positions[v, 0] - node_positions[u, 0]
    dx = node_positions[v, 1] - node_positions[u, 1]
    dir_uv = compute_direction_index(dy, dx, num_directions)
    dir_vu = compute_direction_index(-dy, -dx, num_directions)
    A_u, A_v, B_u, B_v, g_u, g_v = _gather_edge_factors(edge_indices, A, B, gate, dir_uv, dir_vu)
    return LoRAGeometry(
        edge_indices=edge_indices,
        restriction_maps=restriction_maps,
        lora_alpha=lora_alpha,
        A_u_edge=A_u,
        A_v_edge=A_v,
        B_u_edge=B_u,
        B_v_edge=B_v,
        gate_u_edge=g_u,
        gate_v_edge=g_v,
        edge_mask=edge_mask,
    )


def create_sudoku_lora_geometry(
    edge_indices: jnp.ndarray,
    map_u: jnp.ndarray,  # [E] cell-slot 0-8
    map_v: jnp.ndarray,  # [E] cell-slot 0-8
    restriction_maps: jnp.ndarray,
    A: jnp.ndarray,  # [N, B, 9, d_e, r]
    B: jnp.ndarray,  # [N, B, 9, d_v, r]
    lora_alpha: float,
    gate: jnp.ndarray | None = None,
    edge_mask: jnp.ndarray | None = None,
) -> LoRAGeometry:
    """Sudoku LoRA geometry: select factors by shared-cell slot (replaces direction)."""
    A_u, A_v, B_u, B_v, g_u, g_v = _gather_edge_factors(edge_indices, A, B, gate, map_u, map_v)
    return LoRAGeometry(
        edge_indices=edge_indices,
        restriction_maps=restriction_maps,
        lora_alpha=lora_alpha,
        A_u_edge=A_u,
        A_v_edge=A_v,
        B_u_edge=B_u,
        B_v_edge=B_v,
        gate_u_edge=g_u,
        gate_v_edge=g_v,
        edge_mask=edge_mask,
    )
