"""Restriction-map construction: parameterization, initialization, and sharing.

Restriction maps ``F_{i->e} in R^{d_e x d_v}`` project agent states into a
shared edge stalk. Rather than learn one map per edge, we learn a small set of
*base* maps and share them by edge type:

* ``directional`` — one map per grid direction (N/E/S/W, or 8-way). Maze/MNIST.
* ``shared`` — a single global map used on every edge. MNIST (global sharing).
* ``sudoku`` — 9 maps, one per shared-cell slot; with ``soft_slice`` init each
  is a selector onto one of 9 disjoint ``d_e``-blocks of the ``d_v``-stalk.

These nn.Modules produce the base maps; the ``build_*`` functions gather them
into the per-edge ``[E, 2, d_e, d_v]`` layout the geometries consume.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn

Initializer = Callable[[jax.Array, tuple[int, ...], jnp.dtype], jnp.ndarray]


def get_direction_names(num_directions: int) -> tuple[str, ...]:
    """Ordered direction names for 4-way or 8-way grid connectivity."""
    if num_directions == 4:
        return ("N", "E", "S", "W")
    if num_directions == 8:
        return ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    raise ValueError("num_directions must be 4 or 8")


def normalize_rm_sharing(rm_sharing: str) -> str:
    """Canonicalize the ``rm_sharing`` string (``nesw`` is a legacy alias)."""
    rm_sharing = rm_sharing.lower()
    return "directional" if rm_sharing == "nesw" else rm_sharing


def compute_direction_index(
    dy: jnp.ndarray, dx: jnp.ndarray, num_directions: int = 4
) -> jnp.ndarray:
    """Map a position delta ``(dy, dx)`` to a direction index (JAX-traceable).

    4-way: 0=N, 1=E, 2=S, 3=W (vertical takes priority).
    8-way: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW.
    """
    if num_directions not in (4, 8):
        raise ValueError("num_directions must be 4 or 8")
    is_north, is_south = dy < 0, dy > 0
    is_east, is_west = dx > 0, dx < 0
    if num_directions == 4:
        return jnp.where(is_north, 0, jnp.where(is_south, 2, jnp.where(is_east, 1, 3)))
    return jnp.where(
        is_north & is_east,
        1,
        jnp.where(
            is_north & is_west,
            7,
            jnp.where(
                is_north,
                0,
                jnp.where(
                    is_south & is_east,
                    3,
                    jnp.where(
                        is_south & is_west, 5, jnp.where(is_south, 4, jnp.where(is_east, 2, 6))
                    ),
                ),
            ),
        ),
    )


def make_rm_initializer(init_method: str, d_e: int, d_v: int) -> Initializer:
    """Initializer for restriction maps of shape ``(d_e, d_v)`` or ``(E, 2, d_e, d_v)``.

    ``orthonormal`` (default, paper) gives each map orthonormal rows; ``default``
    is Glorot-normal; ``identity`` broadcasts ``eye(d_e, d_v)``.
    """
    if init_method == "orthonormal":
        base_init = nn.initializers.orthogonal()
    elif init_method == "default":
        base_init = nn.initializers.glorot_normal()
    elif init_method == "identity":

        def init(key, shape, dtype=jnp.float32):
            if shape[-2:] != (d_e, d_v):
                raise ValueError(f"identity init expects trailing {(d_e, d_v)}, got {shape}")
            return jnp.broadcast_to(jnp.eye(d_e, d_v, dtype=dtype), shape)

        return init
    else:
        raise ValueError(f"unknown rm_init={init_method!r} (orthonormal|default|identity)")

    def init(key, shape, dtype=jnp.float32):
        if shape[-2:] != (d_e, d_v):
            raise ValueError(f"rm init expects trailing {(d_e, d_v)}, got {shape}")
        if len(shape) == 2:
            return base_init(key, shape, dtype)
        if len(shape) != 4:
            raise ValueError(f"rm init expects (d_e,d_v) or (E,2,d_e,d_v), got {shape}")
        keys = jax.random.split(key, shape[0] * shape[1])
        mats = jax.vmap(lambda k: base_init(k, (d_e, d_v), dtype))(keys)
        return mats.reshape(shape)

    return init


def make_sudoku_rm_initializer(init_method: str, d_e: int, d_v: int) -> Initializer:
    """Initializer for the 9 Sudoku base maps, shape ``(9, d_e, d_v)``.

    ``soft_slice`` (paper) seeds each map ``k`` as an identity selector onto the
    k-th ``d_e``-block of the ``d_v``-stalk plus small noise — the inductive
    bias that map ``k`` extracts cell ``k`` from an agent's 9-cell view. Requires
    ``9 * d_e <= d_v`` (e.g. 9*32 = 288).
    """
    if init_method == "soft_slice":
        if 9 * d_e > d_v:
            raise ValueError(
                f"soft_slice needs 9*d_e <= d_v (got 9*{d_e}={9 * d_e} > {d_v}); "
                "the 9 selector blocks would not fit in the vertex stalk."
            )

        def init(key, shape, dtype=jnp.float32):
            mats = nn.initializers.normal(stddev=0.01)(key, shape, dtype)
            eye = jnp.eye(d_e, dtype=dtype)
            for k in range(9):
                mats = mats.at[k, :, k * d_e : (k + 1) * d_e].add(eye)
            return mats

        return init

    if init_method == "orthonormal":
        base_init = nn.initializers.orthogonal()
    elif init_method == "default":
        base_init = nn.initializers.glorot_normal()
    elif init_method == "identity":

        def init(key, shape, dtype=jnp.float32):
            return jnp.broadcast_to(jnp.eye(d_e, d_v, dtype=dtype), shape)

        return init
    else:
        raise ValueError(f"unknown sudoku rm_init={init_method!r}")

    def init(key, shape, dtype=jnp.float32):
        keys = jax.random.split(key, 9)
        return jax.vmap(lambda k: base_init(k, (d_e, d_v), dtype))(keys)

    return init


class DirectionalRestrictionMaps(nn.Module):
    """Learn one ``[d_e, d_v]`` base map per grid direction (4- or 8-way)."""

    stalk_dim: int  # d_v
    edge_stalk_dim: int | None = None  # d_e (defaults to d_v)
    init_method: str = "orthonormal"
    num_directions: int = 4

    @nn.compact
    def __call__(self) -> dict[str, jnp.ndarray]:
        d_v = self.stalk_dim
        d_e = self.edge_stalk_dim or self.stalk_dim
        init = make_rm_initializer(self.init_method, d_e, d_v)
        return {
            name: self.param(f"R_{name}", init, (d_e, d_v))
            for name in get_direction_names(self.num_directions)
        }


class SharedRestrictionMap(nn.Module):
    """Learn a single ``[d_e, d_v]`` base map shared across all edges (global sharing)."""

    stalk_dim: int  # d_v
    edge_stalk_dim: int | None = None  # d_e
    init_method: str = "orthonormal"

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        d_v = self.stalk_dim
        d_e = self.edge_stalk_dim or self.stalk_dim
        init = make_rm_initializer(self.init_method, d_e, d_v)
        return self.param("R_shared", init, (d_e, d_v))


class SudokuRestrictionMaps(nn.Module):
    """Learn the 9 Sudoku base maps ``R_indices [9, d_e, d_v]`` (one per cell-slot)."""

    stalk_dim: int  # d_v
    edge_stalk_dim: int  # d_e
    init_method: str = "soft_slice"

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        init = make_sudoku_rm_initializer(self.init_method, self.edge_stalk_dim, self.stalk_dim)
        return self.param("R_indices", init, (9, self.edge_stalk_dim, self.stalk_dim))


def build_directional_restriction_maps(
    R_dict: dict[str, jnp.ndarray],
    edge_indices: jnp.ndarray,
    node_positions: jnp.ndarray,
    num_directions: int = 4,
) -> jnp.ndarray:
    """Gather directional base maps into ``[E, 2, d_e, d_v]`` by edge direction."""
    R_stack = jnp.stack([R_dict[n] for n in get_direction_names(num_directions)])  # [K,d_e,d_v]
    u, v = edge_indices[:, 0], edge_indices[:, 1]
    dy = node_positions[v, 0] - node_positions[u, 0]
    dx = node_positions[v, 1] - node_positions[u, 1]
    dir_uv = compute_direction_index(dy, dx, num_directions)
    dir_vu = compute_direction_index(-dy, -dx, num_directions)
    return jnp.stack([R_stack[dir_uv], R_stack[dir_vu]], axis=1)


def build_shared_restriction_maps(R_shared: jnp.ndarray, num_edges: int) -> jnp.ndarray:
    """Broadcast one base map onto all edges/endpoints -> ``[E, 2, d_e, d_v]``."""
    return jnp.broadcast_to(R_shared[None, None], (num_edges, 2, *R_shared.shape))


def build_sudoku_restriction_maps(
    R_stack: jnp.ndarray, map_u: jnp.ndarray, map_v: jnp.ndarray
) -> jnp.ndarray:
    """Gather the 9 Sudoku base maps per edge by shared-cell slot -> ``[E, 2, d_e, d_v]``."""
    return jnp.stack([R_stack[map_u], R_stack[map_v]], axis=1)
