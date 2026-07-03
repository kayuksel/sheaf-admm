"""Views: the map between a global grid and the per-agent local representation.

Two task families share this module:

* **Grid tasks (maze / MNIST).** Agents sit on a regular grid of centers
  (:func:`grid_agent_centers`), each owning a square ``patch_size`` patch
  (:func:`patchify_batch_jax`). The agent graph is the grid adjacency
  (:func:`build_grid_edge_indices`). After ADMM the per-agent logit patches are
  stitched back to a global grid by averaging overlaps
  (:func:`reassemble_logits`). Maze borders are handled by a wall-token pre-pad
  *before* patch extraction (see :func:`prepare_maze_patches`), so boundary
  agents see walls rather than zeros.

* **Sudoku.** The 81 cells are covered by 27 constraint agents — 9 rows, 9
  columns, 9 boxes — each a 9-cell view (:func:`sudoku_slice_batch_jax`). The
  agent graph is the constraint multigraph (:func:`build_sudoku_multigraph`):
  each cell is a 3-clique over its (row, col, box) agents, with per-edge
  shared-cell slots. Reassembly averages the three views that cover each cell
  (:func:`reassemble_sudoku_logits`).

Tensor conventions: images are ``[B, H, W, C]``, agent patches are
``[N, B, ph, pw, C]``, node states are ``[N, B, d_v]``, edges are ``[E, 2]``.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

# =============================================================================
# Grid agents: centers, patches, edges
# =============================================================================


def grid_agent_centers(image_hw: Sequence[int], stride: int, patch_size: int) -> np.ndarray:
    """Place agent centers on a regular grid.

    The first center sits at ``patch_size // 2`` along each axis and centers
    step by ``stride`` thereafter (``range(patch_size // 2, dim, stride)``).
    Returns ``[N, 2]`` integer ``(y, x)`` coordinates in row-major order.
    """
    h, w = int(image_hw[0]), int(image_hw[1])
    if stride <= 0:
        raise ValueError("stride must be positive")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")

    center = patch_size // 2
    coords = [(y, x) for y in range(center, h, stride) for x in range(center, w, stride)]
    return np.asarray(coords, dtype=np.int64)


def patchify_batch_jax(
    batch_images: jnp.ndarray, centers_yx: jnp.ndarray, patch_size: int
) -> jnp.ndarray:
    """Extract one ``patch_size`` patch per agent from each image.

    ``batch_images``: ``[B, H, W, C]``; ``centers_yx``: ``[N, 2]`` integer
    centers. The image is zero-padded by ``patch_size // 2`` on each spatial
    side so every center yields a full patch. Returns ``[N, B, ps, ps, C]``.
    """
    images = jnp.asarray(batch_images)
    centers = jnp.asarray(centers_yx, dtype=jnp.int32)
    if images.ndim != 4:
        raise ValueError(f"expected images of shape (B,H,W,C); got {images.shape}")
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("centers_yx must have shape (N, 2)")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")

    pad = patch_size // 2
    padded = jnp.pad(
        images,
        ((0, 0), (pad, pad), (pad, pad), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    batch_size, channels = images.shape[0], images.shape[3]

    def slice_center(center: jnp.ndarray) -> jnp.ndarray:
        cy, cx = center[0], center[1]
        # The pre-pad shifts the in-bounds origin to (cy, cx) on the padded grid.
        return jax.lax.dynamic_slice(
            padded, (0, cy, cx, 0), (batch_size, patch_size, patch_size, channels)
        )

    return jax.vmap(slice_center)(centers)


def build_grid_edge_indices(centers: np.ndarray, stride: int, connectivity: int = 4) -> np.ndarray:
    """Undirected grid edges between adjacent agent centers, each stored once.

    Orientation is fixed so every pair appears a single time:

    * 4-way: right, down.
    * 8-way: right, down, down-right, down-left.

    Returns ``[E, 2]`` int32 ``(u, v)`` indices into ``centers``.
    """
    if stride <= 0:
        raise ValueError("stride must be positive")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    if centers.size == 0:
        return np.zeros((0, 2), dtype=np.int32)

    center_to_idx = {tuple(map(int, c)): i for i, c in enumerate(centers)}
    offsets = [(0, stride), (stride, 0)]
    if connectivity == 8:
        offsets += [(stride, stride), (stride, -stride)]

    edges = []
    for i, (cy, cx) in enumerate(centers):
        for dy, dx in offsets:
            j = center_to_idx.get((int(cy + dy), int(cx + dx)))
            if j is not None:
                edges.append([i, j])

    return np.asarray(edges, dtype=np.int32) if edges else np.zeros((0, 2), dtype=np.int32)


def node_positions(centers: np.ndarray) -> jnp.ndarray:
    """Float32 ``[N, 2]`` node positions for direction-indexed restriction maps."""
    return jnp.asarray(centers, dtype=jnp.float32)


# =============================================================================
# Grid reassembly: stitch agent logit patches back to a global grid
# =============================================================================


def _overlap_slices(
    center_yx: Sequence[int], patch_size: int, image_hw: tuple[int, int]
) -> tuple[slice, slice, slice, slice]:
    """In-bounds image slices and the matching local-patch slices for one agent."""
    h, w = image_hw
    pad = patch_size // 2
    cy, cx = int(center_yx[0]), int(center_yx[1])
    y0, x0 = cy - pad, cx - pad

    y_start, x_start = max(y0, 0), max(x0, 0)
    y_end, x_end = min(y0 + patch_size, h), min(x0 + patch_size, w)

    py_start = y_start - y0
    px_start = x_start - x0
    return (
        slice(y_start, y_end),
        slice(x_start, x_end),
        slice(py_start, py_start + (y_end - y_start)),
        slice(px_start, px_start + (x_end - x_start)),
    )


def reassemble_logits(
    patches: np.ndarray,
    centers_yx: np.ndarray,
    image_hw: Sequence[int],
    num_classes: int,
    mode: str = "mean",
) -> np.ndarray:
    """Stitch per-agent logit patches ``[N, B, ph, pw, C]`` into a global grid.

    * ``"mean"``: average overlapping agent logits — ``logits_sum / max(counts, 1)``.
    * ``"winner_conf"``: per pixel, keep the logits of the most confident agent
      (max softmax probability).

    Returns ``[B, H, W, num_classes]``.
    """
    patch_arr = np.asarray(patches)
    centers = np.asarray(centers_yx)
    if patch_arr.ndim != 5:
        raise ValueError("patches must have shape (N, B, ph, pw, num_classes)")
    if centers.ndim != 2 or centers.shape[0] != patch_arr.shape[0]:
        raise ValueError("centers_yx must align with patches on the agent axis")
    if patch_arr.shape[-1] != num_classes:
        raise ValueError("num_classes must match the patch channel dimension")
    mode = str(mode).lower()
    if mode not in {"mean", "winner_conf"}:
        raise ValueError(f"unknown reassembly mode {mode!r} (mean|winner_conf)")

    num_agents, batch, ph, pw, _ = patch_arr.shape
    h, w = int(image_hw[0]), int(image_hw[1])
    if ph != pw:
        raise ValueError("patches must be square")

    if mode == "mean":
        logits_sum = np.zeros((batch, h, w, num_classes), dtype=np.float32)
        counts = np.zeros((batch, h, w, 1), dtype=np.float32)
        for a in range(num_agents):
            ys, xs, pys, pxs = _overlap_slices(centers[a], ph, (h, w))
            if ys.start == ys.stop or xs.start == xs.stop:
                continue
            logits_sum[:, ys, xs, :] += patch_arr[a, :, pys, pxs, :]
            counts[:, ys, xs, :] += 1.0
        return logits_sum / np.maximum(counts, 1.0)

    best_logits = np.zeros((batch, h, w, num_classes), dtype=np.float32)
    best_conf = np.full((batch, h, w), -np.inf, dtype=np.float32)
    for a in range(num_agents):
        ys, xs, pys, pxs = _overlap_slices(centers[a], ph, (h, w))
        if ys.start == ys.stop or xs.start == xs.stop:
            continue
        chunk = patch_arr[a, :, pys, pxs, :].astype(np.float32, copy=False)
        probs = np.exp(chunk - np.max(chunk, axis=-1, keepdims=True))
        probs /= np.maximum(np.sum(probs, axis=-1, keepdims=True), 1e-12)
        conf = np.max(probs, axis=-1)  # [B, h', w']
        cur = best_conf[:, ys, xs]
        better = conf > cur
        best_conf[:, ys, xs] = np.where(better, conf, cur)
        best_logits[:, ys, xs, :] = np.where(better[..., None], chunk, best_logits[:, ys, xs, :])
    return best_logits


# =============================================================================
# Label patches (per-agent supervision)
# =============================================================================


def extract_patch_jax(
    image: jnp.ndarray, center_yx: jnp.ndarray, patch_size: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract a zero-padded patch and its validity mask around ``center_yx``.

    ``image``: ``[H, W]`` or ``[H, W, C]``. Returns ``(patch, mask)`` where the
    mask is 1.0 on real pixels and 0.0 on the zero-padding border.
    """
    pad = patch_size // 2
    cy, cx = center_yx[0], center_yx[1]
    multichannel = image.ndim == 3
    h, w = image.shape[:2]

    mask = jnp.ones((h, w), dtype=jnp.float32)
    if multichannel:
        image_padding = ((pad, pad), (pad, pad), (0, 0))
        image_start = (cy, cx, 0)
        image_size = (patch_size, patch_size, image.shape[2])
    else:
        image_padding = ((pad, pad), (pad, pad))
        image_start = (cy, cx)
        image_size = (patch_size, patch_size)

    padded_image = jnp.pad(image, image_padding, mode="constant", constant_values=0)
    padded_mask = jnp.pad(mask, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)
    patch = jax.lax.dynamic_slice(padded_image, image_start, image_size)
    patch_mask = jax.lax.dynamic_slice(padded_mask, (cy, cx), (patch_size, patch_size))
    return patch, patch_mask


def get_agent_label_patches_jax(
    labels: jnp.ndarray, centers: jnp.ndarray, patch_size: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per-agent label patches for patch-local supervision.

    ``labels``: ``[B, H, W]``; ``centers``: ``[N, 2]``. Returns
    ``(label_patches, label_masks)`` both ``[N, B, ph, pw]``; overlapping pixels
    are supervised once per covering agent.
    """
    vmapped = jax.vmap(
        jax.vmap(extract_patch_jax, in_axes=(None, 0, None)),
        in_axes=(0, None, None),
    )
    centers = jnp.asarray(centers, dtype=jnp.int32)
    patches, masks = vmapped(labels, centers, patch_size)  # [B, N, ph, pw]
    return jnp.transpose(patches, (1, 0, 2, 3)), jnp.transpose(masks, (1, 0, 2, 3))


# =============================================================================
# Sudoku views: 27-agent constraint slices and reassembly
# =============================================================================


def sudoku_slice_batch_jax(batch_grids: jnp.ndarray) -> jnp.ndarray:
    """Slice a 9x9 grid into the 27 constraint-agent views.

    ``batch_grids``: ``[B, 9, 9, C]``. Returns ``[B, 27, 9, C]`` where agents
    0-8 are rows, 9-17 columns (transpose), 18-26 the 3x3 boxes (reshape
    ``(3,3,3,3)`` then transpose ``(0,2,1,3)`` so each box flattens row-major).
    """
    grids = jnp.asarray(batch_grids)
    if grids.ndim != 4:
        raise ValueError(f"expected grids of shape (B,9,9,C); got {grids.shape}")
    C = grids.shape[3]

    def views_for(grid: jnp.ndarray) -> jnp.ndarray:
        rows = grid
        cols = jnp.transpose(grid, (1, 0, 2))
        boxes = grid.reshape(3, 3, 3, 3, C).transpose(0, 2, 1, 3, 4).reshape(9, 9, C)
        return jnp.concatenate([rows, cols, boxes], axis=0)

    return jax.vmap(views_for)(grids)


def reassemble_sudoku_logits(batch_logits: jnp.ndarray) -> jnp.ndarray:
    """Average the three covering views back to a 9x9 grid.

    ``batch_logits``: ``[B, 27, 9, C]``. Returns ``[B, 9, 9, C]``, the mean of
    the row, column, and box reconstructions (inverse of
    :func:`sudoku_slice_batch_jax`).
    """
    if batch_logits.ndim != 4:
        raise ValueError(f"expected logits of shape (B,27,9,C); got {batch_logits.shape}")

    def reconstruct(views: jnp.ndarray) -> jnp.ndarray:
        C = views.shape[-1]
        from_rows = views[0:9]
        from_cols = jnp.transpose(views[9:18], (1, 0, 2))
        from_boxes = views[18:27].reshape(3, 3, 3, 3, C).transpose(0, 2, 1, 3, 4).reshape(9, 9, C)
        return (from_rows + from_cols + from_boxes) / 3.0

    return jax.vmap(reconstruct)(batch_logits)


def build_sudoku_multigraph(
    grid_size: int = 9,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build the Sudoku constraint multigraph over the 27 agents.

    Each of the 81 cells is covered by exactly 3 agents (its row, column, box);
    those 3 agents form a clique for that cell, giving 3 edges per cell. The
    edge metadata records, per endpoint, *which* local 0-8 slot holds the shared
    cell — used to select the per-slot restriction map / LoRA factor.

    Returns ``(edge_indices [E, 2], map_u [E], map_v [E])`` with ``E = 243``.
    """
    global_ids = jnp.arange(grid_size * grid_size).reshape(1, grid_size, grid_size, 1)
    slices = np.asarray(sudoku_slice_batch_jax(global_ids)[0, :, :, 0])  # [27, 9]

    coverage: list[list[tuple[int, int]]] = [[] for _ in range(grid_size * grid_size)]
    for agent_id in range(slices.shape[0]):
        for local_idx in range(grid_size):
            coverage[int(slices[agent_id, local_idx])].append((agent_id, local_idx))

    edge_u, edge_v, map_u, map_v = [], [], [], []
    for agents in coverage:
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                (u, u_local), (v, v_local) = agents[i], agents[j]
                edge_u.append(u)
                edge_v.append(v)
                map_u.append(u_local)
                map_v.append(v_local)

    edges = jnp.asarray(np.stack([edge_u, edge_v], axis=1), dtype=jnp.int32)
    return edges, jnp.asarray(map_u, dtype=jnp.int32), jnp.asarray(map_v, dtype=jnp.int32)


@functools.lru_cache(maxsize=1)
def get_cached_sudoku_graph() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Cached :func:`build_sudoku_multigraph` for the standard 9x9 board."""
    return build_sudoku_multigraph(grid_size=9)


def build_sudoku_cell_indices(grid_size: int = 9) -> jnp.ndarray:
    """Global cell ids (0..80) seen by each agent: ``[27, 9]`` int32."""
    global_ids = jnp.arange(grid_size * grid_size).reshape(1, grid_size, grid_size, 1)
    return sudoku_slice_batch_jax(global_ids)[0, :, :, 0].astype(jnp.int32)


def sudoku_num_nodes(grid_size: int = 9) -> int:
    """Number of Sudoku constraint agents (= 3 * grid_size), derived not hardcoded."""
    edges, _, _ = build_sudoku_multigraph(grid_size)
    return int(np.asarray(edges).max()) + 1


def sudoku_node_positions(num_nodes: int) -> jnp.ndarray:
    """Principled ``head_valid`` node positions for the Sudoku agents.

    Position ``(agent // 9, agent % 9)`` lays the 27 agents on a 3x9 grid. This
    is only used to direction-index restriction maps; the Sudoku graph itself is
    the constraint multigraph, so the exact coordinates are immaterial as long
    as they are distinct. Do not reuse maze patch-grid centers here: distinct
    Sudoku agents must keep distinct direction-index coordinates.
    """
    agent_ids = jnp.arange(num_nodes, dtype=jnp.int32)
    return jnp.stack([agent_ids // 9, agent_ids % 9], axis=1).astype(jnp.float32)


# =============================================================================
# Maze wall-border pre-pad for boundary agents.
# =============================================================================


def prepare_maze_patches(
    inputs_img: jnp.ndarray,
    centers: jnp.ndarray,
    patch_size: int,
    num_input_classes: int,
    *,
    border_fill: float | None = None,
) -> jnp.ndarray:
    """Extract maze agent patches with a wall-token border pre-pad.

    :func:`patchify_batch_jax` zero-pads internally; for mazes that would feed
    boundary agents spurious zeros. We instead pad the image with a wall border
    of width ``patch_size // 2`` *before* extraction (and shift centers), so
    boundary patches see walls.

    ``inputs_img`` is ``[B, H, W]`` int tokens; patches are one-hot
    ``[N, B, ps, ps, num_input_classes]``. ``border_fill`` defaults to the wall
    token (``TOKEN_IDS["wall"] = 1``).
    """
    from .common import TOKEN_IDS

    inputs_img = jnp.asarray(inputs_img)
    border = patch_size // 2
    B = inputs_img.shape[0]
    eH, eW = int(inputs_img.shape[1]), int(inputs_img.shape[2])
    bH, bW = eH + 2 * border, eW + 2 * border
    centers_bordered = jnp.asarray(centers, dtype=jnp.int32) + border

    fill = float(TOKEN_IDS["wall"]) if border_fill is None else float(border_fill)
    bordered = jnp.full((B, bH, bW), fill, dtype=inputs_img.dtype)
    bordered = bordered.at[:, border : border + eH, border : border + eW].set(inputs_img)
    onehot = jax.nn.one_hot(bordered, num_input_classes, dtype=jnp.float32)
    return patchify_batch_jax(onehot, centers_bordered, patch_size)


# =============================================================================
# Misc
# =============================================================================


def infer_image_hw(
    seq_len: int, height: int | None = None, width: int | None = None
) -> tuple[int, int]:
    """Resolve ``(H, W)`` from ``seq_len``; require both dims for non-square grids."""
    if height is not None or width is not None:
        if height is None or width is None:
            raise ValueError("provide both height and width or neither")
        if height * width != seq_len:
            raise ValueError(f"dims {(height, width)} do not match seq_len={seq_len}")
        return int(height), int(width)
    root = int(math.isqrt(seq_len))
    if root * root != seq_len:
        raise ValueError("cannot infer non-square grid from seq_len; pass height/width")
    return root, root
