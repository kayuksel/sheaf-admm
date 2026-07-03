"""Data layer: views (grid <-> agents), on-disk loaders, and dataset builders.

* :mod:`~sheaf_admm.data.views` — the global-grid <-> per-agent map for maze /
  MNIST (grid patches) and Sudoku (constraint slices), plus reassembly and the
  agent graphs.
* :mod:`~sheaf_admm.data.loaders` — memory-mapped :class:`ImageDataset` and
  :class:`PuzzleDataset` with format-contract validation.
* ``build_maze`` / ``build_mnist`` / ``build_sudoku`` — argparse CLIs (run via
  ``python -m sheaf_admm.data.build_*``) for local/toy dataset generation.
"""

from __future__ import annotations

from .common import (
    DIHEDRAL_INVERSE,
    SUDOKU_VOCAB_SIZE,
    TOKEN_IDS,
    PuzzleDatasetMetadata,
    dihedral_transform,
    inverse_dihedral_transform,
    save_json,
    save_npy,
    shuffle_in_unison,
)
from .loaders import ImageDataset, PuzzleDataset, PuzzleSet
from .views import (
    build_grid_edge_indices,
    build_sudoku_cell_indices,
    build_sudoku_multigraph,
    get_agent_label_patches_jax,
    get_cached_sudoku_graph,
    grid_agent_centers,
    infer_image_hw,
    node_positions,
    patchify_batch_jax,
    prepare_maze_patches,
    reassemble_logits,
    reassemble_sudoku_logits,
    sudoku_node_positions,
    sudoku_num_nodes,
    sudoku_slice_batch_jax,
)

__all__ = [
    # common
    "TOKEN_IDS",
    "SUDOKU_VOCAB_SIZE",
    "DIHEDRAL_INVERSE",
    "PuzzleDatasetMetadata",
    "dihedral_transform",
    "inverse_dihedral_transform",
    "save_json",
    "save_npy",
    "shuffle_in_unison",
    # loaders
    "ImageDataset",
    "PuzzleDataset",
    "PuzzleSet",
    # views: grid
    "grid_agent_centers",
    "patchify_batch_jax",
    "build_grid_edge_indices",
    "node_positions",
    "reassemble_logits",
    "get_agent_label_patches_jax",
    "prepare_maze_patches",
    "infer_image_hw",
    # views: sudoku
    "sudoku_slice_batch_jax",
    "reassemble_sudoku_logits",
    "build_sudoku_multigraph",
    "get_cached_sudoku_graph",
    "build_sudoku_cell_indices",
    "sudoku_num_nodes",
    "sudoku_node_positions",
]
