"""Views: grid patchify/reassemble round-trips, Sudoku slice/reassemble round-trips,
and the constraint multigraph has the expected 243 edges over 27 agents."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

from sheaf_admm.data import views as V


def test_grid_patchify_reassemble_roundtrip():
    """Mean-reassembly of patches cut from one image recovers that image on the
    covered pixels (every pixel is covered with stride 2, patch 3)."""
    H, W, C = 8, 8, 3
    stride, ps = 2, 3
    centers = V.grid_agent_centers((H, W), stride=stride, patch_size=ps)
    B = 2
    img = jax.random.normal(jax.random.PRNGKey(0), (B, H, W, C))
    patches = V.patchify_batch_jax(img, jnp.asarray(centers), ps)
    assert patches.shape == (centers.shape[0], B, ps, ps, C)

    recon = V.reassemble_logits(np.asarray(patches), centers, (H, W), C, mode="mean")
    assert recon.shape == (B, H, W, C)
    np.testing.assert_allclose(recon, np.asarray(img), atol=1e-5)


def test_grid_agent_centers_and_edges():
    centers = V.grid_agent_centers((8, 8), stride=2, patch_size=3)
    assert centers.shape == (16, 2)  # 4x4 grid of centers
    edges4 = V.build_grid_edge_indices(centers, stride=2, connectivity=4)
    edges8 = V.build_grid_edge_indices(centers, stride=2, connectivity=8)
    # 4-way: right+down on a 4x4 grid = 2*4*3 = 24; 8-way adds the two diagonals.
    assert edges4.shape == (24, 2)
    assert edges8.shape[0] > edges4.shape[0]
    assert int(edges8.max()) < centers.shape[0]


def test_sudoku_slice_reassemble_roundtrip():
    """The 27 row/col/box views reassemble (averaged) back to the 9x9 grid."""
    B, C = 2, 5
    grid = jax.random.normal(jax.random.PRNGKey(1), (B, 9, 9, C))
    views = V.sudoku_slice_batch_jax(grid)
    assert views.shape == (B, 27, 9, C)
    recon = V.reassemble_sudoku_logits(views)
    assert recon.shape == (B, 9, 9, C)
    np.testing.assert_allclose(np.asarray(recon), np.asarray(grid), atol=1e-5)


def test_sudoku_multigraph_structure():
    """81 cells x 3 edges per cell-clique = 243 edges over 27 agents."""
    edges, map_u, map_v = V.build_sudoku_multigraph(9)
    assert edges.shape == (243, 2)
    assert map_u.shape == (243,) and map_v.shape == (243,)
    assert int(edges.max()) + 1 == 27
    # Cell-slot indices are in 0..8 (the 9-cell local view).
    assert int(map_u.max()) < 9 and int(map_v.max()) < 9
    assert int(map_u.min()) >= 0 and int(map_v.min()) >= 0


def test_sudoku_cell_indices():
    """Each of the 27 agents sees 9 cells; the 81 board cells are each seen 3 times."""
    cell_ids = V.build_sudoku_cell_indices(9)
    assert cell_ids.shape == (27, 9)
    flat = np.asarray(cell_ids).reshape(-1)
    counts = np.bincount(flat, minlength=81)
    assert counts.shape == (81,)
    np.testing.assert_array_equal(counts, np.full(81, 3))


def test_maze_wall_border_prepad():
    """``prepare_maze_patches`` one-hots tokens and pads the *border* with the wall
    token, so a boundary agent whose patch runs out of bounds sees walls there
    (not the zero-padding ``patchify`` would otherwise insert)."""
    from sheaf_admm.data.common import TOKEN_IDS

    B, H, W, ps, ncls = 1, 6, 6, 3, 6
    inp = jnp.zeros((B, H, W), dtype=jnp.int32)  # interior token 0 everywhere
    centers = V.grid_agent_centers((H, W), stride=2, patch_size=ps)
    patches = V.prepare_maze_patches(inp, jnp.asarray(centers), ps, ncls)
    assert patches.shape == (centers.shape[0], B, ps, ps, ncls)

    # The agent at center (5, 5) has a 3x3 patch covering rows/cols 4,5,6; row 6
    # and col 6 are out of bounds -> wall token, while in-bounds pixels stay 0.
    cs = np.asarray(centers)
    idx = int(np.where((cs[:, 0] == 5) & (cs[:, 1] == 5))[0][0])
    am = np.argmax(np.asarray(patches[idx, 0]), axis=-1)  # [ps, ps]
    assert am[2, 2] == TOKEN_IDS["wall"]  # bottom-right (out of bounds) -> wall
    assert am[0, 0] == 0  # top-left (in bounds) -> interior token


def test_maze_task_prepare_rectangular_metadata():
    """Rectangular OOD maze splits carry height/width metadata through the task hook."""
    from sheaf_admm.training.tasks import MazeTask

    B, H, W = 1, 19, 37
    batch = {
        "inputs": np.zeros((B, H * W), dtype=np.int32),
        "labels": np.zeros((B, H * W), dtype=np.int32),
        "height": H,
        "width": W,
    }
    fwd, targets, aux = MazeTask().prepare(batch)
    assert aux["image_hw"] == (H, W)
    assert fwd["patches"].shape[1] == B
    assert targets["labels_img"].shape == (B, H, W)


def test_puzzle_dataset_batches_include_grid_metadata(tmp_path):
    """Puzzle batches preserve metadata dimensions for non-square grids."""
    from sheaf_admm.data.common import PuzzleDatasetMetadata
    from sheaf_admm.data.loaders import PuzzleDataset

    split_dir = tmp_path / "train"
    split_dir.mkdir()
    H, W = 19, 37
    np.save(split_dir / "all__inputs.npy", np.zeros((1, H * W), dtype=np.uint8))
    np.save(split_dir / "all__labels.npy", np.zeros((1, H * W), dtype=np.uint8))
    np.save(split_dir / "all__puzzle_indices.npy", np.array([0, 1], dtype=np.int32))
    np.save(split_dir / "all__group_indices.npy", np.array([0, 1], dtype=np.int32))
    np.save(split_dir / "all__puzzle_identifiers.npy", np.array([0], dtype=np.int32))
    PuzzleDatasetMetadata(
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        vocab_size=6,
        seq_len=H * W,
        num_puzzle_identifiers=1,
        total_groups=1,
        mean_puzzle_examples=1.0,
        total_puzzles=1,
        sets=["all"],
        height=H,
        width=W,
    ).write(split_dir / "dataset.json")

    _set, batch = next(PuzzleDataset(tmp_path, "train").iter_test_batches(1))
    assert batch["height"] == H
    assert batch["width"] == W


def test_puzzle_metadata_ignores_legacy_format_keys(tmp_path):
    """Old standard-token dataset metadata may include now-unused format keys."""
    from sheaf_admm.data.common import PuzzleDatasetMetadata

    payload = {
        "pad_id": 0,
        "ignore_label_id": 0,
        "blank_identifier_id": 0,
        "vocab_size": 6,
        "seq_len": 35,
        "num_puzzle_identifiers": 1,
        "total_groups": 1,
        "mean_puzzle_examples": 1.0,
        "total_puzzles": 1,
        "sets": ["all"],
        "height": 5,
        "width": 7,
        "input_format": "standard",
        "num_input_channels": None,
    }
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(payload))

    metadata = PuzzleDatasetMetadata.from_json(path)
    assert metadata.seq_len == 35
    assert metadata.height == 5
    assert metadata.width == 7
