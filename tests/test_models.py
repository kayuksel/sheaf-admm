"""Model wiring + parameter-budget assertions.

These tests initialize the public experiment configs and lock their parameter
counts, so silent shape drift in encoder/decoder/geometry does not slip in.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from sheaf_admm.data import views as V
from sheaf_admm.models import MPNNModel, SheafADMMModel, model_config_from_dict

_CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "experiment"


def _count(params) -> int:
    return int(sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params)))


def _load_model_cfg(name: str):
    d = yaml.safe_load((_CONFIGS / f"{name}.yaml").read_text())
    return model_config_from_dict(d["model"])


def test_model_config_ignores_legacy_noop_keys():
    cfg = model_config_from_dict({"num_classes": 6, "task": "grid", "mpnn_num_rounds": 40})
    assert cfg.num_classes == 6


def _grid_inputs(cfg, *, H=7, W=7, B=2, stride, patch_size, connectivity, channels):
    centers = V.grid_agent_centers((H, W), stride=stride, patch_size=patch_size)
    edges = jnp.asarray(V.build_grid_edge_indices(centers, stride, connectivity))
    npos = V.node_positions(centers)
    patches = jnp.zeros((centers.shape[0], B, patch_size, patch_size, channels))
    return patches, edges, npos


def _sudoku_inputs(cfg, *, B=2):
    edges, map_u, map_v = V.build_sudoku_multigraph(9)
    cell_ids = V.build_sudoku_cell_indices(9)
    patches = jnp.zeros((27, B, 9, cfg.num_classes))
    return patches, edges, map_u, map_v, cell_ids


def _init_sheaf(cfg, fwd_kwargs):
    model = SheafADMMModel(config=cfg)
    return model.init(
        jax.random.PRNGKey(0), **fwd_kwargs, num_iters=2, loss_window=1, training=False
    )


def _init_mpnn(cfg, fwd_kwargs):
    model = MPNNModel(config=cfg)
    return model.init(jax.random.PRNGKey(0), **fwd_kwargs, num_rounds=2, training=False)


# --------------------------------------------------------------------------- #
# Parameter budgets (the paper's matched ~182K maze, ~49K compute-matched MPNN).
# --------------------------------------------------------------------------- #


def test_maze_sheaf_param_count():
    """Full Maze Sheaf-ADMM (d_v=10, d_e=5, LoRA rank 4, l1box) == 181,859 (~182K)."""
    cfg = _load_model_cfg("maze_sheaf")
    patches, edges, npos = _grid_inputs(
        cfg, stride=2, patch_size=3, connectivity=8, channels=cfg.num_classes
    )
    params = _init_sheaf(cfg, dict(patches=patches, edge_indices=edges, node_positions=npos))
    assert _count(params) == 181_859


def test_maze_mpnn_param_matched_count():
    """MPNN baseline parameter-matched to the Sheaf model (d_v=84) == 182,007 (~182K)."""
    cfg = _load_model_cfg("maze_mpnn")
    patches, edges, npos = _grid_inputs(
        cfg, stride=2, patch_size=3, connectivity=8, channels=cfg.num_classes
    )
    params = _init_mpnn(cfg, dict(patches=patches, edge_indices=edges, node_positions=npos))
    assert _count(params) == 182_007


def test_maze_mpnn_compute_matched_count():
    """Compute-matched MPNN (d_v=10, d_e=5, msg=5) == 48,807 (~49K)."""
    cfg = _load_model_cfg("maze_mpnn")
    cm = cfg.__class__(**{**cfg.__dict__, "d_v": 10, "d_e": 5, "mpnn_message_dim": 5})
    patches, edges, npos = _grid_inputs(
        cm, stride=2, patch_size=3, connectivity=8, channels=cm.num_classes
    )
    params = _init_mpnn(cm, dict(patches=patches, edge_indices=edges, node_positions=npos))
    assert _count(params) == 48_807


# --------------------------------------------------------------------------- #
# Sudoku encoder/decoder shapes (the slice-block contract: d_v = 9 * cell_dim).
# --------------------------------------------------------------------------- #


def test_sudoku_sheaf_builds_and_shapes():
    cfg = _load_model_cfg("sudoku_sheaf")
    assert cfg.d_v % 9 == 0 and cfg.d_v == 9 * cfg.d_e  # slice-selector contract
    patches, edges, map_u, map_v, cell_ids = _sudoku_inputs(cfg)
    model = SheafADMMModel(config=cfg)
    params = model.init(
        jax.random.PRNGKey(0),
        patches,
        edges,
        num_iters=2,
        loss_window=2,
        map_u=map_u,
        map_v=map_v,
        cell_ids=cell_ids,
        training=False,
    )
    logits, state, geom = model.apply(
        params,
        patches,
        edges,
        num_iters=2,
        loss_window=2,
        map_u=map_u,
        map_v=map_v,
        cell_ids=cell_ids,
        training=False,
    )
    B = patches.shape[1]
    assert logits.shape == (2, 27, B, 9, cfg.num_classes)  # [W, N, B, 9 cells, 10 digits]
    assert state.z.shape == (27, B, cfg.d_v)
    # Lock the public Sudoku fixed-RM config count.
    assert _count(params) == 543_025


def test_sudoku_lora_param_count_exceeds_fixed():
    """The LoRA variant adds the encoder's A/B factor heads on top of the fixed model."""
    fixed = _load_model_cfg("sudoku_sheaf")
    lora = _load_model_cfg("sudoku_sheaf_lora")
    p_fixed = _init_sheaf(
        fixed,
        dict(
            zip(
                ("patches", "edge_indices", "map_u", "map_v", "cell_ids"),
                _sudoku_inputs(fixed),
                strict=True,
            )
        ),
    )
    p_lora = _init_sheaf(
        lora,
        dict(
            zip(
                ("patches", "edge_indices", "map_u", "map_v", "cell_ids"),
                _sudoku_inputs(lora),
                strict=True,
            )
        ),
    )
    assert _count(p_fixed) == 543_025 and _count(p_lora) == 2_029_233
    assert _count(p_lora) > _count(p_fixed)


def test_sudoku_encoder_block_layout():
    """The Sudoku encoder lays the 9 cells contiguously: ``h`` has width ``d_v``."""
    from sheaf_admm.models import create_encoder

    cfg = _load_model_cfg("sudoku_sheaf")
    enc = create_encoder(
        "sudoku",
        comm_dim=cfg.d_v,
        edge_stalk_dim=cfg.d_e,
        d_model=cfg.enc_d_model,
        num_blocks=cfg.enc_num_blocks,
        comm_norm_type=cfg.comm_norm_type,
        objective_mode=cfg.objective_mode,
    )
    x = jnp.zeros((5, 9, cfg.num_classes))  # B'=5 agents, 9 cells, 10 digits
    out = enc.init_with_output(jax.random.PRNGKey(0), x)[0]
    assert out["h"].shape == (5, cfg.d_v)
    assert out["q_diag"].shape == (5, cfg.d_v) and out["q"].shape == (5, cfg.d_v)
    assert "lower" in out  # non_negative objective


def test_mnist_sheaf_builds():
    """Project-mode (Fz=0) global-RM MNIST classification model builds and decodes."""
    cfg = _load_model_cfg("mnist_sheaf")
    patches, edges, npos = _grid_inputs(
        cfg, H=9, W=9, stride=3, patch_size=3, connectivity=8, channels=1
    )
    params = _init_sheaf(cfg, dict(patches=patches, edge_indices=edges, node_positions=npos))
    model = SheafADMMModel(config=cfg)
    logits, _state, _geom = model.apply(
        params, patches, edges, num_iters=2, loss_window=1, node_positions=npos, training=False
    )
    # classification decoder -> per-agent class logits [W, N, B, num_classes]
    assert logits.shape[-1] == cfg.num_classes
    assert np.all(np.isfinite(np.asarray(logits)))
