"""End-to-end pipeline guard: a tiny Maze overfit.

Builds ~32 small 9x9 mazes, trains the full Sheaf-ADMM model (encode -> LoRA
restriction maps -> unrolled CG ADMM -> decode -> windowed CE loss) for ~150
steps on a single fixed batch with ``ema_decay=0`` (so eval uses the live
trained params, not a slow-moving EMA shadow), and asserts the train loss drops
well below 0.2 and the eval solve-rate climbs above 50%. This exercises the whole
data -> model -> loss -> optimizer -> eval stack and catches any regression that
breaks training end-to-end. CPU-only, ~20s; marked ``slow``.
"""

from __future__ import annotations

import jax
import pytest

from sheaf_admm.data.build_maze import MazeConfig, build
from sheaf_admm.data.loaders import PuzzleDataset
from sheaf_admm.models import ModelConfig
from sheaf_admm.training.loop import build_model, create_train_state, evaluate, make_train_step
from sheaf_admm.training.tasks import make_task


def _overfit_config() -> ModelConfig:
    # The maze_sheaf architecture (d_v=10, LoRA rank 4, l1box), shrunk only on the
    # encoder/decoder MLP width for CPU speed; solver knobs match the maze config.
    return ModelConfig(
        num_classes=6,
        d_v=10,
        d_e=5,
        encoder_arch="mlp_v2",
        enc_hidden_dim=128,
        comm_norm_type="layernorm",
        objective_mode="l1box_diag",
        x_solver="diagonal_prox",
        z_solver="unrolled_cg",
        z_mode="prox",
        gamma=5.0,
        cg_iters=5,
        rm_sharing="directional",
        rm_init="orthonormal",
        rm_mode="context",
        lora_rank=4,
        lora_init_style="standard",
        num_directions=8,
        rho_init=0.25,
        decoder_arch="mlp_concat_v2",
        dec_hidden_dim=128,
    )


@pytest.mark.slow
def test_maze_overfit_drops_loss_and_solves(tmp_path):
    data_dir = tmp_path / "maze"
    build(
        MazeConfig(
            height=9,
            width=9,
            train_size=32,
            test_size=16,
            min_path_length=6,
            seed=0,
            output_dir=data_dir,  # MazeConfig.output_dir is a Path
        )
    )

    task = make_task("maze", patch_size=3, stride=2, connectivity=8, num_classes=6)
    ds = PuzzleDataset(str(data_dir), split="train")
    batch = next(b for _set, b in ds.iter_test_batches(32))  # one fixed batch to overfit
    fwd, targets, _aux = task.prepare(batch)

    cfg = _overfit_config()
    model = build_model(cfg, "sheaf")
    K = 20
    state = create_train_state(
        model,
        fwd,
        model_type="sheaf",
        lr=1e-3,
        weight_decay=1e-6,
        warmup_steps=20,
        grad_clip=1.0,
        ema_decay=0.0,
        k_init=K,
        loss_window=4,
        seed=0,
    )
    train_step = make_train_step(task, "sheaf", "per_node")

    rng = jax.random.PRNGKey(0)
    final_loss = None
    for _step in range(150):
        rng, sub = jax.random.split(rng)
        state, final_loss = train_step(state, fwd, targets, sub, n_iter=K, loss_window=4)

    assert float(final_loss) < 0.2, f"train loss did not overfit: {float(final_loss)}"

    metrics = evaluate(
        state, task, [batch], model_type="sheaf", graph_readout="per_node", k_eval=40
    )
    assert metrics["solved"] > 0.5, f"live-param solve rate too low: {metrics}"
    assert metrics["cell_acc"] > 0.9, f"cell accuracy too low: {metrics}"
