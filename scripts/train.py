"""Hydra training entrypoint for Sheaf-ADMM (and the MPNN baseline).

One entrypoint for every task (maze / mnist / sudoku) and both model families
(``model_type=sheaf|mpnn``); the task and HPs come entirely from config.

    python scripts/train.py +experiment=maze_sheaf
    python scripts/train.py +experiment=sudoku_sheaf training.seed=123
    python scripts/train.py +experiment=mnist_sheaf

Runs do not log by default. Set ``wandb.mode=online`` to enable Weights & Biases.
Importing ``sheaf_admm`` pins ``float32`` matmul precision to ``highest`` (the
paper setting) before any compilation.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import hydra
import jax
import numpy as np
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import sheaf_admm as _sheaf_admm  # noqa: F401,E402  (sets matmul precision on import)
from sheaf_admm.data import ImageDataset, PuzzleDataset  # noqa: E402
from sheaf_admm.models import model_config_from_dict  # noqa: E402
from sheaf_admm.training import (  # noqa: E402
    build_model,
    create_train_state,
    evaluate,
    make_task,
    make_train_step,
    sample_k,
)


def _puzzle_batch(batch):
    out = {"inputs": np.asarray(batch["inputs"]), "labels": np.asarray(batch["labels"])}
    for key in ("height", "width"):
        if key in batch:
            out[key] = batch[key]
    return out


def _train_batches(cfg: DictConfig, epoch: int):
    """Yield ``{inputs/images, labels}`` batches for one training epoch."""
    d = cfg.data
    if d.loader == "puzzle":
        ds = PuzzleDataset(d.dir, d.train_split)
        for _set, batch in ds.iter_train_batches(
            cfg.training.batch_size, seed=cfg.training.seed + epoch
        ):
            yield _puzzle_batch(batch)
    else:  # image
        ds = ImageDataset(d.dir, d.train_split)
        for batch in ds.iter_batches(
            cfg.training.batch_size, shuffle=True, seed=cfg.training.seed + epoch
        ):
            yield {"images": np.asarray(batch["images"]), "labels": np.asarray(batch["labels"])}


def _val_batches(cfg: DictConfig, split: str):
    d = cfg.data
    if d.loader == "puzzle":
        ds = PuzzleDataset(d.dir, split)
        for _set, batch in ds.iter_test_batches(cfg.training.batch_size):
            yield _puzzle_batch(batch)
    else:
        ds = ImageDataset(d.dir, split)
        for batch in ds.iter_batches(cfg.training.batch_size, shuffle=False):
            yield {"images": np.asarray(batch["images"]), "labels": np.asarray(batch["labels"])}


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    sys.stdout.reconfigure(line_buffering=True)  # flush per line so sbatch log-tailing works live
    print(OmegaConf.to_yaml(cfg))
    t = cfg.training
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name,
        group=cfg.wandb.group,
        tags=list(cfg.wandb.tags),
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    task = make_task(cfg.task, **OmegaConf.to_container(cfg.task_cfg, resolve=True))
    model_cfg = model_config_from_dict(OmegaConf.to_container(cfg.model, resolve=True))
    model = build_model(model_cfg, cfg.model_type)
    graph_readout = model_cfg.mpnn_graph_readout

    sample_fwd, _, _ = task.prepare(next(_train_batches(cfg, 0)))
    state = create_train_state(
        model,
        sample_fwd,
        model_type=cfg.model_type,
        lr=t.lr,
        weight_decay=t.weight_decay,
        warmup_steps=t.warmup_steps,
        grad_clip=t.grad_clip,
        ema_decay=t.ema_decay,
        k_init=t.K_train,
        loss_window=t.loss_window,
        seed=cfg.training.seed,
    )
    run.summary["params"] = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    print(f"[init] model_type={cfg.model_type} params={run.summary['params']:,}")

    train_step = make_train_step(task, cfg.model_type, graph_readout)
    rng = jax.random.PRNGKey(cfg.training.seed)
    rng_np = np.random.default_rng(cfg.training.seed)
    history: list[dict] = []
    best: dict[str, float] = {}
    step = 0

    for epoch in range(t.epochs):
        for batch in _train_batches(cfg, epoch):
            fwd, targets, _ = task.prepare(batch)
            rng, sub = jax.random.split(rng)
            k = (
                sample_k(rng_np, t.train_iters_dist, t.train_iters_min, t.K_train)
                if cfg.model_type == "sheaf"
                else t.mpnn_train_rounds
            )
            state, loss = train_step(state, fwd, targets, sub, n_iter=k, loss_window=t.loss_window)
            step += 1
            loss = float(loss)
            run.log({"train/loss": loss, "train/k": k, "epoch": epoch}, step=step)
            if t.exit_on_nan and not np.isfinite(loss):
                print(f"[epoch {epoch}] non-finite loss — stopping (exit_on_nan).")
                run.finish(exit_code=1)
                return

        if epoch % t.val_interval == 0 or epoch == t.epochs - 1:
            k_eval = t.K_eval if cfg.model_type == "sheaf" else t.mpnn_eval_rounds
            row = {"epoch": epoch, "loss": loss}
            for split in cfg.data.val_splits:
                m = evaluate(
                    state,
                    task,
                    _val_batches(cfg, split),
                    model_type=cfg.model_type,
                    graph_readout=graph_readout,
                    k_eval=k_eval,
                )
                row[split] = m
                run.log({f"val/{split}/{kk}": vv for kk, vv in m.items()}, step=step)
                for kk, vv in m.items():  # track best-so-far in the run summary
                    key = f"best/{split}/{kk}"
                    best[key] = max(best.get(key, vv), vv)
                print(
                    f"[epoch {epoch}] loss={loss:.4f}  {split}: "
                    + "  ".join(f"{kk}={vv * 100:.2f}%" for kk, vv in m.items())
                )
            history.append(row)
            run.summary.update(best)

    out = Path(HydraConfig.get().runtime.output_dir)
    with open(out / "checkpoint.pkl", "wb") as f:
        pickle.dump(
            {
                "params": jax.device_get(state.params),
                "ema_params": jax.device_get(state.ema_params),
                "config": OmegaConf.to_container(cfg, resolve=True),
            },
            f,
        )
    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[done] saved checkpoint + history to {out}")
    run.finish()


if __name__ == "__main__":
    main()
