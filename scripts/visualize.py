"""Emit the paper's analysis artifacts from a trained Sheaf-ADMM checkpoint.

Loads a ``checkpoint.pkl`` written by ``scripts/train.py`` (a dict with
``params`` / ``ema_params`` / ``config``), rebuilds the model + task from the
embedded config, pulls one validation batch, and writes the three artifacts:

* ``prediction_evolution.pdf`` — global prediction vs ADMM count ``k``;
* ``coordination_dynamics.pdf`` — primal/dual residual heatmaps + curves;
* ``xz_trajectories.pdf`` — per-agent x-vs-z 2-D trajectories.

    python -m scripts.visualize --checkpoint outputs/.../checkpoint.pkl --out-dir /tmp/viz

The ``--task`` / ``--data-dir`` flags override what the checkpoint config says
(useful for visualizing on an OOD split). Evaluation uses the EMA parameters by
default (``--params ema``), matching how the trainer reports metrics.
Only Sheaf-ADMM checkpoints are supported; MPNN baselines do not expose an ADMM
trajectory.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import sheaf_admm as _sheaf_admm  # noqa: F401,E402  (sets matmul precision on import)
from sheaf_admm.data import ImageDataset, PuzzleDataset  # noqa: E402
from sheaf_admm.models import model_config_from_dict  # noqa: E402
from sheaf_admm.training import build_model, make_task  # noqa: E402
from sheaf_admm.viz import (  # noqa: E402
    plot_coordination_dynamics,
    plot_prediction_evolution,
    plot_xz_trajectories,
    run_trajectory,
)

# Default per-task ADMM counts for the prediction-evolution panels.
DEFAULT_KS = {"maze": (1, 3, 5, 10, 30), "sudoku": (1, 3, 5, 10, 20), "mnist": (1, 3, 5, 10, 30)}


def _first_batch(cfg: dict, task_name: str, data_dir: str, split: str):
    loader = cfg["data"].get("loader", "puzzle" if task_name in ("maze", "sudoku") else "image")
    if loader == "puzzle":
        ds = PuzzleDataset(data_dir, split)
        for _set, b in ds.iter_test_batches(batch_size=8):
            batch = {"inputs": np.asarray(b["inputs"]), "labels": np.asarray(b["labels"])}
            for key in ("height", "width"):
                if key in b:
                    batch[key] = b[key]
            return batch
    ds = ImageDataset(data_dir, split)
    for b in ds.iter_batches(batch_size=8, shuffle=False):
        return {"images": np.asarray(b["images"]), "labels": np.asarray(b["labels"])}
    raise RuntimeError("no batch available from the dataset")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="path to checkpoint.pkl")
    p.add_argument("--out-dir", required=True, help="directory for the artifact files")
    p.add_argument("--task", default=None, help="override task (maze|sudoku|mnist)")
    p.add_argument("--data-dir", default=None, help="override dataset directory")
    p.add_argument("--split", default=None, help="dataset split (default: first val split)")
    p.add_argument("--params", choices=("ema", "raw"), default="ema", help="which weights to use")
    p.add_argument(
        "--num-iters", type=int, default=30, help="ADMM steps for the trajectory artifacts"
    )
    p.add_argument("--batch-index", type=int, default=0, help="example index within the batch")
    p.add_argument(
        "--ks", type=int, nargs="+", default=None, help="ADMM counts for prediction evolution"
    )
    args = p.parse_args()

    with open(args.checkpoint, "rb") as f:
        ckpt = pickle.load(f)
    cfg = ckpt["config"]
    if cfg.get("model_type") != "sheaf":
        raise ValueError(
            "scripts.visualize supports Sheaf-ADMM checkpoints only "
            f"(got model_type={cfg.get('model_type')!r})."
        )
    params = ckpt.get("ema_params") if args.params == "ema" else ckpt["params"]
    if params is None:
        params = ckpt["params"]

    task_name = args.task or cfg["task"]
    data_dir = args.data_dir or cfg["data"]["dir"]
    split = args.split or cfg["data"]["val_splits"][0]
    ks = tuple(args.ks) if args.ks else DEFAULT_KS[task_name]

    task = make_task(task_name, **cfg.get("task_cfg", {}))
    model_cfg = model_config_from_dict(cfg["model"])
    model = build_model(model_cfg, cfg["model_type"])

    batch = _first_batch(cfg, task_name, data_dir, split)
    fwd, targets, aux = task.prepare(batch)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    written.append(
        plot_prediction_evolution(
            model,
            params,
            task,
            fwd,
            targets,
            aux,
            ks,
            str(out_dir / "prediction_evolution.pdf"),
            batch_index=args.batch_index,
            title=f"{task_name}: prediction vs k",
        )
    )

    centers = aux.get("centers")
    traj = run_trajectory(
        model,
        params,
        fwd,
        num_iters=args.num_iters,
        batch_index=args.batch_index,
        centers=centers,
    )
    written.append(
        plot_coordination_dynamics(
            traj,
            str(out_dir / "coordination_dynamics.pdf"),
            title=f"{task_name}: coordination dynamics (rho={traj.rho:.3g})",
        )
    )
    written.append(
        plot_xz_trajectories(
            traj,
            str(out_dir / "xz_trajectories.pdf"),
            title=f"{task_name}: x vs z trajectories",
        )
    )

    for path in written:
        print(f"[viz] wrote {path}")


if __name__ == "__main__":
    main()
