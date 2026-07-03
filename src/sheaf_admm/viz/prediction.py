"""Prediction-evolution artifacts (paper fig 3/4/5).

Renders the aggregated global prediction at increasing ADMM iteration counts
``k`` to show coordination sharpening the answer. For each ``k`` we run the
model with ``num_iters=k`` (``loss_window=1``, take the final x-iterate),
reassemble the per-agent logit patches into a global grid, and render one panel
per ``k`` in a row.

Per task:

* **Maze** — blue path overlay (predicted ``path`` token) on the grayscale maze,
  with start/goal markers.
* **Sudoku** — the 9x9 board with given clues (black) and filled digits (blue),
  constraint-violating cells highlighted red.
* **MNIST** — per-pixel argmax-class consensus as a discrete heatmap.
"""

from __future__ import annotations

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sheaf_admm.data import views as V
from sheaf_admm.data.common import TOKEN_IDS

from ._render import (
    render_maze_panel,
    render_mnist_panel,
    render_sudoku_panel,
    save_figure,
)


def _final_logits(model, params, fwd, k):
    """Per-agent logits at the last x-iterate of a ``k``-iteration run: ``[N, B, *out]``."""
    logits_window, _state, _geom = model.apply(
        params,
        fwd["patches"],
        fwd["edge_indices"],
        num_iters=int(k),
        loss_window=1,
        **fwd["model_kwargs"],
        training=False,
    )
    return logits_window[-1]


def _maze_predictions(model, params, fwd, aux, ks, batch_index):
    centers, image_hw = aux["centers"], aux["image_hw"]
    preds = []
    for k in ks:
        logits = np.asarray(_final_logits(model, params, fwd, k))  # [N, B, ph, pw, C]
        recon = V.reassemble_logits(
            logits, np.asarray(centers), image_hw, logits.shape[-1], mode="mean"
        )
        preds.append(np.argmax(recon[batch_index], axis=-1))
    return preds, image_hw


def _sudoku_predictions(model, params, fwd, ks, batch_index):
    preds = []
    for k in ks:
        logits = _final_logits(model, params, fwd, k)  # [N, B, 9, 10]
        recon = V.reassemble_sudoku_logits(jnp.transpose(logits, (1, 0, 2, 3)))  # [B, 9, 9, 10]
        preds.append(np.argmax(np.asarray(recon[batch_index]), axis=-1))
    return preds


def _mnist_grid(task, fwd):
    """Recompute the MNIST agent ``(centers, image_hw)`` (the task's ``aux`` is empty)."""
    ph = int(fwd["patches"].shape[2])
    # The padded image side is recoverable from the agent count and stride.
    n_per_side = int(round(np.sqrt(fwd["patches"].shape[0])))
    # First center is at patch_size//2; centers step by stride.
    side = (ph // 2) + (n_per_side - 1) * task.stride + 1
    centers = V.grid_agent_centers((side, side), stride=task.stride, patch_size=ph)
    return centers, (side, side)


def _mnist_predictions(model, params, task, fwd, ks, batch_index):
    centers, image_hw = _mnist_grid(task, fwd)
    preds = []
    for k in ks:
        logits = _final_logits(model, params, fwd, k)  # [N, B, ph, pw, C] or [N, B, C]
        logits = np.asarray(logits)
        if logits.ndim == 5:
            recon = V.reassemble_logits(
                logits, np.asarray(centers), image_hw, logits.shape[-1], mode="mean"
            )
            preds.append(("pixel", np.argmax(recon[batch_index], axis=-1)))
        else:
            # Classification head: one class per agent. Render the per-agent argmax
            # over the agent lattice (the spatial class-consensus map).
            n_per_side = int(round(np.sqrt(logits.shape[0])))
            grid = np.argmax(logits[:, batch_index], axis=-1).reshape(n_per_side, n_per_side)
            preds.append(("agent", grid))
    return preds


def plot_prediction_evolution(
    model,
    params,
    task,
    fwd: dict,
    targets: dict,
    aux: dict,
    ks,
    out_path: str,
    *,
    batch_index: int = 0,
    title: str | None = None,
):
    """Render the aggregated prediction at each ADMM count ``k`` as a one-row panel.

    ``task`` is the :mod:`sheaf_admm.training.tasks` hook (its ``.name`` selects
    the renderer). ``fwd``/``targets``/``aux`` are its ``prepare`` output. Writes
    the figure to ``out_path``.
    """
    ks = list(ks)
    task_name = task.name
    fig, axes = plt.subplots(1, len(ks), figsize=(2.6 * len(ks), 2.9), squeeze=False)
    axes = axes[0]

    if task_name == "maze":
        preds, _image_hw = _maze_predictions(model, params, fwd, aux, ks, batch_index)
        maze_img = np.asarray(targets["labels_img"][batch_index])
        wall, start, goal = TOKEN_IDS["wall"], TOKEN_IDS["start"], TOKEN_IDS["goal"]
        for ax, k, pred in zip(axes, ks, preds, strict=True):
            render_maze_panel(ax, maze_img, pred, wall, start, goal, TOKEN_IDS["path"])
            ax.set_title(f"k = {k}", fontsize=11)
    elif task_name == "sudoku":
        preds = _sudoku_predictions(model, params, fwd, ks, batch_index)
        givens = np.asarray(targets["inputs_grid"][batch_index])
        for ax, k, pred in zip(axes, ks, preds, strict=True):
            render_sudoku_panel(ax, givens, pred)
            ax.set_title(f"k = {k}", fontsize=11)
    elif task_name == "mnist":
        preds = _mnist_predictions(model, params, task, fwd, ks, batch_index)
        for ax, k, (_kind, pred) in zip(axes, ks, preds, strict=True):
            render_mnist_panel(ax, pred, num_classes=task.num_classes)
            ax.set_title(f"k = {k}", fontsize=11)
    else:
        raise ValueError(f"unknown task name {task_name!r} (maze|sudoku|mnist)")

    if title:
        fig.suptitle(title, fontsize=12)
    save_figure(fig, out_path)
    return out_path
