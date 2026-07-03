"""Task hooks: decomposition, loss, and evaluation per benchmark.

A :class:`Task` turns a raw batch into three pieces (``prepare``):

* ``fwd``    — model inputs (all arrays): ``patches``, ``edge_indices``, and
  ``model_kwargs`` (graph extras the model needs: ``node_positions`` for grids,
  ``map_u``/``map_v``/``cell_ids`` for Sudoku). Safe to pass through ``jit``.
* ``targets`` — label arrays for the loss / metrics.
* ``aux``    — non-jittable extras (agent centers, image size) used only for the
  eager eval-time reassembly.

``loss(logits_window, targets)`` scores the windowed per-agent logits;
``evaluate(logits_final, targets, aux)`` reassembles the global prediction and
returns task metrics. Adding a benchmark = adding a Task.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from sheaf_admm.data import views as V

PATH_TOKEN = 5  # maze: label token marking a path cell


def _ce(logits, labels):
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels)


def _windowed(loss_one, logits_window):
    """Average a per-iterate loss over the ADMM loss window (axis 0)."""
    return jnp.mean(jax.vmap(loss_one)(logits_window))


def _grid_graph(image_hw, stride, patch_size, connectivity):
    centers = V.grid_agent_centers(image_hw, stride=stride, patch_size=patch_size)
    edge_indices = jnp.asarray(V.build_grid_edge_indices(centers, stride, connectivity))
    return centers, edge_indices, V.node_positions(centers)


class MazeTask:
    name = "maze"

    def __init__(self, *, patch_size=3, stride=2, connectivity=8, num_classes=6):
        self.patch_size, self.stride, self.connectivity = patch_size, stride, connectivity
        self.num_classes = num_classes

    def prepare(self, batch):
        inputs = jnp.asarray(batch["inputs"])
        labels = jnp.asarray(batch["labels"])
        B = inputs.shape[0]
        H, W = V.infer_image_hw(inputs.shape[1], batch.get("height"), batch.get("width"))
        inp_img, lab_img = inputs.reshape(B, H, W), labels.reshape(B, H, W)
        centers, edges, npos = _grid_graph((H, W), self.stride, self.patch_size, self.connectivity)
        patches = V.prepare_maze_patches(inp_img, centers, self.patch_size, self.num_classes)
        label_patches, masks = V.get_agent_label_patches_jax(lab_img, centers, self.patch_size)
        fwd = {"patches": patches, "edge_indices": edges, "model_kwargs": {"node_positions": npos}}
        targets = {"label_patches": label_patches, "masks": masks, "labels_img": lab_img}
        aux = {"centers": centers, "image_hw": (H, W)}
        return fwd, targets, aux

    def loss(self, logits_window, targets):
        lp, masks = targets["label_patches"], targets["masks"]

        def one(logits):
            ce = _ce(logits, lp) * masks
            return jnp.sum(ce) / jnp.maximum(jnp.sum(masks), 1.0)

        return _windowed(one, logits_window)

    def evaluate(self, logits_final, targets, aux):
        recon = jnp.asarray(
            V.reassemble_logits(
                logits_final, aux["centers"], aux["image_hw"], self.num_classes, mode="mean"
            )
        )
        pred = jnp.argmax(recon, axis=-1)
        labels = targets["labels_img"]
        solved = jnp.all((pred == PATH_TOKEN) == (labels == PATH_TOKEN), axis=(1, 2))
        return {"solved": float(jnp.mean(solved)), "cell_acc": float(jnp.mean(pred == labels))}


class MNISTTask:
    name = "mnist"

    def __init__(self, *, patch_size=3, stride=3, connectivity=8, num_classes=10):
        self.patch_size, self.stride, self.connectivity, self.num_classes = (
            patch_size,
            stride,
            connectivity,
            num_classes,
        )

    def prepare(self, batch):
        images = jnp.asarray(batch["images"])
        labels = jnp.asarray(batch["labels"])
        H, W = images.shape[1], images.shape[2]
        centers, edges, npos = _grid_graph((H, W), self.stride, self.patch_size, self.connectivity)
        patches = V.patchify_batch_jax(images, centers, self.patch_size)
        fwd = {"patches": patches, "edge_indices": edges, "model_kwargs": {"node_positions": npos}}
        return fwd, {"labels": labels}, {}

    def loss(self, logits_window, targets):
        labels = targets["labels"]

        def one(logits):
            return jnp.mean(_ce(logits, jnp.broadcast_to(labels, logits.shape[:-1])))

        return _windowed(one, logits_window)

    def loss_graph(self, logits, targets):  # MPNN graph readout [B,C]
        return jnp.mean(_ce(logits, targets["labels"]))

    def evaluate(self, logits_final, targets, aux):
        agg = jnp.mean(jax.nn.softmax(logits_final, axis=-1), axis=0)  # [B,C]
        return {"acc": float(jnp.mean(jnp.argmax(agg, -1) == targets["labels"]))}

    def evaluate_graph(self, logits, targets, aux):
        return {"acc": float(jnp.mean(jnp.argmax(logits, -1) == targets["labels"]))}


class SudokuTask:
    name = "sudoku"
    num_classes = 10

    def prepare(self, batch):
        inputs = jnp.asarray(batch["inputs"])
        labels = jnp.asarray(batch["labels"])
        B = inputs.shape[0]
        inp, lab = inputs.reshape(B, 9, 9), labels.reshape(B, 9, 9)
        edges, map_u, map_v = V.build_sudoku_multigraph(9)
        cell_ids = V.build_sudoku_cell_indices(9)
        patches = jnp.transpose(
            V.sudoku_slice_batch_jax(jax.nn.one_hot(inp, self.num_classes)), (1, 0, 2, 3)
        )
        slice_labels = jnp.transpose(
            V.sudoku_slice_batch_jax(lab[..., None])[..., 0], (1, 0, 2)
        ).astype(jnp.int32)
        fwd = {
            "patches": patches,
            "edge_indices": edges,
            "model_kwargs": {"map_u": map_u, "map_v": map_v, "cell_ids": cell_ids},
        }
        targets = {
            "slice_labels": slice_labels,
            "labels_grid": lab.astype(jnp.int32),
            "inputs_grid": inp.astype(jnp.int32),
        }
        return fwd, targets, {}

    def loss(self, logits_window, targets):
        labels = targets["slice_labels"]

        def one(logits):
            return jnp.mean(_ce(logits, labels))

        return _windowed(one, logits_window)

    def evaluate(self, logits_final, targets, aux):
        recon = V.reassemble_sudoku_logits(jnp.transpose(logits_final, (1, 0, 2, 3)))
        pred = jnp.argmax(recon, axis=-1)
        labels = targets["labels_grid"]
        empty = targets["inputs_grid"] == 0
        return {
            "cell_acc": float(jnp.mean(pred == labels)),
            "solved": float(jnp.mean(jnp.all(pred == labels, axis=(1, 2)))),
            "completion": float(jnp.sum((pred == labels) * empty) / jnp.maximum(jnp.sum(empty), 1)),
        }


def make_task(name: str, **kwargs):
    tasks = {"maze": MazeTask, "mnist": MNISTTask, "sudoku": SudokuTask}
    if name not in tasks:
        raise KeyError(f"unknown task {name!r}; available: {sorted(tasks)}")
    return tasks[name](**kwargs)
