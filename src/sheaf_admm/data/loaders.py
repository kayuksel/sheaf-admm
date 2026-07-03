"""On-disk dataset loaders: memory-mapped, with format-contract validation.

Two formats:

* :class:`ImageDataset` — classification caches (MNIST). A split directory holds
  ``images.npy`` + ``labels.npy``, and ``metadata.json`` carries per-split shapes
  (so robustness splits with different padding/noise can coexist with the train
  split, e.g. ``test_pad_16``).
* :class:`PuzzleDataset` — TRM-style structured-prediction sets (maze, sudoku). A
  split directory holds ``all__{inputs,labels,puzzle_indices,group_indices,
  puzzle_identifiers}.npy`` + ``dataset.json``. Training samples by *group* (one
  puzzle drawn per group, then rows within that puzzle); evaluation walks rows
  sequentially.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .common import PuzzleDatasetMetadata

# =============================================================================
# Image datasets (classification)
# =============================================================================


class ImageDataset:
    """Memory-mapped loader for cached image classification datasets (e.g. MNIST)."""

    def __init__(self, dataset_dir: str | Path, split: str = "train") -> None:
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.metadata = json.loads((self.dataset_dir / "metadata.json").read_text())

        split_dir = self.dataset_dir / split
        self.images = np.load(split_dir / "images.npy", mmap_mode="r")
        self.labels = np.load(split_dir / "labels.npy", mmap_mode="r")
        self._validate()

    def _validate(self) -> None:
        if self.images.shape[0] != self.labels.shape[0]:
            raise ValueError("images and labels must share the same length on axis 0")

        # Prefer a per-split shape (robustness splits change H/W); fall back to global.
        split_shapes = self.metadata.get("split_shapes", {})
        expected_shape = split_shapes.get(self.split, self.metadata.get("image_shape"))
        if expected_shape is not None:
            expected_shape = tuple(expected_shape)
            if tuple(self.images.shape[1:]) != expected_shape:
                raise ValueError(
                    f"image_shape mismatch for split {self.split!r}: "
                    f"metadata {expected_shape}, observed {tuple(self.images.shape[1:])}"
                )

        expected_count = self.metadata.get("splits", {}).get(self.split)
        if expected_count is not None and expected_count != self.images.shape[0]:
            raise ValueError(
                f"metadata count {expected_count} != observed {self.images.shape[0]} "
                f"for split={self.split}"
            )

    @property
    def num_examples(self) -> int:
        return int(self.images.shape[0])

    def iter_batches(
        self,
        batch_size: int,
        shuffle: bool = False,
        seed: int | None = None,
        drop_last: bool = False,
        rank: int = 0,
        num_replicas: int = 1,
    ) -> Iterator[Mapping[str, np.ndarray]]:
        """Yield ``{"images", "labels"}`` batches, optionally sharded across replicas."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if batch_size % num_replicas != 0:
            raise ValueError("batch_size must be divisible by num_replicas")

        indices = np.arange(self.num_examples, dtype=np.int64)
        if shuffle:
            np.random.default_rng(seed).shuffle(indices)

        local_batch = batch_size // num_replicas
        for start in range(0, self.num_examples, batch_size):
            end = min(start + batch_size, self.num_examples)
            if drop_last and end - start < batch_size:
                break
            local_start = rank * local_batch
            local_end = min(local_start + local_batch, end - start)
            if local_start >= end - start:
                continue
            batch_indices = indices[start:end][local_start:local_end]
            yield {
                "images": self.images[batch_indices],
                "labels": self.labels[batch_indices],
            }


# =============================================================================
# Puzzle datasets (structured prediction)
# =============================================================================


@dataclass
class PuzzleSet:
    """One named set within a split, with TRM cumulative-index bookkeeping."""

    inputs: np.ndarray
    labels: np.ndarray
    puzzle_indices: np.ndarray  # [P+1] cumulative row offsets per puzzle
    group_indices: np.ndarray  # [G+1] cumulative puzzle offsets per group
    puzzle_identifiers: np.ndarray  # [P] identifier per puzzle

    @property
    def num_examples(self) -> int:
        return int(self.inputs.shape[0])

    @property
    def num_puzzles(self) -> int:
        return int(self.puzzle_indices.shape[0] - 1)

    @property
    def num_groups(self) -> int:
        return int(self.group_indices.shape[0] - 1)

    def validate(self, seq_len: int, label_seq_len: int | None = None) -> None:
        if label_seq_len is None:
            label_seq_len = seq_len
            if self.inputs.shape != self.labels.shape:
                raise ValueError("inputs and labels must share shape")
        elif self.inputs.shape[0] != self.labels.shape[0]:
            raise ValueError("inputs and labels must have the same batch size")
        if self.inputs.ndim != 2:
            raise ValueError(f"expected 2D inputs, got shape {self.inputs.shape}")
        if self.inputs.shape[1] != seq_len:
            raise ValueError(f"input seq_len {self.inputs.shape[1]} != expected {seq_len}")
        if self.labels.shape[1] != label_seq_len:
            raise ValueError(f"label seq_len {self.labels.shape[1]} != expected {label_seq_len}")

        _validate_cumulative(self.puzzle_indices, self.num_examples, "puzzle_indices")
        _validate_cumulative(self.group_indices, self.num_puzzles, "group_indices")
        if self.puzzle_identifiers.shape != (self.num_puzzles,):
            raise ValueError(
                f"puzzle_identifiers shape {self.puzzle_identifiers.shape} "
                f"!= num_puzzles {self.num_puzzles}"
            )


def _validate_cumulative(indices: np.ndarray, expected_last: int, name: str) -> None:
    if indices.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {indices.shape}")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"{name} must be integer typed, got {indices.dtype}")
    if indices[0] != 0:
        raise ValueError(f"{name} must start at 0, got {indices[0]}")
    if np.any(np.diff(indices) < 0):
        raise ValueError(f"{name} must be monotone nondecreasing")
    if indices[-1] != expected_last:
        raise ValueError(f"{name} last offset {indices[-1]} != expected {expected_last}")


def _puzzle_ids_for_rows(puzzle_indices: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return np.searchsorted(puzzle_indices, rows, side="right") - 1


def _sample_batch(
    rng: np.random.Generator,
    group_order: np.ndarray,
    puzzle_indices: np.ndarray,
    group_indices: np.ndarray,
    start_index: int,
    global_batch_size: int,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Fill one global batch by walking ``group_order``: one random puzzle per group."""
    batch_rows: list[np.ndarray] = []
    batch_puzzle_ids: list[np.ndarray] = []
    current_size = 0

    while start_index < group_order.size and current_size < global_batch_size:
        group_id = int(group_order[start_index])
        start_index += 1
        puzzle_id = int(rng.integers(group_indices[group_id], group_indices[group_id + 1]))
        row_start = int(puzzle_indices[puzzle_id])
        row_end = int(puzzle_indices[puzzle_id + 1])
        puzzle_size = row_end - row_start
        take = min(puzzle_size, global_batch_size - current_size)
        if take > 0:
            batch_rows.append(row_start + rng.choice(puzzle_size, take, replace=False))
            batch_puzzle_ids.append(np.full(take, puzzle_id, dtype=np.int64))
            current_size += take

    if not batch_rows:
        return start_index, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return start_index, np.concatenate(batch_rows), np.concatenate(batch_puzzle_ids)


class PuzzleDataset:
    """Memory-mapped loader for TRM-style puzzle datasets (maze, sudoku)."""

    def __init__(self, dataset_dir: str | Path, split: str = "train") -> None:
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.metadata = PuzzleDatasetMetadata.from_json(self.dataset_dir / split / "dataset.json")
        self.sets: dict[str, PuzzleSet] = {}
        self._train_call = 0
        self._load_sets()
        self._validate_metadata_totals()

    def _load_sets(self) -> None:
        field_modes: Mapping[str, str | None] = {
            "inputs": "r",
            "labels": "r",
            "puzzle_indices": None,
            "group_indices": None,
            "puzzle_identifiers": None,
        }
        split_dir = self.dataset_dir / self.split

        for set_name in self.metadata.sets:
            arrays = {
                field: np.load(split_dir / f"{set_name}__{field}.npy", mmap_mode=mmap_mode)
                for field, mmap_mode in field_modes.items()
            }
            puzzle_set = PuzzleSet(**arrays)
            puzzle_set.validate(seq_len=self.metadata.seq_len)
            self.sets[set_name] = puzzle_set

    def _validate_metadata_totals(self) -> None:
        total_puzzles = sum(s.num_puzzles for s in self.sets.values())
        total_groups = sum(s.num_groups for s in self.sets.values())
        total_examples = sum(s.num_examples for s in self.sets.values())

        if total_puzzles != self.metadata.total_puzzles:
            raise ValueError(
                f"metadata.total_puzzles={self.metadata.total_puzzles} != observed {total_puzzles}"
            )
        if total_groups != self.metadata.total_groups:
            raise ValueError(
                f"metadata.total_groups={self.metadata.total_groups} != observed {total_groups}"
            )
        expected_mean = total_examples / max(total_puzzles, 1)
        if not np.isclose(expected_mean, self.metadata.mean_puzzle_examples):
            raise ValueError(
                f"metadata.mean_puzzle_examples={self.metadata.mean_puzzle_examples} "
                f"!= observed {expected_mean}"
            )
        for set_name, dataset in self.sets.items():
            if dataset.puzzle_identifiers.size == 0:
                continue
            max_id = int(dataset.puzzle_identifiers.max())
            if max_id >= self.metadata.num_puzzle_identifiers:
                raise ValueError(
                    f"{set_name} puzzle_identifier max {max_id} exceeds "
                    f"metadata.num_puzzle_identifiers"
                )

    def _metadata_batch_fields(self) -> dict[str, int]:
        out: dict[str, int] = {}
        if self.metadata.height is not None:
            out["height"] = int(self.metadata.height)
        if self.metadata.width is not None:
            out["width"] = int(self.metadata.width)
        return out

    def iter_test_batches(
        self, batch_size: int, rank: int = 0, num_replicas: int = 1
    ) -> Iterator[tuple[str, Mapping[str, np.ndarray]]]:
        """Walk every set's rows sequentially, yielding ``(set_name, batch)``."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if batch_size % num_replicas != 0:
            raise ValueError("batch_size must be divisible by num_replicas")

        local_batch = batch_size // num_replicas
        for set_name, dataset in self.sets.items():
            total = dataset.num_examples
            for start in range(0, total, batch_size):
                end = min(total, start + batch_size)
                local_start = start + rank * local_batch
                local_end = min(end, local_start + local_batch)
                if local_start >= end:
                    continue
                rows = np.arange(local_start, local_end, dtype=np.int64)
                puzzle_ids = _puzzle_ids_for_rows(dataset.puzzle_indices, rows)
                yield (
                    set_name,
                    {
                        "inputs": dataset.inputs[rows],
                        "labels": dataset.labels[rows],
                        "puzzle_ids": puzzle_ids.astype(np.int32),
                        "puzzle_identifiers": dataset.puzzle_identifiers[puzzle_ids],
                        **self._metadata_batch_fields(),
                    },
                )

    def iter_train_batches(
        self,
        batch_size: int,
        seed: int,
        epochs_per_iter: int = 1,
        rank: int = 0,
        num_replicas: int = 1,
    ) -> Iterator[tuple[str, Mapping[str, np.ndarray]]]:
        """Group-sampled training batches: one random puzzle per group, full batches only."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if epochs_per_iter < 1:
            raise ValueError("epochs_per_iter must be >= 1")
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if batch_size % num_replicas != 0:
            raise ValueError("batch_size must be divisible by num_replicas")

        self._train_call += 1
        local_batch = batch_size // num_replicas
        rng = np.random.default_rng(seed + self._train_call)

        for set_name, dataset in self.sets.items():
            group_order = np.concatenate(
                [rng.permutation(dataset.num_groups) for _ in range(int(epochs_per_iter))]
            )
            start_index = 0
            while start_index < group_order.size:
                start_index, rows, puzzle_ids = _sample_batch(
                    rng,
                    group_order=group_order,
                    puzzle_indices=dataset.puzzle_indices,
                    group_indices=dataset.group_indices,
                    start_index=start_index,
                    global_batch_size=batch_size,
                )
                if rows.size < batch_size:
                    break
                keep = np.arange(rows.size, dtype=np.int64)[
                    rank * local_batch : (rank + 1) * local_batch
                ]
                batch_rows, batch_puzzle_ids = rows[keep], puzzle_ids[keep]
                yield (
                    set_name,
                    {
                        "inputs": dataset.inputs[batch_rows],
                        "labels": dataset.labels[batch_rows],
                        "puzzle_ids": batch_puzzle_ids.astype(np.int32),
                        "puzzle_identifiers": dataset.puzzle_identifiers[batch_puzzle_ids],
                        **self._metadata_batch_fields(),
                    },
                )
