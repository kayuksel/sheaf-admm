"""Framework-agnostic data primitives: tokens, dihedral symmetries, on-disk metadata.

These are the pieces shared by the dataset builders and the on-disk format
contract that the loaders validate against. Everything here is plain NumPy /
dataclasses — no JAX — because it runs at dataset-build and disk-IO time.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Maze token convention. Inputs carry 1-4; labels add 5 (path). vocab_size = 6
# (slot 0 is the pad/ignore id).
TOKEN_IDS: Mapping[str, int] = {
    "wall": 1,
    "empty": 2,
    "start": 3,
    "goal": 4,
    "path": 5,
}

# Sudoku token convention: '.' -> 0 (empty), '1'-'9' -> 1-9.
SUDOKU_VOCAB_SIZE = 10

# The 8 dihedral symmetries and their inverses (mirrors the TRM datasets).
DIHEDRAL_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]


def dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """Apply the ``tid``-th dihedral symmetry (rotations, flips, diagonal mirrors)."""
    if tid == 0:
        return arr  # identity
    if tid == 1:
        return np.rot90(arr, k=1)
    if tid == 2:
        return np.rot90(arr, k=2)
    if tid == 3:
        return np.rot90(arr, k=3)
    if tid == 4:
        return np.fliplr(arr)  # horizontal flip
    if tid == 5:
        return np.flipud(arr)  # vertical flip
    if tid == 6:
        return arr.T  # main-diagonal reflection
    if tid == 7:
        return np.fliplr(np.rot90(arr, k=1))  # anti-diagonal reflection
    raise ValueError(f"Unsupported dihedral transform index: {tid}")


def inverse_dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """Undo :func:`dihedral_transform`."""
    return dihedral_transform(arr, DIHEDRAL_INVERSE[int(tid)])


@dataclass
class PuzzleDatasetMetadata:
    """On-disk metadata for a TRM-style puzzle split (``dataset.json``).

    Puzzle inputs and labels use one token per cell, so ``seq_len = H*W`` for
    grid-shaped puzzle datasets.
    """

    pad_id: int
    ignore_label_id: int | None
    blank_identifier_id: int
    vocab_size: int
    seq_len: int
    num_puzzle_identifiers: int
    total_groups: int
    mean_puzzle_examples: float
    total_puzzles: int
    sets: Sequence[str]
    height: int | None = None
    width: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @staticmethod
    def from_json(path: Path) -> PuzzleDatasetMetadata:
        payload = json.loads(Path(path).read_text())
        # Older internal standard-token datasets carried these now-unused keys.
        payload.pop("input_format", None)
        payload.pop("num_input_channels", None)
        return PuzzleDatasetMetadata(**payload)


def save_npy(path: Path, array: np.ndarray) -> None:
    """Save a ``.npy`` array, creating the parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Save a JSON payload, creating the parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def shuffle_in_unison(rng: np.random.Generator, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Shuffle a set of arrays along axis 0 with one shared permutation."""
    if not arrays:
        return ()
    length = arrays[0].shape[0]
    for arr in arrays:
        if arr.shape[0] != length:
            raise ValueError("All arrays must share the same length on axis 0.")
    perm = rng.permutation(length)
    return tuple(arr[perm] for arr in arrays)
