"""Sudoku builder: download HF ``Ritvik19/Sudoku-Dataset``, take first N, augment.

The dataset (~2.2 GB parquet, ~17M rows) is downloaded once and iterated locally
(HF streaming was much slower in replication runs), filtered to
``difficulty <= difficulty_max`` (default 2.0 — note ``difficulty`` is binary 0/1
in this dataset, so the filter passes *everything*: "easy" is effectively "the first
N puzzles"), and each kept puzzle is parsed from its
81-char string into a 9x9 token grid (``'.' -> 0``, ``'1'-'9' -> 1-9``; vocab 10).
Train and test draw from one shared iterator so the splits never overlap; the test
split is named ``test_hard``. Train puzzles are written under all 8 dihedral
symmetries.

On-disk format is the TRM standard token layout (``seq_len = 81``, vocab 10).

Usage:
    python -m sheaf_admm.data.build_sudoku --output-dir datasets/sudoku_easy \
        --train-size 50000 --test-size 2000 --difficulty-max 2.0
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    SUDOKU_VOCAB_SIZE,
    PuzzleDatasetMetadata,
    dihedral_transform,
    save_npy,
)


@dataclass
class SudokuConfig:
    dataset: str = "Ritvik19/Sudoku-Dataset"
    height: int = 9
    width: int = 9
    seed: int = 0
    train_size: int = 50000
    test_size: int = 2000
    difficulty_max: float = 2.0
    train_augment: bool = True
    test_augment: bool = False
    output_dir: Path = Path("datasets/sudoku_easy")

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload


def _char_to_token(ch: str) -> int:
    return 0 if ch == "." else int(ch)


def parse_sudoku(text: str, height: int = 9, width: int = 9) -> np.ndarray:
    """Parse an 81-char Sudoku string into a 9x9 int8 token grid."""
    expected = height * width
    if len(text) != expected:
        raise ValueError(f"expected {expected} chars, got {len(text)}")
    grid = np.zeros((height, width), dtype=np.int8)
    for i, ch in enumerate(text):
        r, c = divmod(i, width)
        grid[r, c] = _char_to_token(ch)
    return grid


def _build_split_from_puzzles(
    split: str,
    puzzles: Iterable[tuple[np.ndarray, np.ndarray]],
    cfg: SudokuConfig,
    apply_augmentation: bool,
) -> None:
    """Write a split from an iterable of ``(input_grid, label_grid)`` 9x9 token pairs."""
    output_root = cfg.output_dir / split
    inputs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    puzzle_indices: list[int] = [0]
    group_indices: list[int] = [0]
    puzzle_identifiers: list[int] = []
    total_examples = 0
    num_puzzles = 0
    aug_variants = list(range(8)) if apply_augmentation else [0]

    for base_input, base_label in puzzles:
        for k in aug_variants:
            input_grid = base_input if k == 0 else dihedral_transform(base_input, k)
            label_grid = base_label if k == 0 else dihedral_transform(base_label, k)
            inputs.append(input_grid.astype(np.uint8).reshape(-1))
            labels.append(label_grid.astype(np.uint8).reshape(-1))
            total_examples += 1
        num_puzzles += 1
        puzzle_indices.append(total_examples)
        group_indices.append(num_puzzles)
        puzzle_identifiers.append(0)

    inputs_arr = np.stack(inputs)
    labels_arr = np.stack(labels)
    save_npy(output_root / "all__inputs.npy", inputs_arr)
    save_npy(output_root / "all__labels.npy", labels_arr)
    save_npy(output_root / "all__puzzle_indices.npy", np.array(puzzle_indices, dtype=np.int32))
    save_npy(output_root / "all__group_indices.npy", np.array(group_indices, dtype=np.int32))
    save_npy(
        output_root / "all__puzzle_identifiers.npy", np.array(puzzle_identifiers, dtype=np.int32)
    )

    metadata = PuzzleDatasetMetadata(
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        vocab_size=SUDOKU_VOCAB_SIZE,
        seq_len=cfg.height * cfg.width,
        num_puzzle_identifiers=1,
        total_groups=num_puzzles,
        mean_puzzle_examples=total_examples / max(num_puzzles, 1),
        total_puzzles=num_puzzles,
        sets=["all"],
        height=cfg.height,
        width=cfg.width,
    )
    metadata.write(output_root / "dataset.json")
    print(
        json.dumps(
            {
                "split": split,
                "examples": int(inputs_arr.shape[0]),
                "puzzles": int(num_puzzles),
                "augmented": apply_augmentation,
            },
            indent=2,
        )
    )


def _stream_puzzles(
    stream_iterator: Any, cfg: SudokuConfig, max_count: int
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Yield up to ``max_count`` parsed ``(input, label)`` grids passing the filter."""
    found = 0
    for example in stream_iterator:
        if found >= max_count:
            break
        diff = example.get("difficulty")
        try:
            if diff is None or float(diff) > cfg.difficulty_max:
                continue
        except (ValueError, TypeError):
            continue
        yield (
            parse_sudoku(example["puzzle"], cfg.height, cfg.width),
            parse_sudoku(example["solution"], cfg.height, cfg.width),
        )
        found += 1


def build(cfg: SudokuConfig) -> None:
    """Stream the HF dataset and write the train / ``test_hard`` splits."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("the 'datasets' library is required: pip install datasets") from exc

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    # Non-streaming: download once (~2.2 GB) then iterate the Arrow-backed split
    # locally. HF streaming took tens of minutes for a few hundred puzzles here;
    # this materializes in about a minute and then iterates instantly.
    stream_iter = iter(load_dataset(cfg.dataset, split="train"))

    _build_split_from_puzzles(
        "train", _stream_puzzles(stream_iter, cfg, cfg.train_size), cfg, cfg.train_augment
    )
    _build_split_from_puzzles(
        "test_hard", _stream_puzzles(stream_iter, cfg, cfg.test_size), cfg, cfg.test_augment
    )
    (cfg.output_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    print(f"Done. Dataset saved to {cfg.output_dir}")


def default_config() -> SudokuConfig:
    return SudokuConfig()


def parse_args() -> SudokuConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=SudokuConfig.output_dir)
    parser.add_argument("--train-size", type=int, default=SudokuConfig.train_size)
    parser.add_argument("--test-size", type=int, default=SudokuConfig.test_size)
    parser.add_argument("--difficulty-max", type=float, default=SudokuConfig.difficulty_max)
    parser.add_argument("--no-train-augment", action="store_true")
    parser.add_argument("--test-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=SudokuConfig.seed)
    args = parser.parse_args()
    return SudokuConfig(
        output_dir=args.output_dir,
        train_size=args.train_size,
        test_size=args.test_size,
        difficulty_max=args.difficulty_max,
        train_augment=not args.no_train_augment,
        test_augment=args.test_augment,
        seed=args.seed,
    )


if __name__ == "__main__":
    build(parse_args())
