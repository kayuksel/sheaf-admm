"""Perfect-maze builder: DFS carve, BFS shortest path, dihedral augmentation.

A "perfect" maze (unique path between any two cells) is carved by randomized
DFS backtracking. Start/goal are sampled from open cells; the shortest path is
found by BFS and accepted only if ``len(path) - 1 >= min_path_length`` (the
default 18 matches the paper's 19x19 ``std3`` sets). Each accepted base maze is
written under all admissible dihedral symmetries (square mazes use all 8;
rectangles use the 4 that preserve the aspect ratio).

On-disk format is the TRM standard token layout (one int per cell, vocab 6):
``inputs`` carry wall/empty/start/goal (1-4), ``labels`` add the path token (5).

Size generalization is built in: ``--ood-sizes`` writes the train + standard
OOD suite (1.5x/2x/4x larger boards, with ``_long`` variants demanding longer
paths) in one run.

Usage:
    python -m sheaf_admm.data.build_maze --height 19 --width 19 \
        --min-path-length 18 --train-size 10000 --test-size 1000 \
        --output-dir datasets/maze_std3_19px_10k --ood-sizes
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from .common import (
    TOKEN_IDS,
    PuzzleDatasetMetadata,
    dihedral_transform,
    save_npy,
)


@dataclass
class MazeConfig:
    height: int = 19
    width: int = 19
    train_size: int = 10000
    test_size: int = 1000
    min_path_length: int = 18
    train_augment: bool = True
    test_augment: bool = False
    seed: int = 0
    max_retries: int = 200
    output_dir: Path = Path("datasets/maze")

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload


# --- maze generation ---------------------------------------------------------


def _neighbors(cell: tuple[int, int]) -> Iterable[tuple[int, int]]:
    r, c = cell
    for dr, dc in ((2, 0), (-2, 0), (0, 2), (0, -2)):
        yield r + dr, c + dc


def _backtrack_maze(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """Carve a perfect maze by randomized DFS on the odd-index lattice."""
    grid = np.full((height, width), TOKEN_IDS["wall"], dtype=np.int8)
    odd_rows = list(range(1, height, 2))
    odd_cols = list(range(1, width, 2))
    start_r, start_c = int(rng.choice(odd_rows)), int(rng.choice(odd_cols))
    grid[start_r, start_c] = TOKEN_IDS["empty"]
    stack = [(start_r, start_c)]
    while stack:
        r, c = stack[-1]
        unvisited = [
            (nr, nc)
            for nr, nc in _neighbors((r, c))
            if 0 < nr < height
            and 0 < nc < width
            and grid[nr, nc] == TOKEN_IDS["wall"]
            and nr in odd_rows
            and nc in odd_cols
        ]
        if not unvisited:
            stack.pop()
            continue
        nr, nc = unvisited[int(rng.integers(len(unvisited)))]
        grid[r + (nr - r) // 2, c + (nc - c) // 2] = TOKEN_IDS["empty"]
        grid[nr, nc] = TOKEN_IDS["empty"]
        stack.append((nr, nc))
    return grid


def _shortest_path(
    grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> list[tuple[int, int]]:
    """BFS shortest path over open cells; empty list if unreachable."""
    h, w = grid.shape
    queue: list[tuple[int, int]] = [start]
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    visited = np.zeros_like(grid, dtype=bool)
    visited[start] = True
    for pos in queue:
        if pos == goal:
            break
        r, c = pos
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < h
                and 0 <= nc < w
                and not visited[nr, nc]
                and grid[nr, nc] != TOKEN_IDS["wall"]
            ):
                visited[nr, nc] = True
                parent[(nr, nc)] = (r, c)
                queue.append((nr, nc))
    if goal not in parent and start != goal:
        return []
    path = [goal]
    cur = goal
    while cur != start:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path


def _pick_positions(
    grid: np.ndarray, rng: np.random.Generator
) -> tuple[tuple[int, int], tuple[int, int]]:
    empties = np.argwhere(grid == TOKEN_IDS["empty"])
    if len(empties) < 2:
        raise ValueError("Maze too small for start/goal placement.")
    start_idx, goal_idx = rng.choice(len(empties), size=2, replace=False)
    return tuple(int(x) for x in empties[start_idx]), tuple(int(x) for x in empties[goal_idx])


def _paint_grid(
    base: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    path: Sequence[tuple[int, int]],
    with_path: bool,
) -> np.ndarray:
    """Paint start/goal (and optionally the path) onto a copy of the maze."""
    painted = base.copy()
    (sr, sc), (gr, gc) = start, goal
    if with_path:
        for r, c in path:
            painted[r, c] = TOKEN_IDS["path"]
    painted[sr, sc] = TOKEN_IDS["start"]
    painted[gr, gc] = TOKEN_IDS["goal"]
    return painted


# --- split assembly ----------------------------------------------------------


def _dihedral_variants(height: int, width: int, augment: bool) -> list[int]:
    if not augment:
        return [0]
    if height == width:
        return list(range(8))
    return [0, 2, 4, 5]  # identity, 180-rot, hflip, vflip (preserve aspect ratio)


def _build_split(
    split: str,
    count: int,
    cfg: MazeConfig,
    rng: np.random.Generator,
    seen_hashes: set,
    apply_augmentation: bool,
) -> None:
    output_root = cfg.output_dir / split
    inputs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    puzzle_indices: list[int] = [0]
    group_indices: list[int] = [0]
    puzzle_identifiers: list[int] = []
    total_examples = 0
    aug_variants = _dihedral_variants(cfg.height, cfg.width, apply_augmentation)

    for puzzle_id in range(count):
        base_grid = start = goal = path = None
        for _ in range(cfg.max_retries * 4):
            grid = _backtrack_maze(cfg.height, cfg.width, rng)
            grid_hash = hash(grid.tobytes())
            if grid_hash in seen_hashes:
                continue
            s, g = _pick_positions(grid, rng)
            p = _shortest_path(grid, s, g)
            if p and len(p) - 1 >= cfg.min_path_length:
                seen_hashes.add(grid_hash)
                base_grid, start, goal, path = grid, s, g, p
                break
        if base_grid is None:
            raise RuntimeError(
                f"Failed to generate maze with min_path_length={cfg.min_path_length}"
            )

        for k in aug_variants:
            input_grid = _paint_grid(base_grid, start, goal, path, with_path=False)
            label_grid = _paint_grid(base_grid, start, goal, path, with_path=True)
            if k != 0:
                input_grid = dihedral_transform(input_grid, k)
                label_grid = dihedral_transform(label_grid, k)
            inputs.append(input_grid.astype(np.uint8).reshape(-1))
            labels.append(label_grid.astype(np.uint8).reshape(-1))
            total_examples += 1

        puzzle_indices.append(total_examples)
        group_indices.append(puzzle_id + 1)
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
        vocab_size=max(TOKEN_IDS.values()) + 1,
        seq_len=cfg.height * cfg.width,
        num_puzzle_identifiers=1,
        total_groups=len(group_indices) - 1,
        mean_puzzle_examples=total_examples / max(count, 1),
        total_puzzles=count,
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
                "puzzles": int(count),
                "input_seq_len": int(inputs_arr.shape[1]),
            },
            indent=2,
        )
    )


# --- spec-builder API (used by build_dataset specs) --------------------------


def default_config() -> MazeConfig:
    return MazeConfig()


def build_split(
    cfg: MazeConfig,
    split: str,
    count: int,
    rng: np.random.Generator | None = None,
    seen_hashes: set | None = None,
    apply_augmentation: bool | None = None,
) -> None:
    if rng is None:
        rng = np.random.default_rng(cfg.seed)
    if seen_hashes is None:
        seen_hashes = set()
    if apply_augmentation is None:
        apply_augmentation = cfg.train_augment if split == "train" else cfg.test_augment
    _build_split(split, count, cfg, rng, seen_hashes, apply_augmentation)


# --- standard OOD size-generalization suite ----------------------------------

# (split, height, width, min_path_length, seed) relative to a 19x19 base — the
# paper's std3 size-generalization grid. Scaled boards keep ~1-cell border via
# odd dimensions; ``_long`` variants demand longer paths at the same size.
_OOD_SUITE: list[tuple[str, int, int, int, int]] = [
    ("test_ood_1.5x", 29, 29, 42, 2000),
    ("test_ood_2x", 37, 37, 36, 3500),
    ("test_ood_2x_long", 37, 37, 54, 3600),
    ("test_ood_2xW", 19, 37, 35, 3000),
    ("test_ood_4xW", 19, 73, 69, 4000),
    ("test_ood_4x", 73, 73, 72, 5000),
    ("test_ood_4x_long", 73, 73, 109, 6000),
]


def build(cfg: MazeConfig, ood_sizes: bool = False) -> None:
    """Build the train split, the same-size test split, and optionally the OOD suite."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    seen_hashes: set = set()

    build_split(
        cfg,
        "train",
        cfg.train_size,
        rng=np.random.default_rng(cfg.seed),
        seen_hashes=seen_hashes,
        apply_augmentation=cfg.train_augment,
    )
    build_split(
        cfg,
        "test",
        cfg.test_size,
        rng=np.random.default_rng(cfg.seed + 1000),
        seen_hashes=seen_hashes,
        apply_augmentation=cfg.test_augment,
    )

    if ood_sizes:
        for name, h, w, mpl, seed in _OOD_SUITE:
            ood_cfg = replace(cfg, height=h, width=w, min_path_length=mpl, seed=seed)
            build_split(
                ood_cfg,
                name,
                cfg.test_size,
                rng=np.random.default_rng(seed),
                seen_hashes=set(),
                apply_augmentation=False,
            )
    print("Done.")


def parse_args() -> tuple[MazeConfig, bool]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=MazeConfig.height)
    parser.add_argument("--width", type=int, default=MazeConfig.width)
    parser.add_argument("--train-size", type=int, default=MazeConfig.train_size)
    parser.add_argument("--test-size", type=int, default=MazeConfig.test_size)
    parser.add_argument("--min-path-length", type=int, default=MazeConfig.min_path_length)
    parser.add_argument("--seed", type=int, default=MazeConfig.seed)
    parser.add_argument("--max-retries", type=int, default=MazeConfig.max_retries)
    parser.add_argument("--output-dir", type=Path, default=MazeConfig.output_dir)
    parser.add_argument("--no-train-augment", action="store_true")
    parser.add_argument("--test-augment", action="store_true")
    parser.add_argument(
        "--ood-sizes",
        action="store_true",
        help="Also build the standard size-generalization OOD suite.",
    )
    args = parser.parse_args()
    cfg = MazeConfig(
        height=args.height,
        width=args.width,
        train_size=args.train_size,
        test_size=args.test_size,
        min_path_length=args.min_path_length,
        train_augment=not args.no_train_augment,
        test_augment=args.test_augment,
        seed=args.seed,
        max_retries=args.max_retries,
        output_dir=args.output_dir,
    )
    return cfg, args.ood_sizes


if __name__ == "__main__":
    _cfg, _ood = parse_args()
    build(_cfg, ood_sizes=_ood)
